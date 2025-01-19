import logging
import os
import uuid
from multiprocessing import Pool, Queue, Process, Value

import numpy as np

from battle_tower_agent import (
    BattleTowerAgent,
    BattleTowerAAgent,
    TowerState,
    POKEMON_MAX_MOVES,
    NUM_POKEMON_IN_SINGLES,
    in_battle,
    ROM_DIR,
    check_key_pixels,
    won_set,
    lost_set,
    pokemon_is_fainted,
    get_party_status, in_move_select, is_next_opponent_box, at_save_battle_video,
)

from battle_tower_search_agent import InvalidMoveSelected

from battle_tower_database.interface import BattleTowerDBInterface, BattleTowerServerDBInterface

DEFAULT_MOVE = 0
SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search Team.dst')
SEARCH_TMP_SAVESTATE_DIR = os.path.join(ROM_DIR, 'search')

logger = logging.getLogger('SearchTowerAgent')
logging.basicConfig(level=logging.DEBUG)

HP_PER_POKEMON = 100

def opp_pokemon_is_out(frame):
    """
    This function checks whether the opponent's pokemon is fully out (i.e. that we're in battle,
    there is a name/HP bar, and that it isn't fainted/we're waiting for the next pokemon
    """
    # There's no good "one" check for the opp's bar, so I am checking various key pixels
    # (mostly at the far right b/c the name bar slides to the left when a pokemon faints)
    key_pixels = [
        ((44, 117), (40, 48, 40)), # far right sticking out dark pixel in the HP arrow bar
        ((45, 120), (96, 72, 56)), # far right sticking out slightly brigher pixel in arrow bar
        ((22, 108), (40, 48, 40)), # upper right gray corner of opp pkmn box
        ((50, 108), (40, 48, 40)), # lower right gray corner of opp pkmn box
    ]

    return check_key_pixels(frame, key_pixels)


def get_opponent_hp_bar(frame):
    """This gets the opponent's Pokemon's current HP (as an integer from 0s to 100)"""
    # while the bar technically has multiple rows, we only need the first (which makes indexing easier)
    # also this *only* captures the HP bar, nothing surrounding it
    hp_bar = frame[43, 50:98, :]

    # any missing HP is black, easier to check for missing HP than current HP (b/c current can be red, yellow, or green)
    missing_hp_color = np.array([0,0,0])
    # NOTE: we can't just do hp_bar != missing_hp_color b/c some of the BGR values are 0, meaning we'd get false positives
    # so instead we calculate the # of pixels that are black
    remaining_hp_bar = np.any(hp_bar != missing_hp_color, axis=-1).sum()
    remaining_hp = int(remaining_hp_bar / hp_bar.shape[0] * 100)

    return remaining_hp

def get_opponent_pokemon_name(frame):
    """This gets the part of the frame containing the pokemon's name, gender, and level"""
    name = frame[27:37, 2:98, :]
    return name

class SwappedPokemon(Exception):
    pass


class HPWatcher:
    # NOTE: this class is reliant on the fact that each Pkmn on the foes team is unique b/c of the way it tracks pokemon switches
    def __init__(self, damage_value: Value = None):
        self.opp_hp = None
        self.damage_dealt = 0
        self.opp_pkm = None

        # this is an optional multiprocessing value useful for keeping track of every search agent
        self.damage_value = damage_value

    def __call__(self, cur_frame):
        if opp_pokemon_is_out(cur_frame):
            opp_hp = get_opponent_hp_bar(cur_frame)
            opp_pkmn = get_opponent_pokemon_name(cur_frame)

            if self.opp_pkm is None or (opp_pkmn != self.opp_pkm).any():
                self.opp_pkm = opp_pkmn
                self.opp_hp = opp_hp
            else:
                hp_diff = self.opp_hp - opp_hp
                self.damage_dealt += hp_diff
                self.opp_hp = opp_hp

                # minor optimization; HP changes every, say, 3 frames, so we don't need to acquire the lock if the opp's HP doesn't change this frame
                if self.damage_value is not None and hp_diff != 0:
                    with self.damage_value.get_lock():
                        self.damage_value.value += hp_diff

        # since we're kinda hacking `wait_for`, it needs to return a bool to indicate that the check failed
        return False

class BattleTowerSearchV2SubAgent(BattleTowerAgent):
    """This class is used by the BattleTowerSearchAgent to 'look ahead' for the next possible moves"""

    strategy = 'move_select'
    def __init__(self, savestate_file: str, moves: int | list[int], swap_to: int | None = None, damage_value: Value = None):
        super().__init__(render=False, savestate_file=savestate_file)

        if isinstance(moves, int):
            moves = [moves]

        self.moves = moves
        self.move_idx = 0
        self.team = '' # there is no logging to a DB for these searches, so we don't need to specify a teams
        self.swap_to = swap_to

        self.hp_watcher = HPWatcher(damage_value=damage_value)

    def play_remainder_of_battle(self) -> TowerState:
        """
        This function starts from the move_select or pokemon_select screen and plays until one of these stopping conditions:
        1. we go to Pokemon select (either through U-turn/the like or through fainting)
        2. the battle ends in a win
        3. we fail to select a move that is included in the search (but not the 'default' move)

        This is expected to be called in either _select_and_execute_move function
            or the _select_pokemon function of the SearchAgent and thus expects it to be in one of those states.

        Returns the state after finishing the battle (same as `play_battle`)
        """
        self.move_idx = 0

        if self.swap_to and pokemon_is_fainted(self.cur_frame):
            # we can hit the right arrow until we hover over the correct slot
            self._general_button_press(['RIGHT'] * self.cur_frame)
            self._general_button_press('A')

        # if we didn't swap pokemon, we're in move_select, but _run_battle_loop expects us to be in the fight screen, we have to do that;
        # it's slightly wasteful, but most games take thousands of frames, and it only costs us about 20 total so w/e
        self.state = self._wait_for(
            (in_battle, TowerState.BATTLE),
            button_press='B'
        )
        return super()._run_battle_loop()

    def _wait_for_battle_states(self):
        """
        Whenever we're in a battle, these are the possible states we could reach after clicking any move
        This is a special case of _wait_for that we'll tend to use in the battle loop
        """
        return self._wait_for(
            (self.hp_watcher, TowerState.WAITING), # NOTE: I'm doing this *before* any other checks b/c once a check is found, no other checks are run
            (in_battle, TowerState.BATTLE),
            (in_move_select, TowerState.MOVE_SELECT),
            (pokemon_is_fainted, TowerState.SWAP_POKEMON),
            (is_next_opponent_box, TowerState.WON_BATTLE),
            (at_save_battle_video, TowerState.END_OF_SET),
            button_press='A',
            # since we need some more advanced logic, I don't want anything advancing automatically
            # NOTE: DON'T CHANGE THIS OR ELSE IT CAUSES A REALLY TRICKY BUG WHEN WAITING FOR BOTH BATTLE AND MOVE_SELECT
            check_first=True,
        )

    def _select_and_execute_move(self) -> TowerState:
        # The search subagent starts by making each move in-order, and once we've gotten past the moves that we
        #  want to search over, we go back to using the 'default' move (i.e. the first one, which is as we saw with the 'A' agent, is pretty solid)
        state = self.state

        if self.move_idx < len(self.moves):
            move = self.moves[self.move_idx]
        else:
            move = DEFAULT_MOVE

        advanced_game = False

        # There's one slight snag, we may or may not be able to select the move (e.g. due to torment, choice specs)
        #  but you are *still* in move select, unlike certain other conditions
        # There's no (good) way to know until after we click it, so we've just got to keep trying until we get it
        # It's a tad inefficient, but it *is* compatible w/ searching and choice moves b/c:
        # 1. if we are searching over a set of moves that are different, we break
        # 2. if we are searching the same move consecutively, then even if it isn't the first move, we'll eventually goto the choice selected move.
        for i in range(POKEMON_MAX_MOVES):
            chosen_move = move + i
            self._goto_move(chosen_move)

            self._general_button_press('A')
            state = self._wait_for_battle_states()

            # if clicking on one of the moves that *we're exploring* (not the 'default' move after the fact b/c of choice or encore or whatever)
            #   didn't advance the game there is no point in continuing to search down that path so we raise an error
            # NOTE: make sure we catch these error elsewhere
            if state == TowerState.MOVE_SELECT and self.move_idx < len(self.moves):
                logger.debug(f"Attempted to search over move {chosen_move} but it could not be chosen; stopping search.")
                raise InvalidMoveSelected()
            elif state == TowerState.SWAP_POKEMON:
                logger.debug(f'Finished move search by swapping Pokemon. Current Pokemon did {self.hp_watcher.damage_dealt} damage.')
                raise SwappedPokemon()
            # any other state but MOVE_SELECT means that the move 'worked' (i.e. advanced the game)
            elif state != TowerState.MOVE_SELECT:
                advanced_game = True
                self.state = TowerState.BATTLE  # this is important b/c we need to reset the state back to BATTLE

                break

        if not advanced_game:
            self._log_error_image(f'could_not_search_move', state)
            raise ValueError(f'Could not select a move while in move select (for some reason), on move idx {self.move_idx}, possible moves: {self.moves}')

        self.move_idx += 1

        return state

def search_moves(savestate_file: str, moves: list[int], search_queue: Queue, damage_value: Value):
    """
    Given the savestate file, plays the remainder of the game until it reaches a stopping point.
    It continuously updates the damage dealt value provided, even while the battle is still ongoing
    Adds the result (if we won the battle yet, total damage dealt after the search stops, the move list, and also the # of turns played out) to the provided multiprocessing queue
    NOTE: this must be called in a new process or else Desmume will complain about already being initialized
    """

    agent = BattleTowerSearchV2SubAgent(savestate_file, moves, damage_value=damage_value)

    try:
        state = agent.play_remainder_of_battle()
    except InvalidMoveSelected: #  this means we were unable to select a move in the search so we should just stop the search
        state = TowerState.LOST_SET
    except SwappedPokemon: # this means we stopped the search early due to a pokemon fainting so we stopped the search early
        state = TowerState.LOST_SET
    except: # NOTE: I'm listing out the above errors even though it doesn't change stuff b/c I want to remember what I did for later
        state = TowerState.LOST_SET

    won = False

    if state == TowerState.WON_BATTLE:
        won = True
    elif state == TowerState.END_OF_SET:
        state = agent._wait_for(
            (won_set, TowerState.WON_SET),
            (lost_set, TowerState.LOST_SET),
            button_press='B', # I want to skip dialog and also not accidentally re-start another dialog, so I choose B over A
        )

        if state == TowerState.WON_SET:
            won = True

    with damage_value.get_lock():
        damage_dealt = damage_value.value

    search_queue.put((won, damage_dealt, moves, agent.move_idx))

class BattleTowerSearchV2Agent(BattleTowerAgent):

    def __init__(self,
        render=False,
        savestate_file=SEARCH_TEAM_SAVESTATE,
        db_interface: BattleTowerDBInterface = None,
        depth=1,
        team=None,
    ):
        """
        V2 Strategy:
        1. Only search until current Pokemon faints
          * If current Pokemon selects a winning move combination, go with that (or maybe 1 more move just to confirm)
          * If there is no winning combination, use the total amount of damage they did before fainting and use that to determine which move to take
          * [OPTIONAL] if there is no winning move, try swapping to a different Pokemon and seeing how effective it is w/ search_depth=1
        2. Whenever we swap, if there are two options, do a search for each of them
          * [IMPLEMENTATION] search_whatever can have the move combo and also swap=None, 1, or 2
        3. We may stop searching before all moves have been fully searched. Whenever a search returns, we check:
          1. If that search won the battle, we immediately stop.
          2. If that search didn't win the battle, we check whether it is the highest-damaging search *at that wall-clock time of the search*
             * If that move's damage is higher than anything else at that point, we stop searching and go with it
             * Otherwise, we keep waiting.
             * It's possible that a different move will lead to a larger damage overall, just take more time,
               but w/ the current moveset of each Pokemon, that's unlikely (in a sense, it's a tradeoff of time vs accuracy)

        depth is how many combinations of moves that we'll try, although it's more like a class than an actual # of steps we'll go down the tree
        If depth is 1 or 2, it's all possible permutations of move 1 or 2 nodes down the tree. If depth is 3, we also include swapping Pokemon
        """
        super().__init__(render, savestate_file, db_interface)

        self.depth = depth
        self.strategy = f'search_v2_depth_{depth}'
        if team is None:
            self.team = """Garchomp @ Focus Sash  
Ability: Sand Veil  
EVs: 4 HP / 252 Atk / 252 Spe  
Jolly Nature  
- Outrage  
- Earthquake  
- Fire Fang  
- Swords Dance  

Suicune @ Leftovers  
Ability: Pressure  
EVs: 252 HP / 252 Def / 4 SpD  
Bold Nature  
IVs: 0 Atk  
- Surf  
- Ice Beam  
- Calm Mind  
- Toxic  

Scizor @ Choice Band  
Ability: Technician  
EVs: 252 HP / 252 Atk / 4 SpD  
Adamant Nature  
- Bullet Punch  
- Bug Bite  
- Aerial Ace  
- Superpower""" # normally I do rest suicune, but I don't want to play a battle out crazy long just spamming rest

    def _select_and_execute_move(self) -> TowerState:
        savestate_file = uuid.uuid4().hex + '.dst'
        savestate_path = os.path.join(SEARCH_TMP_SAVESTATE_DIR, savestate_file)
        self.env.emu.savestate.save_file(savestate_path)

        state = self.state
        if self.depth == 1:
            possible_moves = [[move] for move in range(POKEMON_MAX_MOVES)]
        elif self.depth == 2:
            possible_moves = [[first, second] for first in range(POKEMON_MAX_MOVES) for second in range(POKEMON_MAX_MOVES)]
        else:
            raise NotImplementedError("I don't currently have anything for swapping Pokemon yet")

        logger.debug(f'Searching over {possible_moves}')

        # to help w/ efficiency (b/c especially early on, it can take a while to 'lose' when you make a bad move; literally PP stalled against Shedinja)
        # as soon as I get the first 'winning' result, we're going with it (this also prioritizes moves that will help us win *fast*)
        search_processes = []
        result_queue = Queue()

        damage_values = [Value('i', 0) for _ in range(len(possible_moves))]

        def damage_argmax():
            max_idx = -1
            max_value = -np.inf
            for i, v in enumerate(damage_values):
                with v.get_lock():
                    if v.value > max_value:
                        max_value = v.value
                        max_idx = i

            return max_idx, max_value

        for i, move_list in enumerate(possible_moves):
            p = Process(target=search_moves, args=(savestate_path, move_list, result_queue, damage_values[i]))
            search_processes.append(p)
            p.start()

        best_result = None
        completed_processes = 0
        while best_result is None and completed_processes < len(search_processes):
            result = result_queue.get(block=True)
            completed_processes += 1

            won_battle = result[0]
            if won_battle:
                best_result = result
                logger.debug('Found a winning move, stopping early.')
            else:
                damage_dealt = result[1]

                _, max_damage = damage_argmax()

                if damage_dealt >= max_damage: # if we're as good or better than any other search at this very moment, we can stop early
                    best_result = result
                    logger.debug('Stopping early b/c we found a move that is the best so far.')

        if best_result is not None:
            for p in search_processes:
                p.terminate()

        for p in search_processes: # multiprocessing thing; to prevent threads from becoming zombies, we join
            p.join()

        logger.debug(f'Damage values at the end of the search: {[v.value for v in damage_values]}')

        if best_result:
            move = best_result[2][0] # remember, the result is a tuple of (won, damage, move_list, turns)
            logger.info(f'After searching with a depth of {self.depth}, move {move} did {best_result[1]} damage in {best_result[3]} turns'
                        + ' and lead to a win.' if best_result[0] else '.')
        else:
            # okay it's highly unlikely, but technically possible, that we don't get a best_result from the above search
            # if, e.g. the last search move had done the most amount of damage when all other searches terminated, but then the opponent healed
            #  and so it did less final damage
            max_idx, max_damage = damage_argmax()
            move = possible_moves[max_idx][0]
            logger.info(f'After exhausting all searches with a depth of {self.depth}, '
                        f'got into a rare situation where no best move was initially found. '
                        f'Choosing move {move}, which lead to {max_damage} damage dealt.')

        # it is *technically* possible that no move lead to a win, and that there are also some moves that aren't
        #  possible to make, so we still have to do this whole thing *just in case* (see `AAgent` for more info about trying moves)
        advanced_game = False
        for i in range(POKEMON_MAX_MOVES):
            chosen_move = move + i
            self._goto_move(chosen_move)

            self._general_button_press('A')
            state = self._wait_for_battle_states()

            if state != TowerState.MOVE_SELECT:
                advanced_game = True
                self.state = TowerState.BATTLE  # this is important b/c we need to reset the state back to BATTLE

                break

        if not advanced_game:
            self._log_error_image('search_could_not_make_move', state)
            raise ValueError(f'Could not select a move while in move select (for some reason)')

        # it's polite to clean up the savestate dir after finishing the search
        if os.path.exists(savestate_path):
            os.remove(savestate_path)

        return state

    def _swap_to_next_pokemon(self):
        """When a pokemon faints, this function swaps to the next Pokemon"""
        # sometimes, 'wait_for' is a little zeleous with the 'A' button and we already select a Pokemon
        # in that case, we can just hit B to go back (we don't want to just hit A again b/c that pokemon is most likely the fainted one)
        if not pokemon_is_fainted(self.cur_frame):
            self._general_button_press('B')

            # ... but if that didn't work, then we're in an unknown state
            if not pokemon_is_fainted(self.cur_frame) or not self.state == TowerState.SWAP_POKEMON:
                self._log_error_image(message='swap_pokemon', state=self.state)
                raise ValueError(f"Something is out of order here, `_swap_to_next_pokemon` was called but no Pokemon is fainted or the state wasn't properly set (it is currently {self.state}.")

        party_status = get_party_status(self.cur_frame)

        logger.info(f'A Pokemon has fainted, current party status: ' + ' | '.join([f'Slot {i+1} {"healthy" if status else "fainted"}' for i, status in enumerate(party_status)]))

        swapped_pokemon = False
        for i, slot_is_healthy in enumerate(party_status):
            if slot_is_healthy: # once we get to a healthy Pokemon, we need to hit A twice to select it and send it out on the field
                self._general_button_press('A')
                self._general_button_press('A')
                swapped_pokemon = True
                logger.info(f'Swapping to slot {i}')
                break
            else:
                self._general_button_press('RIGHT') # if the currently selected slot is fainted, we can try the next one by just hitting right

        if not swapped_pokemon:
            self._log_error_image(message='no_pokemon_to_swap')
            raise ValueError("Something went wrong here. We should have found and swapped to a healthy Pokemon by now, but we couldn't find any healthy Pokemon")

# Other optimization notes (for speech or accuracy, maybe turn these into slight variations?):
# 1. Maybe only do a search for the very first turn, but then also re-do the search whenever you have to swap Pokemon
# 2. For easier battles (e.g. before battle 21) only do search for the first N moves, and then just go from there
# 3. Don't do *any* search for the first 20 battles b/c there is only a small chance (~10%) that it doesn't get to battle 21
# 4. After receiving the 1st win, keep searching for either first_sample time * DURATION_MOD, or wait for N more wins and then just go w/ 'default' move
# 5. Figure out some way to go w/ the 'last used move' instead of the 'default' move (but what about status moves?)
# 6. Keep track of *how many times* a move led to a win (maybe even re-running the same move combo multiple times) and choosing the 'best' move that way
# 7. What about random search?
# 8. BIG OPTIMIZATION: keep the processes alive and more specifically desmume; loading a savestate is pretty quick, but there is a bit of a delay whenever you start up desmume
# 9. When doing a depth of 2, use *both* moves, don't just use the first move (which means it'll take 2 turns)


# V2 Strategy:
# 1. Only search until current Pokemon faints
#   * If current Pokemon selects a winning move combination, go with that (or maybe 1 more move just to confirm)
#   * If there is no winning combination, use the total amount of damage they did before fainting and use that to determine which move to take
# 2. Whenever we swap, if there are two options, do a search for each of them
#   * [IMPLEMENTATION] search_whatever can have the move combo and also swap=None, 1, or 2
if __name__ == '__main__':
    agent = BattleTowerSearchV2Agent(
        render=True,
        depth=1,
        #db_interface=BattleTowerServerDBInterface()
    )

    agent.play()
    #
    # from battle_tower_agent import *
    # from pokemon_env import *
    # import keyboard
    # import win32api
    # import win32gui
    # import time
    #
    # emu = DeSmuME()
    # emu.open(ROM_FILE)
    # emu.savestate.load_file('ROM\Pokemon - Platinum Battle Tower Search Team.dst')
    # # emu.savestate.load_file('ROM\\14 Win Streak.dst')
    # emu.volume_set(0)
    #
    #
    # # Create the window for the emulator
    # window = emu.create_sdl_window()
    #
    # # Get handle for desmume sdl window
    # window_handle = win32gui.FindWindow(None, "Desmume SDL")
    #
    # checks = [
    #     is_dialog_box,
    #     is_save_dialog,
    #     is_save_overwrite_dialog,
    #     in_pokemon_select,
    #     is_ready_for_battle_tower,
    #     pokemon_is_fainted,
    #     in_battle,
    #     is_next_opponent_box,
    #     won_set,
    #     at_save_battle_video,
    #     lost_set,
    #     in_move_select,
    # ]
    #
    # CONTROLS = {
    #     "enter": Keys.KEY_START,
    #     "right shift": Keys.KEY_SELECT,
    #     "q": Keys.KEY_L,
    #     "w": Keys.KEY_R,
    #     "a": Keys.KEY_Y,
    #     "s": Keys.KEY_X,
    #     "x": Keys.KEY_A,
    #     "z": Keys.KEY_B,
    #     "up": Keys.KEY_UP,
    #     "down": Keys.KEY_DOWN,
    #     "right": Keys.KEY_RIGHT,
    #     "left": Keys.KEY_LEFT,
    # }
    #
    # self_opp_pkmn = None
    # self_damage_dealt = 0
    # self_opp_hp = None
    #
    #
    # while not window.has_quit():
    #     # Check if any buttons are pressed and process them
    #     # I like to just whipe all keys first so that I don't have to worry about removing keys or whatnot
    #     emu.input.keypad_rm_key(Keys.NO_KEY_SET)
    #
    #     for key, emulated_button in CONTROLS.items():
    #         if keyboard.is_pressed(key):
    #             emu.input.keypad_add_key(keymask(emulated_button))
    #         else:
    #             emu.input.keypad_rm_key(keymask(emulated_button))
    #
    #     screen_buffer = emu.display_buffer_as_rgbx()
    #     screen_pixels = np.frombuffer(screen_buffer, dtype=np.uint8)
    #     screen = screen_pixels[:SCREEN_PIXEL_SIZE_BOTH * 4]
    #     screen = screen.reshape((SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 4))[..., :3]  # drop the alpha channel
    #
    #     if keyboard.is_pressed('t'):
    #         image_path = os.path.join('images', 'Decision Making', input('Enter image path:') + '.PNG')
    #
    #         cv2.imwrite(image_path, screen)
    #
    #     # Check if touch screen is pressed and process it
    #     if win32api.GetKeyState(0x01) < 0:
    #         # Get coordinates of click relative to desmume window
    #         x, y = win32gui.ScreenToClient(window_handle, win32gui.GetCursorPos())
    #         # Adjust y coord to account for clicks on top (non-touch) screen
    #         y -= SCREEN_HEIGHT
    #
    #         if x in range(0, SCREEN_WIDTH) and y in range(0, SCREEN_HEIGHT):
    #             emu.input.touch_set_pos(x, y)
    #         else:
    #             emu.input.touch_release()
    #     else:
    #         emu.input.touch_release()
    #
    #     for check in checks:
    #         if check(screen):
    #             print(f'{check.__name__}: {check(screen)}')
    #
    #     if pokemon_is_fainted(screen):
    #         print(get_party_status(screen))
    #
    #     if in_move_select(screen):
    #         opp_hp = get_opponent_hp_bar(screen)
    #         opp_pkmn = get_opponent_pokemon_name(screen)
    #
    #
    #     if is_next_opponent_box(screen):
    #         print('Next opp:', get_battle_number(screen))
    #
    #     emu.cycle()
    #     window.draw()