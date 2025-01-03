import logging
import os
import uuid
from multiprocessing import Pool, Queue, Process

import numpy as np

from battle_tower_agent import (
    BattleTowerAgent,
    BattleTowerAAgent,
    TowerState,
    POKEMON_MAX_MOVES,
    NUM_POKEMON_IN_SINGLES,
    in_battle,
    ROM_DIR,
    won_set,
    lost_set,
    pokemon_is_fainted,
    get_party_status,
)

from battle_tower_search_agent import InvalidMoveSelected

from battle_tower_database.interface import BattleTowerDBInterface, BattleTowerServerDBInterface

DEFAULT_MOVE = 0
SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search Team.dst')
SEARCH_TMP_SAVESTATE_DIR = os.path.join(ROM_DIR, 'search')

logger = logging.getLogger('SearchTowerAgent')

HP_PER_POKEMON = 100

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

class BattleTowerSearchV2SubAgent(BattleTowerAgent):
    """This class is used by the BattleTowerSearchAgent to 'look ahead' for the next possible moves"""

    strategy = 'move_select'
    def __init__(self, savestate_file: str, moves: int | list[int], swap_to: int | None = None):
        super().__init__(render=True, savestate_file=savestate_file)

        if isinstance(moves, int):
            moves = [moves]

        self.moves = moves
        self.move_idx = 0
        self.team = '' # there is no logging to a DB for these searches, so we don't need to specify a teams
        self.swap_to = swap_to

        self.opp_hp = None
        self.damage_dealt = 0
        # I'm going to track when a pokemon faints by it's name; if it's any different, it means the opponent lost a Pokemon
        # TODO: *technically* breaks if the opponent uses U-turn but... I think it's fine
        # NOTE: it also is reliant on the fact that each Pkmn on the foes team is unique
        self.opp_pkmn = None

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

    def _select_and_execute_move(self) -> TowerState:
        # The search subagent starts by making each move in-order, and once we've gotten past the moves that we
        #  want to search over, we go back to using the 'default' move (i.e. the first one, which is as we saw with the 'A' agent, is pretty solid)
        state = self.state

        # to determine if we did damage, we check the opponent's HP at the beginning of every turn
        #   and if it's different than when we chedked on the previous turn, we did damage (or they healed and we did negative damage...)
        #  If the Pokemon is different, we can assume that we ko'd them, which means we did whatever their remaining HP was
        #  NOTE: this doesn't track properly w/ U-turn and opponent switching moves but it's the best I have right now
        #  TODO: make this also work w/ entry hazards... ALSO OUTRAGE!
        opp_hp = get_opponent_hp_bar(self.cur_frame)
        opp_pkmn = get_opponent_pokemon_name(self.cur_frame)
        kod_opp = False

        if self.opp_hp is None:
            self.opp_hp = opp_hp

        if self.opp_pkmn is None:
            self.opp_pkmn = opp_pkmn
        elif (self.opp_pkmn != opp_pkmn).any(): # for future self, use any here instead of all (duh)
            kod_opp = True
            print('KOd previous Pokemon')

        if kod_opp: # for future self: if we ko an opponent, the next opponent will have at least as much health
            self.damage_dealt += self.opp_hp
        else:
            self.damage_dealt += self.opp_hp - opp_hp
        print('Total Damage Dealt', self.damage_dealt)

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
            # NOTE: make sure we catch this error elsewhere
            # TODO: maybe come up with a better way to completely escape from the battle loop?
            if state == TowerState.MOVE_SELECT and self.move_idx < len(self.moves):
                logger.debug(f"Attempted to search over move {chosen_move} but it could not be chosen; stopping search.")
                raise InvalidMoveSelected()
            elif state == TowerState.SWAP_POKEMON:
                logger.debug(f'Finished move search by swapping Pokemon. Current Pokemon did {self.damage_dealt} damage.')
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

def search_moves(savestate_file: str, moves: list[int], search_queue: Queue) -> tuple[bool, list[int], int]:
    """
    Given the savestate file, plays the remainder of the game until it reaches a stopping point.
    Adds the result (a bool if the game was won (true if it won, false if it lost or stopped early), the move list, and also the # of turns played out) to the provided multiprocessing queue
    Requires the filename (str) and list of moves (ints) to be provided as a tuple b/c of the `map` requirements
    NOTE: this must be called in a new process or else Desmume will complain about already being initialized
    """

    agent = BattleTowerSearchV2SubAgent(savestate_file, moves)

    try:
        state = agent.play_remainder_of_battle()
    except:
        # by default when searching, if we run into an error, I want to set it to a loss
        # this will most likely happen if we are using a choice item or torment and try to select an invalid move
        state = TowerState.LOST_SET
    if state == TowerState.WON_BATTLE:
        won = True
    elif state == TowerState.LOST_SET:
        won = False
    # if we chose a move and got thrown back to move_select, it means we couldn't choose that move so we should just stop the search
    elif state == TowerState.MOVE_SELECT:
        won = False
    elif state == TowerState.END_OF_SET:
        state = agent._wait_for(
            (won_set, TowerState.WON_SET),
            (lost_set, TowerState.LOST_SET),
            button_press='B', # I want to skip dialog and also not accidentally re-start another dialog, so I choose B over A
        )

        if state == TowerState.WON_SET:
            won = True
        else:
            won = False
    else:
        agent._log_error_image('subagent_post_battle_loop', state)
        raise ValueError("This *really* shouldn't happen, but somehow the state is", state, "after searching through moves")

    search_queue.put((won, moves, agent.move_idx))

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
        3. Instead of waiting for the current Pokemon to faint everytime, we could do what we do in the v1 agent and just stop on the first return
          but that kinda defeats the purpose of tracking the HP bar doesn't it? Here's a compromise:
          * Wait for 2-3 out of the 4 options and choose the one that deals the most damage; chances are that there will always be a 4th option that takes a while to complete


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
        possible_moves = [[3]]
        logger.debug(f'Searching over {possible_moves}')

        # to help w/ efficiency (b/c especially early on, it can take a while to 'lose' when you make a bad move; literally PP stalled against Shedinja)
        # as soon as I get the first 'winning' result, we're going with it (this also prioritizes moves that will help us win *fast*)
        search_processes = []
        result_queue = Queue()

        for move_list in possible_moves:
            p = Process(target=search_moves, args=(savestate_path, move_list, result_queue))
            search_processes.append(p)
            p.start()

        winning_result = None
        completed_processes = 0
        while winning_result is None and completed_processes < len(search_processes):
            result = result_queue.get(block=True)
            completed_processes += 1

            if result[0]:
                winning_result = result

                for p in search_processes:
                    p.terminate()

        for p in search_processes: # multiprocessing thing; to prevent threads from becoming zombies, we join
            p.join()

        if winning_result:
            move = winning_result[1][0] # remember, the result is a tuple of (won, move_list, turns)
            logger.info(f'After searching with a depth of {self.depth}, move {move} won in {winning_result[2]} turns')
        else:
            logger.info(f'After searching with a depth of {self.depth}, could not find a winning move. Just picking {DEFAULT_MOVE}')
            move = DEFAULT_MOVE

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