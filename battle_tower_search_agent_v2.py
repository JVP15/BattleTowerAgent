import logging
import os
import uuid
from multiprocessing import Pool, Queue, Process, Value, Event, Barrier

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

class EarlySearchStop(Exception):
    pass

class HPWatcher:
    """This should be used in `_wait_for` as a check. While the game is cycling, it tracks the HP bar of the opponent Pokemon."""
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
    def __init__(self,
                 savestate_file: str,
                 moves: int | list[int],
                 stop_event: Event,
                 swap_to: int | None = None,
                 damage_value: Value = None,
    ):
        """
        This is an "agent" that has been hacked and slashed apart to play the game halfway through a battle.
        This agent executes the actual search strategy, see BattleTowerSearchV2Agent for that strategy.

        :param savestate_file: A path to a savestate file that the subagent will pick up the game from
        :param moves: the list of moves that the subagent will execute in order (i.e. turn 1 it does move 0, turn 2 move 1, etc), can be a single move
        :param stop_event: a multiprocessing event that tells the search subagent to stop searching immediately (used to implement early search stopping)
        :param swap_to: before making a move, swap to the pokemon in that idx (can be none)
        :damage_value: a multiprocessing Value used to keep track of the damage dealt thus far by the subagent
        """
        super().__init__(render=False, savestate_file=savestate_file)

        if isinstance(moves, int):
            moves = [moves]

        self.moves = moves
        self.move_idx = 0
        self.team = '' # there is no logging to a DB for these searches, so we don't need to specify a teams
        self.swap_to = swap_to

        self.stop_event = stop_event

        self.hp_watcher = HPWatcher(damage_value=damage_value)

    def play_remainder_of_battle(self) -> TowerState:
        """
        This function starts from the move_select or pokemon_select screen and plays until one of these stopping conditions:
        1. we go to Pokemon select (either through U-turn/the like or through fainting)
        2. the battle ends in a win
        3. we fail to select a move that is included in the search (but not the 'default' move)

        This is expected to be called in either _select_move function
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

        return self._wait_for(
            # NOTE: this is how we keep track of the opponent's HP bar as closely as "real time" as possible (since checks are run every frame)
            # The watcher has to look *before* any other checks b/c once a check is found, no other checks are run
            (self.hp_watcher, TowerState.WAITING),
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

    def _select_move(self) -> TowerState:
        # The search subagent starts by making each move in-order, and once we've gotten past the moves that we
        #  want to search over, we go back to using the 'default' move (i.e. the first one, which is as we saw with the 'A' agent, is pretty solid)
        state = self.state

        if self.move_idx < len(self.moves):
            move = self.moves[self.move_idx]
        else:
            move = DEFAULT_MOVE

        advanced_game = False

        # see `AAgent` for more info about why we have to keep trying moves like this
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

    def _act(self, action: str | None = None) -> np.ndarray:
        # _act is called the most frequently (basically every cycle) so it's the best place to check if we stop early
        if self.stop_event.is_set():
            raise EarlySearchStop()
        else:
            return super()._act(action)



def init_search_process(
        savestate_queue: Queue,
        moves: list[int],
        search_queue: Queue,
        early_stop_event: Event,
        search_stop_barrier: Barrier,
        damage_value: Value
):
    """
    When a savestate file is pushed to the savestate queue, plays the remainder of the game until it reaches a stopping point.
    It continuously updates the damage dealt value provided, even while the battle is still ongoing
    Adds the result (if we won the battle yet, total damage dealt after the search stops, the move list, and also the # of turns played out) to the provided multiprocessing queue

    This loops (getting a new file from the savestate queue) until the process is manually shut down.
    NOTE: this must be called in a new process or else Desmume will complain about already being initialized
    """

    while True:
        savestate_file = savestate_queue.get(block=True)

        agent = BattleTowerSearchV2SubAgent(savestate_file, moves, stop_event=early_stop_event, damage_value=damage_value)

        try:
            state = agent.play_remainder_of_battle()
        except (InvalidMoveSelected, #  this means we were unable to select a move in the search so we should just stop the search
                SwappedPokemon, # this means we stopped the search early due to a pokemon fainting so we stopped the search early
                EarlySearchStop # this means some other process finished searching earlier and now all processes need to stop
        ):
            # NOTE: this is effectively treated as a loss, I want to keep it as a different state
            state = TowerState.STOPPED_SEARCH
        except:
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

        # once we've submitted our results, we just have to wait for all other processes to clear up
        search_stop_barrier.wait()


GARCHOMP_SUICUNE_SCIZOR_TEAM = """Garchomp @ Focus Sash  
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

class BattleTowerSearchV2Agent(BattleTowerAgent):

    def __init__(self,
        render=False,
        savestate_file=SEARCH_TEAM_SAVESTATE,
        db_interface: BattleTowerDBInterface = None,
        depth=1,
        team=GARCHOMP_SUICUNE_SCIZOR_TEAM,
    ):
        """
        V2 Strategy:
        1. Only search until current Pokemon faints
          * If current Pokemon selects a winning move combination, go with that (or maybe 1 more move just to confirm)
          * If there is no winning combination, use the total amount of damage they did before fainting and use that to determine which move to take
          * [OPTIONAL] if there is no winning move, try swapping to a different Pokemon and seeing how effective it is w/ search_depth=1
        2. Whenever we swap, if there are two options, do a search for each of them
          * [IMPLEMENTATION] search_whatever can have the move combo and also swap=None, 1, or 2
          * [OPTIONAL] also check this at the beginning of the game? Maybe the lead pokemon is not good against the opponent's lead
        3. We may stop searching before all moves have been fully searched. Whenever a search returns, we check:
          1. If that search won the battle, we immediately stop.
          2. If that search didn't win the battle, we check whether it is the highest-damaging search *at that wall-clock time of the search*
             * If that move's damage is higher than anything else at that point, we stop searching and go with it
             * Otherwise, we keep waiting.
             * It's possible that a different move will lead to a larger damage overall, just take more time,
               but w/ the current moveset of each Pokemon, that's unlikely (in a sense, it's a tradeoff of time vs accuracy)
        4. If we've found a win, we will stop searching and then go w/ the list of moves selected there.
            * This is a bit of a tradeoff b/w speed and performance, but if we've found a win, we're likly only a turn or two away anyways
            * If we have to swap pokemon, this gets reset, which helps in case we thought we won but in fact something changed.

        depth is how many combinations of moves that we'll try, although it's more like a class than an actual # of steps we'll go down the tree
        If depth is 1 or 2, it's all possible permutations of move 1 or 2 nodes down the tree. If depth is 3, we also include swapping Pokemon
        """
        super().__init__(render, savestate_file, db_interface)

        self.depth = depth
        self.strategy = f'search_v2_depth_{depth}'
        self.team = team

        # search optimization here: if we find a winning move combo, lets just use it!
        self.winning_move_list = None
        self.winning_move_idx = 0

        # speed optimization over v1 agent: persistant process (this saves time both spooling up the process and also initing Desmume, which is slow)
        self.result_queue = Queue()

        if depth == 1:
            self.possible_moves = [[move] for move in range(POKEMON_MAX_MOVES)]
        elif depth == 2:
            self.possible_moves = [[first, second] for first in range(POKEMON_MAX_MOVES) for second in range(POKEMON_MAX_MOVES)]
        else:
            raise NotImplementedError("I don't have the hardware to have any deeper trees")

        # TODO: how will this work when we want to swap pokemon? I guess we'll just leave some processes waiting
        # Also will have to change this when I add pokemon swapping
        self.damage_values = [Value('i', 0) for _ in range(len(self.possible_moves))]
        self.savestate_file_queues = [Queue() for _ in range(len(self.possible_moves))]
        self.stop_event = Event()
        self.search_stop_barrier = Barrier(len(self.possible_moves) + 1) # The +1 is for this process

        self.processes = []
        for i, move_list in enumerate(self.possible_moves):
            p = Process(
                target=init_search_process,
                args=(self.savestate_file_queues[i], move_list, self.result_queue, self.damage_values[i])
            )
            self.processes.append(p)
            p.start()


    def _select_move(self) -> TowerState:
        savestate_file = uuid.uuid4().hex + '.dst'
        savestate_path = os.path.join(SEARCH_TMP_SAVESTATE_DIR, savestate_file)
        self.env.emu.savestate.save_file(savestate_path)

        state = self.state

        logger.debug(f'Searching over {self.possible_moves}')

        # this kicks off the search; each process will consume the savestate file and begin the search agent
        for q in self.savestate_file_queues:
            q.put(savestate_path, block=True)

        # see class docstring for the search algorithm here
        best_result = None
        completed_searches = 0
        while best_result is None and completed_searches < len(self.processes):
            # result looks like (won, damage_dealt, move_list, # of turns it took)
            result = self.result_queue.get(block=True)
            completed_searches += 1

            won_battle = result[0]
            if won_battle:
                best_result = result
                logger.debug('Found a winning move, stopping early.')
            else:
                damage_dealt = result[1]

                _, max_damage = self._damage_argmax()

                # if we're as good or better than any other search at this very moment, we can stop early
                if damage_dealt > max_damage:
                    best_result = result
                    logger.debug('Stopping early b/c we found a move that is the best so far.')

        if best_result is not None:
            self.stop_event.set()


        logger.debug(f'Damage values at the end of the search: {[v.value for v in damage_values]}')

        if best_result:
            move = best_result[2][0] # remember, the result is a tuple of (won, damage, move_list, turns)
            log_str = f'After searching with a depth of {self.depth}, move {move} did {best_result[1]} damage in {best_result[3]} turns.'
            if best_result[0]:
                log_str += ' Won the game.'
            logger.info(log_str)
        else:
            # okay it's highly unlikely, but technically possible, that we don't get a best_result from the above search
            # if, e.g. the last search move had done the most amount of damage when all other searches terminated, but then the opponent healed
            #  and so it did less final damage
            max_idx, max_damage = self._damage_argmax()
            move = self.possible_moves[max_idx][0]
            logger.info(f'After exhausting all searches with a depth of {self.depth}, '
                        f'got into a rare situation where no best move was initially found. '
                        f'Choosing move {move}, which lead to {max_damage} damage dealt.')

        # it's polite to clean up the savestate dir after finishing the search
        if os.path.exists(savestate_path):
            os.remove(savestate_path)

        return move

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

        # when we swap pokemon, we need to reset the "winning" move list b/c something CLEARLY went wrong and we didn't win
        self.winning_move_list = None
        self.winning_move_idx = 0

        # TODO: instead of swapping to the next pokemon, implement a way to search over the next possible pokemon (assuming you have a choice)
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

    def _damage_argmax(self):
        max_idx = -1
        max_value = -np.inf
        for i, v in enumerate(self.damage_values):
            with v.get_lock():
                if v.value > max_value:
                    max_value = v.value
                    max_idx = i

        return max_idx, max_value

    def _reset_search(self):
        """
        This function handles all of the multiprocessing stuff that we need to do to make sure that the search is ready for the next round
        """

        # this likely won't be necessary, but I can imagine a situation where the processor is all locked up and so
        #  by the search process ends, we haven't even pulled the savestate file from the queue
        for file_queue in self.savestate_file_queues:
            while not file_queue.empty():
                file_queue.get()

        # TODO: figure out stop event here

        for damage_value in self.damage_values:
            with damage_value.get_lock():
                damage_value.value = 0

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