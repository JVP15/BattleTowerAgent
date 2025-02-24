import logging
import multiprocessing
import os
import uuid

from dataclasses import dataclass
from multiprocessing import Queue, Value, Event, Barrier

import numpy as np

from battle_tower_agent.agent import (
    BattleTowerAgent,
    TowerState,
    POKEMON_MAX_MOVES,
    NUM_POKEMON_IN_SINGLES,
    SEARCH_SAVESTATE_DIR,
    in_battle,
    ROM_DIR,
    check_key_pixels,
    won_set,
    lost_set,
    pokemon_is_fainted,
    get_party_status,
    get_selected_pokemon_in_swap_screen,
    in_move_select,
    is_next_opponent_box,
    at_save_battle_video,
)

from battle_tower_agent.search_agent import InvalidMoveSelected

from battle_tower_agent.battle_tower_database.interface import BattleTowerDBInterface, BattleTowerServerDBInterface

DEFAULT_MOVE = 0

if os.name == 'nt':
    SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search Team.dst')
else:
    SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search Team Linux.dst')

logger = logging.getLogger('SearchTowerAgent')

HP_PER_POKEMON = 100

# minor optimization here, if we haven't either fainted or won by 10 turns (chosen somewhat arbitrarily) then we probably have a good idea of the search space
SUBAGENT_STOPPING_TURN = 10

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

@dataclass
class SearchResultMessage:
    won: bool
    damage_dealt: int
    moves: list[int]
    turns: int
    swap_to: int | None

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

                # MINOR OPTIMIZATION: HP changes every, say, 3 frames, so we don't need to acquire the lock if the opp's HP doesn't change this frame
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

        Args:
            savestate_file: A path to a savestate file that the subagent will pick up the game from
            moves: the list of moves that the subagent will execute in order (i.e. turn 1 it does move 0, turn 2 move 1, etc), can be a single move
            stop_event: a multiprocessing event that tells the search subagent to stop searching immediately (used to implement early search stopping)
            swap_to: before making a move, swap to the pokemon in that idx (can be none)
            damage_value: a multiprocessing Value used to keep track of the damage dealt thus far by the subagent
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

        # if we didn't swap pokemon, we're in move_select, but _run_battle_loop expects us to be in the fight screen, we have to do that;
        # it's slightly wasteful, but most games take thousands of frames, and it only costs us about 20 total so w/e
        self.state = self._wait_for(
            (in_battle, TowerState.BATTLE),
            (pokemon_is_fainted, TowerState.SWAP_POKEMON),
            button_press='B'
        )

        if self.swap_to and not pokemon_is_fainted(self.cur_frame):
            self._log_error_image(message='not_in_swap_screen')
            raise ValueError("Something went wrong here. We are supposed to swap Pokemon but we aren't in the pokemon swap screen")
        elif self.swap_to:
            selected_slot = get_selected_pokemon_in_swap_screen(self.cur_frame)
            if selected_slot is None:  # sometimes, we aren't automatically selecting any slot, which we can fix by hitting 'A'
                self._general_button_press('A')
                selected_slot = get_selected_pokemon_in_swap_screen(self.cur_frame)

            # this will navigate us to the right slot #
            while selected_slot != self.swap_to:
                if selected_slot < self.swap_to:
                    self._general_button_press('RIGHT')
                elif selected_slot > self.swap_to:
                    self._general_button_press('LEFT')

                selected_slot = get_selected_pokemon_in_swap_screen(self.cur_frame)

            # once we get to the chosen pokemon, we have to hit A twice to select it and send it out on the field
            self._general_button_press(['A', 'A'])

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

    def _select_move(self) -> int:
        # The search subagent starts by making each move in-order, and once we've gotten past the moves that we
        #  want to search over, we go back to using the 'default' move (i.e. the first one, which is as we saw with the 'A' agent, is pretty solid)

        if self.move_idx >= SUBAGENT_STOPPING_TURN:
            logger.debug(f'Stopping Search Subagent because we hit turn search limit of {SUBAGENT_STOPPING_TURN}')
            raise EarlySearchStop()

        if self.move_idx < len(self.moves):
            move = self.moves[self.move_idx]
        else:
            move = DEFAULT_MOVE

        self.move_idx += 1

        return move

    def _swap_to_next_pokemon(self):
        raise EarlySearchStop()

    def _act(self, action: str | None = None) -> np.ndarray:
        # _act is called the most frequently (basically every cycle) so it's the best place to check if we stop early
        if self.stop_event.is_set():
            raise EarlySearchStop()
        else:
            return super()._act(action)



def init_search_process(
        savestate_queue: Queue,
        moves: list[int],
        swap_to: int,
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

        try:
            agent = BattleTowerSearchV2SubAgent(savestate_file, moves, swap_to=swap_to, stop_event=early_stop_event,
                                                damage_value=damage_value)

            state = agent.play_remainder_of_battle()
        except (InvalidMoveSelected, #  this means we were unable to select a move in the search so we should just stop the search
                SwappedPokemon, # this means we stopped the search early due to a pokemon fainting so we stopped the search early
                EarlySearchStop # this means some other process finished searching earlier and now all processes need to stop
        ):
            # NOTE: this is effectively treated as a loss, I want to keep it as a different state
            state = TowerState.STOPPED_SEARCH
        except Exception as e:
            state = TowerState.LOST_SET
            logger.warning(f'Encountered an unexpected error while searching: {e}')

        won = False

        if state == TowerState.WON_BATTLE:
            won = True
        elif state == TowerState.END_OF_SET:
            try:
                state = agent._wait_for(
                    (won_set, TowerState.WON_SET),
                    (lost_set, TowerState.LOST_SET),
                    button_press='B', # I want to skip dialog and also not accidentally re-start another dialog, so I choose B over A
                )

                if state == TowerState.WON_SET:
                    won = True

            except EarlySearchStop:
                pass # just don't want to log when we do an early search
            except Exception as e:
                logger.warning(f'Encountered an unexpected error while searching: {e}')

        with damage_value.get_lock():
            damage_dealt = damage_value.value

        result = SearchResultMessage(won=won, damage_dealt=damage_dealt, moves=moves, turns=agent.move_idx, swap_to=swap_to)
        search_queue.put(result)

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
    """
    V2 Strategy:
    1. Only search until current Pokemon faints
      * If current Pokemon selects a winning move combination, go with that (or maybe 1 more move just to confirm)
      * If there is no winning combination, use the total amount of damage they did before fainting and use that to determine which move to take
      * [WON'T BE IMPLEMENTED] if there is no winning move, try swapping to a different Pokemon and seeing how effective it is w/ search_depth=1
    2. Whenever we swap, if there are two options, do a search for each of them
      * [NOT IMPLEMENTED YET] search_whatever can have the move combo and also swap=None, 1, or 2
      * [NOT IMPLEMENTED YET] also check this at the beginning of the game? Maybe the lead pokemon is not good against the opponent's lead
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
        * Since we also keep track of how many turns the battle *should* take, we can do a reset if it ends up suprising us
    5. Limit the search depth to 10 turns (which is a reasonable depth)
      b/c if we search for 20 turns, it isn't likely that we'll find a better move vs just searching for 10 turns.
        * This also synergizes well w/ the "stop searching on found win" b/c we may win and PP stall after 20 turns
        (by "may" I mean we *have*) but this will prevent it.

    These additions have lead to *major* gains in speed over the v1 search agent, and also lead to longer streaks.
    """

    def __init__(self,
        render=False,
        savestate_file=SEARCH_TEAM_SAVESTATE,
        db_interface: BattleTowerDBInterface = None,
        depth=1,
        search_swap=True,
        team=GARCHOMP_SUICUNE_SCIZOR_TEAM,
    ):
        """
        Creates the Battle Tower Search Agent (v2).
        See the class docstring for the strategy.

        Args:
            render: Whether to display the battle as it's going on.
            savestate_file: The initial savestate file that the agent loads the game from.
                There is a somewhat intricate setup needed to run the agent, so I don't recommend changing this.
            db_interface: A BattleTowerDB Interface, letting the agent record it's stats to a DB as it is playing
                (by default it is None, so it won't  record anything).
            depth: How many combinations of moves that we'll try, although it's more like a class than an actual # of steps we'll go down the tree
                If depth is 1 or 2, it's all possible permutations of move 1 or 2 nodes down the tree.
                Actually, only depths of 1 and 2 are supported.
            search_swap: Whether to search over the possible swaps when a Pokemon faints. Slightly worse overall, but saves on compute.
            team: The team (in Pokemon Showdown format) used in the battle tower.
                By default, goes with the team that is chosen with the default savestate.
        """
        super().__init__(render, savestate_file, db_interface)

        self.depth = depth
        self.search_swap = search_swap
        self.strategy = f'search_v2_depth_{depth}'

        if search_swap:
            self.strategy += '_swap'

        self.team = team

        # SEARCH OPTIMIZATION: if we find a winning move combo, lets just use it!
        # This is a bit of a tradeoff b/w speed (fewer searches) and accuracy (since we could find a better search later)
        # Even for a simple 2-turn battle (e.g. Earthquake turn 1 + Outrage) it takes ~20 seconds from turn 1 to finish
        # But it only takes ~15 seconds by using the cached winning move. The time savings will be even greater for games w/ 2,3, or 4 moves out.
        self.winning_move_list = None
        self.winning_move_idx = 0
        self.pred_turns_to_win = 0

        self.savestate_path = None

        # SPEED OPTIMIZATION: (over v1 agent) persistant process (this saves time both spooling up the process and also initing Desmume, which is slow)
        #  seems to be about 20% faster
        self.result_queue = Queue()
        self.stop_event = Event()

        if depth == 1:
            self.possible_moves = [[move] for move in range(POKEMON_MAX_MOVES)]
        elif depth == 2:
            self.possible_moves = [[first, second] for first in range(POKEMON_MAX_MOVES) for second in range(POKEMON_MAX_MOVES)]
        else:
            raise NotImplementedError("I don't have the hardware to have any deeper trees")

        self.damage_values = [Value('i', 0) for _ in range(len(self.possible_moves))]
        self.savestate_file_queues = [Queue() for _ in range(len(self.possible_moves))]
        self.search_stop_barrier = Barrier(len(self.possible_moves) + 1) # The +1 is for this process

        # on linux, desmume needs to use forkserver (maybe spawn is acceptable? haven't tested), but on windows, the default works just fine
        # TODO: this doesn't work yet
        ctx = multiprocessing.get_context('forkserver' if os.name == 'posix' else None)
        self.processes = []
        for i, move_list in enumerate(self.possible_moves):
            p = ctx.Process(
                target=init_search_process, # args look like savestate_queue, move_list, swap_to (set to None), result_queue, stop_event, stop_barrier, and damage
                args=(self.savestate_file_queues[i], move_list, None, self.result_queue, self.stop_event, self.search_stop_barrier, self.damage_values[i])
            )
            self.processes.append(p)
            p.start()

        # we basically re-create everything above (except for the stop event)
        # seperately for searching over swapped pokemon b/c there are only certain conditions to search for swap
        # (if we actually *do* search over possible swaps)
        num_swap_searches = len(self.possible_moves) * (NUM_POKEMON_IN_SINGLES - 1) if self.search_swap else 0
        self.swap_damage_values = [Value('i', 0) for _ in range(num_swap_searches)]
        self.swap_savestate_file_queues = [Queue() for _ in range(num_swap_searches)]
        self.swap_stop_barrier = Barrier(num_swap_searches + 1)

        self.swap_processes = []
        for i, move_list in enumerate(range(num_swap_searches)):
            swapped_pkmn = i // POKEMON_MAX_MOVES + 1 # the "swap_to" pokemon starts from 1
            p = ctx.Process(
                target=init_search_process, # args look like savestate_queue, move_list, swap_to, result_queue, stop_event, stop_barrier, and damage
                args=(self.swap_savestate_file_queues[i], move_list, swapped_pkmn, self.result_queue, self.stop_event, self.swap_stop_barrier, self.swap_damage_values[i])
            )
            self.swap_processes.append(p)
            p.start()

        # now it's actually worthwhile keeping track of the # of Pokemon we have b/c we don't need to swap if we only have 1 pokemon in the back
        self.healthy_pokemon = NUM_POKEMON_IN_SINGLES


    def _run_battle_loop(self) -> TowerState:
        state = super()._run_battle_loop()
        self._reset_winning_moves() # IMPORTANT: need to reset the winning move list after each battle
        self.healthy_pokemon = NUM_POKEMON_IN_SINGLES # also need to reset the # of healthy pokemon here

        return state

    def _select_move(self) -> int:
        if self.winning_move_idx > self.pred_turns_to_win:
            self._reset_winning_moves()
            logger.info(f"We're on turn {self.winning_move_idx} of the 'winning' move list yet we expected to win in {self.pred_turns_to_win}, search will resume.")

        if self.winning_move_list is None:
            logger.debug(f'Searching over {self.possible_moves}')

            best_result = self._do_search(
                savestate_queue=self.savestate_file_queues,
                processes=self.processes,
                damage_values=self.damage_values
            )

            if best_result:
                move = best_result.moves[0]
                log_str = f'After searching with a depth of {self.depth}, move {move} did {best_result.damage_dealt} damage in {best_result.turns} turns.'
                if best_result.won:
                    log_str += f' Won the game. Following {best_result.moves} for the rest of the game.'
                logger.info(log_str)
            else:
                max_idx, max_damage = self._damage_argmax(self.damage_values)
                move = self.possible_moves[max_idx][0]
                logger.info(f'After exhausting all searches with a depth of {self.depth}, '
                            f'choosing move {move}, which lead to {max_damage} damage dealt.')

            # this is very important to make sure that everything is synchronized
            self._reset_search(battle=True)

        else:
            # this will speed up certain scenarios by not having to search over, say, the last turn b/c we've already seen that the particular move wins in that scenario
            if self.winning_move_idx < len(self.winning_move_list):
                move = self.winning_move_list[self.winning_move_idx]
            else:
                move = DEFAULT_MOVE

            logger.info(f'Using cached move list {self.winning_move_list} which leads to a win. On turn {self.winning_move_idx}, choosing move {move}.')

            self.winning_move_idx += 1

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
        self.healthy_pokemon = party_status.sum()
        logger.info(f'A Pokemon has fainted, current party status: ' + ' | '.join([f'Slot {i+1} {"healthy" if status else "fainted"}' for i, status in enumerate(party_status)]))

        # when we swap pokemon, we need to reset the "winning" move list b/c something CLEARLY went wrong and we didn't win
        if self.winning_move_list is not None:
            logger.info(f'Expected to win by following {self.winning_move_list}, but on turn {self.winning_move_idx} the current pokemon fainted. Restarting search.')
        self._reset_winning_moves()

        # TODO [OPTIONAL]: when swapping pokemon, whatever the next move they choose, go w/ it for the next round instead of searching
        do_search = self.healthy_pokemon >= 2 # no point to search unless there is a choice
        if do_search and self.search_swap:
            # NOTE: this *will* set the winning move if we find it during the search so there's that
            best_result = self._do_search(
                savestate_queue=self.swap_savestate_file_queues,
                processes=self.swap_processes,
                damage_values=self.swap_damage_values,
            )

            if best_result:
                swap_slot = best_result.swap_to
                log_str = (f'After searching with a depth of {self.depth}, Pokemon {swap_slot} did {best_result.damage_dealt}' 
                           ' damage in {best_result.turns} turns. Swapping to {swap_slot}.')
                if best_result.won:
                    log_str += f' Won the game. Following {best_result.moves} for the rest of the game.'
                logger.info(log_str)
            else:
                max_idx, max_damage = self._damage_argmax(self.swap_damage_values)
                swap_slot = max_idx // NUM_POKEMON_IN_SINGLES + 1
                logger.info(f'After exhausting all searches with a depth of {self.depth}, '
                            f'swapping to slot {swap_slot} leads to {max_damage} damage dealt.')

            # this is very important to make sure that everything is synchronized
            self._reset_search(swap=True)

        else:
            swap_slot = party_status.argmax()

        selected_slot = get_selected_pokemon_in_swap_screen(self.cur_frame)
        if selected_slot is None: # sometimes, we aren't automatically selecting any slot, which we can fix by hitting 'A'
            self._general_button_press('A')
            selected_slot = get_selected_pokemon_in_swap_screen(self.cur_frame)

        # this will navigate us to the right slot #
        while selected_slot != swap_slot:
            if selected_slot < swap_slot:
                self._general_button_press('RIGHT')
            elif selected_slot > swap_slot:
                self._general_button_press('LEFT')

            selected_slot = get_selected_pokemon_in_swap_screen(self.cur_frame)

        # once we get to the chosen pokemon, we have to hit A twice to select it and send it out on the field
        self._general_button_press(['A', 'A'])
        logger.info(f'Swapping to slot {swap_slot}')

    def _damage_argmax(self, damage_values):
        max_idx = -1
        max_value = -np.inf
        for i, v in enumerate(damage_values):
            with v.get_lock():
                if v.value > max_value:
                    max_value = v.value
                    max_idx = i

        return max_idx, max_value

    def _do_search(self, savestate_queue, processes, damage_values):
        """This implements the search algorithm and returns the best result; it is agnostic enough to be used to select moves and also swap pokemon"""
        savestate_file = uuid.uuid4().hex + '.dst'

        # NOTE: I used to also delete the savestate file after searching, but then I kept getting weird synchronization issues around savestates so I stopped that
        self.savestate_path = os.path.join(SEARCH_SAVESTATE_DIR, savestate_file)
        self.env.emu.savestate.save_file(self.savestate_path)

        # see class docstring for the search algorithm here
        # this kicks off the search; each process will consume the savestate file and begin the search agent
        for q in savestate_queue:
            q.put(self.savestate_path, block=True)

        best_result = None
        completed_searches = 0
        while best_result is None and completed_searches < len(processes):
            result: SearchResultMessage = self.result_queue.get(block=True)
            completed_searches += 1

            won_battle = result.won
            if won_battle:
                best_result = result

                self.winning_move_list = result.moves
                self.pred_turns_to_win = result.turns
                self.winning_move_idx += 1

                logger.debug(f'Found a winning move, stopping early.')

            else:
                damage_dealt = result.damage_dealt

                _, max_damage = self._damage_argmax(damage_values)

                # if we're as good or better than any other search at this very moment, we can stop early
                if damage_dealt > max_damage:
                    best_result = result
                    logger.debug('Stopping early b/c we found a move that is the best so far.')

        if best_result is not None:
            # this tells all processes to halt whatever they were doing and add their results (at that point) to the queue
            self.stop_event.set()

        logger.debug(f'Damage values at the end of the search: {[v.value for v in damage_values]}')

        return best_result

    def _reset_search(self, battle=False, swap=False):
        """
        This function handles all of the multiprocessing stuff that we need to do to make sure that the search is ready for the next round
        Since there are different barriers when you're searching in-battle or when swapping pokemon,
          we can control what we are waiting for with the flags.
        """

        # once all processes, successfully add their results to the queue, this will pass
        #  and we can clean up the queue, damages, and event and be on with our day
        if battle:
            self.search_stop_barrier.wait()
        if swap:
            self.swap_stop_barrier.wait()

        # this likely won't be necessary, but I can imagine a situation where the processor is all locked up and so
        #  by the search process ends, we haven't even pulled the savestate file from the queue
        for file_queue in self.savestate_file_queues:
            while not file_queue.empty():
                file_queue.get()

        for file_queue in self.swap_savestate_file_queues:
            while not file_queue.empty():
                file_queue.get()

        while not self.result_queue.empty():
            self.result_queue.get()

        for damage_value in self.damage_values:
            with damage_value.get_lock():
                damage_value.value = 0
        for damage_value in self.swap_damage_values:
            with damage_value.get_lock():
                damage_value.value = 0

        self.stop_event.clear()

        # it's polite to clean up the savestate dir after finishing the search
        # NOTE: this used to be in _search, but I kept getting issues like "could not load savestate" that I think are
        #  caused by synchronization issues so I moved it here.
        if os.path.exists(self.savestate_path):
            os.remove(self.savestate_path)

    def _reset_winning_moves(self):
        """This function is used to clear the movelist that we follow when we found a winner."""
        self.winning_move_list = None
        self.winning_move_idx = 0
        self.pred_turns_to_win = 0

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    agent = BattleTowerSearchV2Agent(
        render=True,
        depth=1,
        search_swap=False,
        savestate_file='../ROM/Hopefully Faint.dst'
        #db_interface=BattleTowerServerDBInterface()
    )

    agent.play()
