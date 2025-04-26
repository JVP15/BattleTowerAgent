import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid

from dataclasses import dataclass
from multiprocessing import Queue, Value, Event, Barrier
from typing import List, Tuple

import numpy as np

from battle_tower_agent.agent import (
    BattleTowerAgent,
    TowerState,
    POKEMON_MAX_MOVES,
    NUM_POKEMON_IN_SINGLES,
    SEARCH_SAVESTATE_DIR,
    in_battle,
    ROM_DIR,
    won_set,
    lost_set,
    pokemon_is_fainted,
    get_party_status,
    get_selected_pokemon_in_swap_screen,
    in_move_select,
    is_next_opponent_box,
    at_save_battle_video,
    opp_pokemon_is_out,
    get_opponent_hp_bar,
    get_opponent_pokemon_info, get_cur_pokemon_info, our_pokemon_is_out,
)
from battle_tower_agent.max_agent import BattleTowerMaxDamageAgent, dict_to_pokemon_sets_string, JS_TEMPLATE, \
    extract_pokemon_name, SCRIPT_DIR

from battle_tower_agent.search_agent import InvalidMoveSelected

from battle_tower_agent.battle_tower_database.interface import BattleTowerDBInterface, BattleTowerServerDBInterface
from battle_tower_agent.search_agent_v2 import BattleTowerSearchV2Agent, EarlySearchStop, SwappedPokemon, \
    SearchResultMessage, HPWatcher, SUBAGENT_STOPPING_TURN

DEFAULT_MOVE = 0

logger = logging.getLogger('SearchTowerAgent')


SEARCH_PROCESS_DIR = os.path.join(SEARCH_SAVESTATE_DIR, 'processes')
os.makedirs(SEARCH_PROCESS_DIR, exist_ok=True)


if os.name == 'nt':
    SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search V3 Team.dst')
else:
    SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search V3 Team Linux.dst')


class BattleTowerSearchV3SubAgentBattleTower(BattleTowerMaxDamageAgent):
    """This class is used by the BattleTowerSearchAgent to 'look ahead' for the next possible moves"""

    def __init__(self,
                 savestate_file: str,
                 moves: int | list[int],
                 team: dict,
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
        super().__init__(render=False, savestate_file=savestate_file, team=team)

        if isinstance(moves, int):
            moves = [moves]
        self.moves = moves

        self.turn_idx = 0
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
        self.turn_idx = 0

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
            (self._get_pokemon_names, TowerState.WAITING), # we also have to check the names (TODO: could be combined but right now, not necessary)
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

        if self.turn_idx >= SUBAGENT_STOPPING_TURN:
            logger.debug(f'Stopping Search Subagent because we hit turn search limit of {SUBAGENT_STOPPING_TURN}')
            raise EarlySearchStop()

        if self.turn_idx < len(self.moves):
            move = self.moves[self.turn_idx]
        else:
            move = super()._select_move()

        self.turn_idx += 1

        return move

    def _swap_to_next_pokemon(self):
        raise EarlySearchStop()

    def _act(self, action: str | None = None) -> np.ndarray:
        # _act is called the most frequently (basically every cycle) so it's the best place to check if we stop early
        if self.stop_event.is_set():
            raise EarlySearchStop()
        else:
            return super()._act(action)


class SearchProcessClient:
    """Manages a single search worker subprocess"""

    def __init__(self, session_id: str, idx: int, moves: list[int], team_file: str, swap_to: int | None = None):
        self.session_id = session_id
        self.idx = idx
        self.moves = moves
        self.swap_to = swap_to
        self.team_file = team_file

        # Create process-specific directory
        self.process_dir = os.path.join(SEARCH_PROCESS_DIR, f"{session_id}_{idx}")
        os.makedirs(self.process_dir, exist_ok=True)

        # File paths for communication
        self.result_file = os.path.join(self.process_dir, 'result.json')
        self.damage_file = os.path.join(self.process_dir, 'damage.txt')
        self.done_file = os.path.join(self.process_dir, 'done.txt')
        self.stop_file = os.path.join(self.process_dir, 'stop.txt')

        # Process state
        self.process = None
        self.is_running = False
        self.current_damage = 0

        # Thread for monitoring damage
        self.damage_monitor_thread = None
        self.stop_monitoring = threading.Event()

    def start(self, savestate_file: str):
        """Start the search process"""
        # Clean any existing files
        for file_path in [self.result_file, self.damage_file, self.done_file, self.stop_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Build command for the search worker
        worker_script = os.path.join(os.path.dirname(__file__), 'search_worker_v3.py')
        cmd = [
            sys.executable, worker_script,
            '--savestate', savestate_file,
            '--moves', ','.join(map(str, self.moves)),
            '--team', self.team_file,
            '--result-file', self.result_file,
            '--damage-file', self.damage_file,
            '--done-file', self.done_file,
            '--stop-file', self.stop_file
        ]

        if self.swap_to is not None:
            cmd.extend(['--swap-to', str(self.swap_to)])

        # Launch the worker process
        self.process = subprocess.Popen(cmd)
        self.is_running = True

        # Start monitoring damage file
        self.stop_monitoring.clear()
        self.damage_monitor_thread = threading.Thread(target=self._monitor_damage)
        self.damage_monitor_thread.daemon = True
        self.damage_monitor_thread.start()

    def stop(self):
        """Stop the search process"""
        if self.is_running:
            # Signal the process to stop
            with open(self.stop_file, 'w') as f:
                f.write('stop')

            # Stop monitoring thread
            self.stop_monitoring.set()
            if self.damage_monitor_thread and self.damage_monitor_thread.is_alive():
                self.damage_monitor_thread.join(timeout=1.0)

            # Wait for process to exit
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                # Force terminate if still running
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()

            self.is_running = False

    def get_result(self) -> SearchResultMessage | None:
        """Get the search result if available"""
        if os.path.exists(self.result_file):
            try:
                with open(self.result_file, 'r') as f:
                    data = json.load(f)
                return SearchResultMessage(
                    won=data['won'],
                    damage_dealt=data['damage_dealt'],
                    moves=data['moves'],
                    turns=data['turns'],
                    swap_to=data.get('swap_to')
                )
            except Exception as e:
                logger.error(f"Error loading result file: {e}")
        return None

    def is_done(self) -> bool:
        """Check if the search process is complete"""
        return os.path.exists(self.done_file)

    def _monitor_damage(self):
        """Monitor the damage file for updates"""
        while not self.stop_monitoring.is_set():
            if os.path.exists(self.damage_file):
                try:
                    with open(self.damage_file, 'r') as f:
                        damage_str = f.read().strip()
                        if damage_str:
                            self.current_damage = int(damage_str)
                except Exception:
                    pass  # Ignore read errors, will try again

            # there is *definitely* a better way to do this but I want to move fast
            #  b/c this branch isn't even that important
            time.sleep(0.1)

    def cleanup(self):
        """Clean up process resources"""
        self.stop()
        if os.path.exists(self.process_dir):
            try:
                shutil.rmtree(self.process_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up process directory: {e}")


GARCHOMP_SUICUNE_SCIZOR_TEAM = {
    "Garchomp": {
        "item": "Focus Sash",
        "ability": "Sand Veil",
        "evs": {"hp": 4, "atk": 252, "spe": 252},
        "nature": "Jolly",
        "moves": [
            "Earthquake", # I put EQ first b/c all else considered, if both it and Outrage will KO, that is fine w/ me.
            "Outrage",
            "Flamethrower",
            "Swords Dance"
        ]
    },
    "Suicune": {
        "item": "Leftovers",
        "ability": "Pressure",
        "evs": {
            "hp": 252,
            "def": 252,
            "spd": 4
        },
        "nature": "Bold",
        "ivs": {"atk": 0},
        "moves": [
            "Surf",
            "Ice Beam",
            "Calm Mind",
            "Toxic"
        ]
    },
    "Scizor": {
        "item": "Choice Band",
        "ability": "Technician",
        "evs": {"hp": 252, "atk": 252,  "spd": 4 },
        "nature": "Adamant",
        "moves": [
            "Bullet Punch",
            "Bug Bite",
            "Aerial Ace",
            "Superpower"
        ]
    }
}


# b/c we have custom search code for linux, I subclass the MaxDamageAgent here
class BattleTowerSearchV3Agent(BattleTowerMaxDamageAgent):
    """
    V3 Strategy:
    Basically the V2 strategy, except the subagent uses the "max_damage" rules (i.e. it doesn't just hit A)
    """

    def __init__(self,
        render=False,
        savestate_file=SEARCH_TEAM_SAVESTATE,
        db_interface: BattleTowerDBInterface = None,
        depth=1,
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
        self.team_dict = team

        super().__init__(render, savestate_file, db_interface, team=team)

        self.strategy = f'search_v3_depth_{depth}'

        # we're just going to pass most things via files instead of worrying about stin/out
        self.team_file = os.path.join(SEARCH_PROCESS_DIR, f"team_{uuid.uuid4().hex}.json")
        with open(self.team_file, 'w') as f:
            json.dump(team, f)

        # Tracking winning moves
        self.winning_move_list = None
        self.winning_move_idx = 0
        self.pred_turns_to_win = 0

        self.savestate_path = None
        self.session_id = uuid.uuid4().hex

        # Generate possible moves based on search depth
        self.depth = depth
        if depth == 1:
            self.possible_moves = [[move] for move in range(POKEMON_MAX_MOVES)]
        elif depth == 2:
            self.possible_moves = [[first, second] for first in range(POKEMON_MAX_MOVES) for second in
                                   range(POKEMON_MAX_MOVES)]
        else:
            raise NotImplementedError("Only depths 1 and 2 are supported")

        # Create search processes
        self.search_processes = []
        for i, move_list in enumerate(self.possible_moves):
            process = SearchProcessClient(self.session_id, i, move_list, self.team_file)
            self.search_processes.append(process)

        # Track healthy Pokemon
        self.healthy_pokemon = NUM_POKEMON_IN_SINGLES

    def _run_battle_loop(self) -> TowerState:
        state = super()._run_battle_loop()
        self._reset_winning_moves() # IMPORTANT: need to reset the winning move list after each battle
        self.healthy_pokemon = NUM_POKEMON_IN_SINGLES # also need to reset the # of healthy pokemon here

        return state

    def _do_search(self, processes: List[SearchProcessClient]) -> SearchResultMessage | None:
        """Perform the search and return the best result"""
        # Create a temporary savestate
        savestate_file = uuid.uuid4().hex + '.dst'
        self.savestate_path = os.path.join(SEARCH_SAVESTATE_DIR, savestate_file)
        self.env.emu.savestate.save_file(self.savestate_path)

        # Start all search processes
        for process in processes:
            process.start(self.savestate_path)

        # Monitor processes for results
        best_result = None
        completed_searches = 0
        total_processes = len(processes)

        while best_result is None and completed_searches < total_processes:
            # Check for completed processes
            for process in processes:
                if process.is_running and process.is_done():
                    result = process.get_result()
                    completed_searches += 1

                    if result and result.won:
                        # Found a winning move
                        best_result = result

                        self.winning_move_list = result.moves
                        self.pred_turns_to_win = result.turns
                        self.winning_move_idx += 1

                        logger.debug(f'Found a winning move, stopping early.')
                        break
                    # if we didn't win, check if it's the best result *so far*
                    # which is typically good enough
                    elif result:
                        _, max_damage = self._damage_argmax(processes)
                        if result.damage_dealt >= max_damage:
                            best_result = result
                            logger.debug('Found the best move so far, stopping early.')
                            break

            # TODO: there is probably a better way here (likely Threading Events or Queues)
            #   than just polling and sleeping
            time.sleep(0.01)

        # Stop all remaining processes
        for process in processes:
            if process.is_running:
                process.stop()

        # Clean up the savestate file
        if os.path.exists(self.savestate_path):
            try:
                os.remove(self.savestate_path)
            except Exception as e:
                logger.warning(f"Failed to remove savestate file: {e}")

        # Log the final damage values
        damages = [p.current_damage for p in processes]
        logger.debug(f'Damage values at the end of the search: {damages}')

        return best_result


    def _select_move(self) -> int:
        if self.winning_move_idx > self.pred_turns_to_win:
            self._reset_winning_moves()
            logger.info(f"We're on turn {self.winning_move_idx} of the 'winning' move list but expected to win in {self.pred_turns_to_win}, restarting search.")

        # if we haven't "won" yet in a search, we have to keep searching every turn
        if self.winning_move_list is None:
            logger.debug(f'Searching over {self.possible_moves}')

            best_result = self._do_search(self.search_processes)

            if best_result:
                move = best_result.moves[0]
                log_str = f'After searching with a depth of {self.depth}, move {move} did {best_result.damage_dealt} damage in {best_result.turns} turns.'
                if best_result.won:
                    log_str += f' Won the game. Following {best_result.moves} for the rest of the game.'
                    self.winning_move_list = best_result.moves
                    self.pred_turns_to_win = best_result.turns
                    self.winning_move_idx = 1  # We're already using the first move
                logger.info(log_str)
            else:
                max_idx, max_damage = self._damage_argmax(self.search_processes)
                move = self.possible_moves[max_idx][0]
                logger.info(f'After searching with a depth of {self.depth}, '
                            f'choosing move {move}, which led to {max_damage} damage dealt.')
        # when we have found a winning search, we just use that cached move list/whatever the max damage is
        else:
            if self.winning_move_idx < len(self.winning_move_list):
                move = self.winning_move_list[self.winning_move_idx]
                self.winning_move_idx += 1
            else:
                # Fall back to max damage if we're past our winning move list
                move = super()._select_move()

            logger.info(f'Using move {move} from cached winning move list {self.winning_move_list}')

        return move

    def _damage_argmax(self, processes: List[SearchProcessClient]) -> Tuple[int, int]:
        """Get the index and value of the process with the highest damage"""
        max_idx = -1
        max_value = -np.inf

        for i, process in enumerate(processes):
            damage = process.current_damage
            if damage > max_value:
                max_value = damage
                max_idx = i

        return max_idx, max_value

    def _reset_winning_moves(self):
        """This function is used to clear the movelist that we follow when we found a winner."""
        self.winning_move_list = None
        self.winning_move_idx = 0
        self.pred_turns_to_win = 0

    def cleanup(self):
        """Clean up all resources"""
        # Stop all processes
        for process in self.search_processes:
            process.cleanup()

        # Clean up session directory
        session_dir = os.path.join(SEARCH_PROCESS_DIR, self.session_id)
        if os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up session directory: {e}")

        # Clean up team file
        if os.path.exists(self.team_file):
            try:
                os.remove(self.team_file)
            except Exception as e:
                logger.warning(f"Failed to remove team file: {e}")



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    agent = BattleTowerSearchV3Agent(
        render=False,
        depth=1,
        #db_interface=BattleTowerServerDBInterface()
    )

    agent.play()
