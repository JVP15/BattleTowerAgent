import logging
import multiprocessing
import os
import traceback
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
    get_opponent_pokemon_info,
)
from battle_tower_agent.max_agent import BattleTowerMaxDamageAgent, dict_to_pokemon_sets_string

from battle_tower_agent.search_agent import InvalidMoveSelected

from battle_tower_agent.battle_tower_database.interface import BattleTowerDBInterface, BattleTowerServerDBInterface
from battle_tower_agent.search_agent_v2 import BattleTowerSearchV2Agent, EarlySearchStop, SwappedPokemon, \
    SearchResultMessage, HPWatcher, SUBAGENT_STOPPING_TURN

DEFAULT_MOVE = 0

logger = logging.getLogger('SearchTowerAgent')

if os.name == 'nt':
    SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search V3 Team.dst')
else:
    raise NotImplementedError("We don't have a linux savestate for this yet")
    #SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search Team Linux.dst')


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
        super().__init__(render=True, savestate_file=savestate_file, team=team)

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



def init_search_process(
        savestate_queue: Queue,
        moves: int | list[int],
        swap_to: int,
        team: dict,
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
            agent = BattleTowerSearchV3SubAgentBattleTower(savestate_file, moves=moves, team=team, swap_to=swap_to, stop_event=early_stop_event,
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
            logger.warning(traceback.format_exc())

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

        result = SearchResultMessage(won=won, damage_dealt=damage_dealt, moves=moves, turns=agent.turn_idx, swap_to=swap_to)
        search_queue.put(result)

        # once we've submitted our results, we just have to wait for all other processes to clear up
        search_stop_barrier.wait()


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


class BattleTowerSearchV3Agent(BattleTowerSearchV2Agent):
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
        team_str = dict_to_pokemon_sets_string(team)
        self.team_dict = team

        super().__init__(render, savestate_file, db_interface, team=team_str, search_swap=False) # we don't support swapping yet

        self.strategy = f'search_v3_depth_{depth}'

    # this gets called in the search agent's init fn
    def _start_search_processes(self):
        # TODO: there seems to be problems with mutliprocessing on WSL (at least for windows 10), investigate later
        ctx = multiprocessing.get_context('forkserver' if os.name == 'posix' else None)
        for i, move_list in enumerate(self.possible_moves):
            p = ctx.Process(
                target=init_search_process, # args look like savestate_queue, moves, swap_to (set to None), team (in dict form), result_queue, stop_event, stop_barrier, and damage
                args=(self.savestate_file_queues[i], move_list, None, self.team_dict, self.result_queue, self.stop_event, self.search_stop_barrier, self.damage_values[i])
            )
            self.processes.append(p)
            p.start()

        # TODO: support swapping


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    agent = BattleTowerSearchV3Agent(
        render=True,
        depth=1,
        #db_interface=BattleTowerServerDBInterface()
    )

    agent.play()
