#!/usr/bin/env python3
import os
import subprocess
import sys
import json
import logging
import argparse
import tempfile
import traceback
import numpy as np
from dataclasses import dataclass, asdict

from battle_tower_agent.agent import (
    BattleTowerAgent,
    TowerState,
    POKEMON_MAX_MOVES,
    SEARCH_SAVESTATE_DIR,
    in_battle,
    won_set,
    lost_set,
    pokemon_is_fainted,
    get_selected_pokemon_in_swap_screen,
    in_move_select,
    is_next_opponent_box,
    at_save_battle_video,
    opp_pokemon_is_out,
    get_opponent_hp_bar,
    get_opponent_pokemon_info,
    get_cur_pokemon_info,
    our_pokemon_is_out,
)

from battle_tower_agent.max_agent import extract_pokemon_name, JS_TEMPLATE, SCRIPT_DIR, BattleTowerMaxDamageAgent
from battle_tower_agent.search_agent import InvalidMoveSelected
from battle_tower_agent.search_agent_v2 import SUBAGENT_STOPPING_TURN, EarlySearchStop, SwappedPokemon, \
    SearchResultMessage

logger = logging.getLogger('SearchWorkerV3')


class HPWatcher:
    """
    Tracks the HP bar of the opponent Pokemon in real-time.

    Custom b/c it logs to a file
    """

    def __init__(self, damage_file=None):
        self.opp_hp = None
        self.damage_dealt = 0
        self.opp_pkm = None
        self.damage_file = damage_file
        self.last_reported_damage = -1

    def __call__(self, cur_frame):
        if opp_pokemon_is_out(cur_frame):
            opp_hp = get_opponent_hp_bar(cur_frame)
            opp_pkmn = get_opponent_pokemon_info(cur_frame)

            if self.opp_pkm is None or (opp_pkmn != self.opp_pkm).any():
                self.opp_pkm = opp_pkmn
                self.opp_hp = opp_hp
            else:
                hp_diff = self.opp_hp - opp_hp
                self.damage_dealt += hp_diff
                self.opp_hp = opp_hp

                # Report damage to file if it changed
                if self.damage_file and self.damage_dealt != self.last_reported_damage:
                    try:
                        with open(self.damage_file, 'w') as f:
                            f.write(str(self.damage_dealt))
                        self.last_reported_damage = self.damage_dealt
                    except Exception as e:
                        logger.error(f"Error writing damage to file: {e}")

        return False  # Not a condition to stop waiting


class BattleTowerSearchV3SubAgent(BattleTowerMaxDamageAgent):
    """Agent to 'look ahead' for the next possible moves using max damage strategy"""

    strategy = 'move_select'

    def __init__(self,
                 savestate_file: str,
                 moves: int | list[int],
                 team: dict,
                 swap_to: int | None = None,
                 stop_file: str = None,
                 damage_file: str = None,
                 ):
        """
        Args:
            savestate_file: A path to a savestate file that the subagent will pick up the game from
            moves: the list of moves that the subagent will execute in order, can be a single move
            team: Team dictionary with Pokemon details
            swap_to: before making a move, swap to the pokemon in that idx (can be none)
            stop_file: Path to a file whose existence signals that the search should stop
            damage_file: Path to a file where damage updates will be written
        """
        super().__init__(render=False, savestate_file=savestate_file, team=team)

        if isinstance(moves, int):
            moves = [moves]

        self.moves = moves
        self.move_idx = 0
        self.swap_to = swap_to

        self.stop_file = stop_file
        self.hp_watcher = HPWatcher(damage_file=damage_file)

    def play_remainder_of_battle(self) -> TowerState:
        """Plays until win, pokemon swap, or battle end."""
        self.move_idx = 0

        # Get to the battle screen
        self.state = self._wait_for(
            (in_battle, TowerState.BATTLE),
            (pokemon_is_fainted, TowerState.SWAP_POKEMON),
            button_press='B'
        )

        # Handle swapping if needed
        if self.swap_to and not pokemon_is_fainted(self.cur_frame):
            self._log_error_image(message='not_in_swap_screen')
            raise ValueError("Not in pokemon swap screen when expecting to swap")
        elif self.swap_to:
            selected_slot = get_selected_pokemon_in_swap_screen(self.cur_frame)
            if selected_slot is None:
                self._general_button_press('A')
                selected_slot = get_selected_pokemon_in_swap_screen(self.cur_frame)

            # Navigate to the right slot
            while selected_slot != self.swap_to:
                if selected_slot < self.swap_to:
                    self._general_button_press('RIGHT')
                elif selected_slot > self.swap_to:
                    self._general_button_press('LEFT')
                selected_slot = get_selected_pokemon_in_swap_screen(self.cur_frame)

            # Select and send out
            self._general_button_press(['A', 'A'])

            self.state = self._wait_for(
                (in_battle, TowerState.BATTLE),
                button_press='B'
            )

        return self._run_battle_loop()

    def _wait_for_battle_states(self):
        return self._wait_for(
            (self.hp_watcher, TowerState.WAITING),
            (self._get_pokemon_names, TowerState.WAITING),
            (in_battle, TowerState.BATTLE),
            (in_move_select, TowerState.MOVE_SELECT),
            (pokemon_is_fainted, TowerState.SWAP_POKEMON),
            (is_next_opponent_box, TowerState.WON_BATTLE),
            (at_save_battle_video, TowerState.END_OF_SET),
            button_press='A',
            check_first=True,
        )


    def _select_move(self) -> int:
        """Select move based on search depth or max damage"""
        if self.move_idx >= SUBAGENT_STOPPING_TURN:
            logger.debug(f'Stopping Search Subagent because we hit turn limit of {SUBAGENT_STOPPING_TURN}')
            raise EarlySearchStop()

        if self.move_idx < len(self.moves):
            move = self.moves[self.move_idx]
        else:
            # Use max damage strategy for moves beyond the search depth
            move = super()._select_move()

        self.move_idx += 1
        return move

    def _swap_to_next_pokemon(self):
        """Stop search when a pokemon needs to be swapped"""
        raise EarlySearchStop()

    def _act(self, action: str | None = None) -> np.ndarray:
        """Check for stop signal on each action"""
        if self.stop_file and os.path.exists(self.stop_file):
            raise EarlySearchStop()
        return super()._act(action)


def serve_persistent(args):
    """Read JSON commands on stdin, run one search per line, write result & done files."""
    # Load the team once (avoid re-import delay)
    with open(args.team, 'r') as f:
        team_dict = json.load(f)

    while True:
        line = sys.stdin.readline()
        if not line:
            break  # EOF => exit
        if not line.strip():
            continue

        try:
            msg = json.loads(line)
        except Exception as e:
            logger.error(f"Error parsing command: {e}")
            continue

        savestate = msg.get('savestate')
        moves = msg.get('moves', [])
        swap_to = msg.get('swap_to')

        # Clean old files
        for fp in [args.result_file, args.done_file, args.damage_file]:
            if fp and os.path.exists(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass

        # Initialize damage file
        if args.damage_file:
            try:
                with open(args.damage_file, 'w') as f:
                    f.write('0')
            except Exception:
                pass

        try:
            agent = BattleTowerSearchV3SubAgent(
                savestate_file=savestate,
                moves=moves,
                team=team_dict,
                swap_to=swap_to,
                stop_file=args.stop_file,
                damage_file=args.damage_file
            )
            try:
                state = agent.play_remainder_of_battle()
            except (InvalidMoveSelected, SwappedPokemon, EarlySearchStop):
                state = TowerState.STOPPED_SEARCH
            except Exception as e:
                logger.warning(f"Unexpected error during search: {e}\n{traceback.format_exc()}")
                state = TowerState.LOST_SET

            # Determine win
            won = False
            if state == TowerState.WON_BATTLE:
                won = True
            elif state == TowerState.END_OF_SET:
                try:
                    final_state = agent._wait_for(
                        (won_set, TowerState.WON_SET),
                        (lost_set, TowerState.LOST_SET),
                        button_press='B',
                    )
                    if final_state == TowerState.WON_SET:
                        won = True
                except EarlySearchStop:
                    pass
                except Exception as e:
                    logger.warning(f"Error checking win/loss state: {e}")

            damage_dealt = agent.hp_watcher.damage_dealt
            if args.damage_file:
                with open(args.damage_file, 'w') as f:
                    f.write(str(damage_dealt))

            result = SearchResultMessage(
                won=won,
                damage_dealt=damage_dealt,
                moves=moves,
                turns=agent.move_idx,
                swap_to=swap_to
            )

            with open(args.result_file, 'w') as f:
                json.dump(asdict(result), f)

        except Exception as e:
            logger.error(f"Fatal error in persistent search worker: {e}\n{traceback.format_exc()}")
            try:
                with open(args.result_file, 'w') as f:
                    json.dump({
                        'won': False,
                        'damage_dealt': 0,
                        'moves': moves,
                        'turns': 0,
                        'swap_to': swap_to,
                        'error': str(e)
                    }, f)
            except Exception:
                pass

        finally:
            try:
                with open(args.done_file, 'w') as f:
                    f.write('done')
            except Exception:
                pass


def main():
    """Original single‐run entrypoint (preserved for backward‐compat)."""
    parser = argparse.ArgumentParser(description='Battle Tower Search Worker V3')
    parser.add_argument('--savestate', type=str, required=True, help='Path to savestate file')
    parser.add_argument('--moves', type=str, required=True, help='Comma-separated list of moves')
    parser.add_argument('--team', type=str, required=True, help='Path to team configuration JSON file')
    parser.add_argument('--swap-to', type=int, help='Pokemon slot to swap to')
    parser.add_argument('--stop-file', type=str, help='Path to stop signal file')
    parser.add_argument('--damage-file', type=str, help='Path to damage report file')
    parser.add_argument('--result-file', type=str, required=True, help='Path to write results')
    parser.add_argument('--done-file', type=str, required=True, help='Path to signal completion')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse moves
    moves = [int(m) for m in args.moves.split(',')]

    # Load team configuration
    with open(args.team, 'r') as f:
        team_dict = json.load(f)

    # Ensure proper directories exist
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.done_file), exist_ok=True)
    if args.damage_file:
        os.makedirs(os.path.dirname(args.damage_file), exist_ok=True)

    # Initialize damage file
    if args.damage_file:
        with open(args.damage_file, 'w') as f:
            f.write('0')

    try:
        # Create and run the agent
        agent = BattleTowerSearchV3SubAgent(
            savestate_file=args.savestate,
            moves=moves,
            team=team_dict,
            swap_to=args.swap_to,
            stop_file=args.stop_file,
            damage_file=args.damage_file
        )

        # Play the battle
        try:
            state = agent.play_remainder_of_battle()
        except (InvalidMoveSelected, SwappedPokemon, EarlySearchStop):
            state = TowerState.STOPPED_SEARCH
        except Exception as e:
            logger.warning(f'Unexpected error during search: {e}\n{traceback.format_exc()}')
            state = TowerState.LOST_SET

        # Determine if we won
        won = False
        if state == TowerState.WON_BATTLE:
            won = True
        elif state == TowerState.END_OF_SET:
            try:
                state = agent._wait_for(
                    (won_set, TowerState.WON_SET),
                    (lost_set, TowerState.LOST_SET),
                    button_press='B',
                )
                if state == TowerState.WON_SET:
                    won = True
            except EarlySearchStop:
                pass
            except Exception as e:
                logger.warning(f'Error checking win/loss state: {e}')

        # Final damage update
        damage_dealt = agent.hp_watcher.damage_dealt
        if args.damage_file:
            with open(args.damage_file, 'w') as f:
                f.write(str(damage_dealt))

        # Write results
        result = SearchResultMessage(
            won=won,
            damage_dealt=damage_dealt,
            moves=moves,
            turns=agent.move_idx,
            swap_to=args.swap_to
        )

        with open(args.result_file, 'w') as f:
            json.dump(asdict(result), f)

    except Exception as e:
        logger.error(f"Fatal error in search worker: {e}\n{traceback.format_exc()}")
        # Write an error result
        with open(args.result_file, 'w') as f:
            json.dump({
                'won': False,
                'damage_dealt': 0,
                'moves': moves,
                'turns': 0,
                'swap_to': args.swap_to,
                'error': str(e)
            }, f)

    finally:
        # Always signal completion
        with open(args.done_file, 'w') as f:
            f.write('done')


if __name__ == '__main__':
    # If you call with --persistent, we loop reading commands, otherwise we do one run and exit
    parser = argparse.ArgumentParser(description='Battle Tower Search Worker V3 (Persistent or Single‐Run)')
    parser.add_argument('--savestate', type=str, help='Path to savestate (single‐run mode)')
    parser.add_argument('--moves', type=str, help='Comma‐separated moves (single‐run mode)')
    parser.add_argument('--team', type=str, required=True, help='Path to team configuration JSON file')
    parser.add_argument('--swap-to', type=int, help='Pokemon slot to swap to (single‐run mode)')
    parser.add_argument('--stop-file', type=str, help='Path to stop signal file')
    parser.add_argument('--damage-file', type=str, help='Path to damage report file')
    parser.add_argument('--result-file', type=str, required=True, help='Path to write results')
    parser.add_argument('--done-file', type=str, required=True, help='Path to signal completion')
    parser.add_argument('--persistent', action='store_true', help='Keep process alive and read JSON commands on stdin')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='Logging level')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.persistent:
        serve_persistent(args)
    else:
        main()
