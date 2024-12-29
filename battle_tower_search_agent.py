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
    in_battle, ROM_DIR, won_set, lost_set,
)

from battle_tower_database.interface import BattleTowerDBInterface

DEFAULT_MOVE = 0
SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search Team.dst')
SEARCH_TMP_SAVESTATE_DIR = os.path.join(ROM_DIR, 'search')

logger = logging.getLogger('SearchTowerAgent')

class BattleTowerSearchSubAgent(BattleTowerAgent):
    """This class is used by the BattleTowerSearchAgent to 'look ahead' for the next possible moves"""

    strategy = 'move_select'
    def __init__(self, savestate_file: str, moves: int | list[int]):
        super().__init__(render=True, savestate_file=savestate_file)

        if isinstance(moves, int):
            moves = [moves]

        self.moves = moves
        self.move_idx = 0
        self.team = '' # there is no logging to a DB for these searches, so we don't need to specify a team

    def play_remainder_of_battle(self) -> TowerState:
        """
        This function starts from the move_select screen and plays until one of these stopping conditions:
        1. the battle ends (either on a win or loss)
        2. we fail to select a move that is included in the search (but not the 'default' move)

        This is expected to be called in the _select_and_execute_move function of the SearchAgent
          and as such it expects the provided savestate to start it in move select.

        Returns the state after finishing the battle (same as `play_battle`)
        """
        self.move_idx = 0

        # since we're in move_select, but _run_battle_loop expects us to be in the fight screen, we have to do that;
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

        if self.move_idx < len(self.moves):
            move = self.moves[self.move_idx]
        else:
            move = DEFAULT_MOVE

        advanced_game = False

        # There's one slight snag, we may or may not be able to select the move (e.g. due to torment, choice specs)
        #  but you are *still* in move select, unlike certain other conditions
        # There's no (good) way to know until after we click it, so we've just got to keep trying until we get it
        # It's a *tad* inefficient, but it *is* compatible w/ searching and choice moves b/c:
        # 1. if we are searching over a set of moves that are different, we break
        # 2. if we are searching the same move consecutively, then even if it isn't the first move, we'll eventually goto the choice selected move.
        for i in range(POKEMON_MAX_MOVES):
            chosen_move = move + i
            self._goto_move(chosen_move)

            self._general_button_press('A')
            state = self._wait_for_battle_states()

            # if clicking on one of the moves that we're exploring didn't advance the game
            #   there is no point in continuing to search down that path
            if state == TowerState.MOVE_SELECT and self.move_idx < len(self.moves):
                return state

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
    agent = BattleTowerSearchSubAgent(savestate_file, moves)

    state = agent.play_remainder_of_battle()

    if state == TowerState.WON_BATTLE:
        won = True
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

class BattleTowerSearchAgent(BattleTowerAgent):

    def __init__(self,
        render=True,
        savestate_file=SEARCH_TEAM_SAVESTATE,
        db_interface: BattleTowerDBInterface = None,
        depth=1,
        team=None,
    ):
        """
        TODO: write the rest of this docstring
        depth is how many combinations of moves that we'll try, although it's more like a class than an actual # of steps we'll go down the tree
        If depth is 1 or 2, it's all possible permutations of move 1 or 2 nodes down the tree. If depth is 3, we also include swapping Pokemon
        """
        super().__init__(render, savestate_file, db_interface)

        self.depth = depth
        self.strategy = f'search_depth={depth}'
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
        # TODO: investigate early stopping methods, such as when we find a move combo that leads to a win,
        #   may only need to run 1-2 more simulations from there to make sure we win
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

        # to help w/ efficiency (b/c especially early on, it can take a while to 'lose' when you make a bad move; literally PP stalled against Shedinja)
        # as soon as I get the first 'winning' result, we're going with it (this also prioritizes moves that will help us win *fast*)
        search_processes = []
        result_queue = Queue()

        for move_list in possible_moves:
            p = Process(target=search_moves, args=(savestate_path, move_list, result_queue))
            search_processes.append(p)
            p.start()

        winning_result = None
        while winning_result is None:
            result = result_queue.get(block=True)

            if result[0]:
                winning_result = result

                for p in search_processes:
                    p.terminate()

        for p in search_processes: # multiprocessing thing; to prevent threads from becoming zombies, we join
            p.join()

        if winning_result:
            move = winning_result[1][0] # remember, the result is a tuple of (won, move_list, turns)
            logger.log(logging.INFO, f'After searching with a depth of {self.depth}, move {move} won in {winning_result[2]} turns')
        else:
            logger.log(logging.INFO, f'After searching with a depth of {self.depth}, could not find a winning move. Just picking {DEFAULT_MOVE}')
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

# Other optimization notes (for speech or accuracy):
# 1. Maybe only do a search for the very first turn, but then also re-do the search whenever you have to swap Pokemon
# 2. For easier battles (e.g. before battle 21) only do search for the first N moves, and then just go from there
# 3. Don't do *any* search for the first 20 battles b/c there is only a small chance (~10%) that it doesn't get to battle 21
# 4. After receiving the 1st win, keep searching for either first_sample time * DURATION_MOD, or wait for N more wins and then just go w/ 'default' move
# 5. Figure out some way to go w/ the 'last used move' instead of the 'default' move (but what about status moves?)
# 6. Keep track of *how many times* a move led to a win (maybe even re-running the same move combo multiple times) and choosing the 'best' move that way
# 7. What about random search?

if __name__ == '__main__':
    agent = BattleTowerSearchAgent(render=True)

    agent.play()

    # path = r'C:\Users\jorda\Documents\Python\CynthAI\GeminiPlaysPokemon\ROM\search\36522603ecd64c519d1bb6b5495b29fb.dst'
    # subagent = BattleTowerSearchSubAgent(path, [3, 0])
    # subagent.play_remainder_of_battle()