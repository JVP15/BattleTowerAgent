import logging
import multiprocessing
import os
import uuid
from multiprocessing import Queue

from battle_tower_agent.agent import (
    BattleTowerAgent,
    TowerState,
    POKEMON_MAX_MOVES,
    in_battle, ROM_DIR, won_set, lost_set,
    SEARCH_SAVESTATE_DIR,
)

from battle_tower_agent.battle_tower_database.interface import BattleTowerDBInterface

DEFAULT_MOVE = 0

if os.name == 'nt':
    SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search Team.dst')
else:
    SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search Team Linux.dst')

logger = logging.getLogger('SearchTowerAgent')

class InvalidMoveSelected(Exception):
    pass

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

    def _select_move(self) -> int:
        # The search subagent starts by making each move in-order, and once we've gotten past the moves that we
        #  want to search over, we go back to using the 'default' move (i.e. the first one, which is as we saw with the 'A' agent, is pretty solid)

        if self.move_idx < len(self.moves):
            move = self.moves[self.move_idx]
        else:
            move = DEFAULT_MOVE

        self.move_idx += 1

        return move

def search_moves(savestate_file: str, moves: list[int], search_queue: Queue) -> tuple[bool, list[int], int]:
    """
    Given the savestate file, plays the remainder of the game until it reaches a stopping point.
    Adds the result (a bool if the game was won (true if it won, false if it lost or stopped early), the move list, and also the # of turns played out) to the provided multiprocessing queue
    Requires the filename (str) and list of moves (ints) to be provided as a tuple b/c of the `map` requirements
    NOTE: this must be called in a new process or else Desmume will complain about already being initialized
    """

    agent = BattleTowerSearchSubAgent(savestate_file, moves)

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

class BattleTowerSearchAgent(BattleTowerAgent):

    def __init__(self,
        render=False,
        savestate_file=SEARCH_TEAM_SAVESTATE,
        db_interface: BattleTowerDBInterface = None,
        depth=1,
        team=None,
    ):
        """
        Creates the Battle Tower Search Agent (v1).
        The strategy is to just search over all available moves until the end of the battle.
        Args:
            render: Whether to display the battle as it's going on.
            savestate_file: The initial savestate file that the agent loads the game from.
                There is a somewhat intricate setup needed to run the agent, so I don't recommend changing this.
            db_interface: A BattleTowerDB Interface, letting the agent record it's stats to a DB as it is playing
                (by default it is None, so it won't  record anything).
            depth: how many combinations of moves that we'll try, although it's more like a class than an actual # of steps we'll go down the tree
                If depth is 1 or 2, it's all possible permutations of move 1 or 2 nodes down the tree.
                Actually, only depths of 1 and 2 are supported.
            team: The team (in Pokemon Showdown format) used in the battle tower.
                If none, goes with the team that is chosen with the default savestate.
        """
        super().__init__(render, savestate_file, db_interface)

        self.depth = depth
        self.strategy = f'search_depth_{depth}'

        # on linux, desmume needs to use forkserver (maybe spawn is acceptable? haven't tested), but on windows, the default works just fine
        self.mp_context = multiprocessing.get_context('spawn' if os.name == 'posix' else None)

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

    def _select_move(self) -> int:
        savestate_file = uuid.uuid4().hex + '.dst'
        savestate_path = os.path.join(SEARCH_SAVESTATE_DIR, savestate_file)
        self.env.emu.savestate.save_file(savestate_path)

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

        for move_list in possible_moves:
            p = self.mp_context.Process(target=search_moves, args=(savestate_path, move_list, result_queue))
            search_processes.append(p)
            p.start()

        winning_result = None
        completed_processes = 0
        while winning_result is None and completed_processes < len(search_processes):
            result = result_queue.get(block=True) # result looks like (won, move_list, # of turns)
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

        # it's polite to clean up the savestate dir after finishing the search
        if os.path.exists(savestate_path):
            os.remove(savestate_path)

        return move

# Other optimization notes (for speech or accuracy, maybe turn these into slight variations?):
# 1. Maybe only do a search for the very first turn, but then also re-do the search whenever you have to swap Pokemon
# 2. For easier battles (e.g. before battle 21) only do search for the first N moves, and then just go from there
# 3. Don't do *any* search for the first 20 battles b/c there is only a small chance (~10%) that it doesn't get to battle 21
# 4. After receiving the 1st win, keep searching for either first_sample time * DURATION_MOD, or wait for N more wins and then just go w/ 'default' move
# 5. Figure out some way to go w/ the 'last used move' instead of the 'default' move (but what about status moves?)
# 6. Keep track of *how many times* a move led to a win (maybe even re-running the same move combo multiple times) and choosing the 'best' move that way
# 7. What about random search? Does randomly choosing a move during search make a difference?
# 7.5 WHAT ABOUT CHOOSING "MOST EFFECTIVE" MOVE (based on type combo and damage mod) Implement this in v3 agent
# 8. BIG OPTIMIZATION: keep the processes alive and more specifically desmume; loading a savestate is pretty quick, but there is a bit of a delay whenever you start up desmume
# 9. When doing a depth of 2, use *both* moves, don't just use the first move (which means it'll take 2 turns)

if __name__ == '__main__':
    agent = BattleTowerSearchAgent(
        render=True,
        depth=1,
        #db_interface=BattleTowerServerDBInterface()
    )

    agent.play()