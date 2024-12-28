import os
import uuid

from battle_tower_agent import (
    BattleTowerAgent,
    BattleTowerAAgent,
    TowerState,
    POKEMON_MAX_MOVES,
    in_battle, ROM_DIR,
)
from battle_tower_database.interface import BattleTowerDBInterface

DEFAULT_MOVE = 1
SEARCH_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Search Team.dst')

class BattleTowerSearchSubAgent(BattleTowerAgent):
    """This class is used by the BattleTowerSearchAgent to 'look ahead' for the next possible moves"""

    strategy = 'move_select'
    def __init__(self, savestate_file: str, moves: list[int], team):
        super().__init__(render=False, savestate_file=savestate_file)

        self.moves = moves
        self.move_idx = 0
        self.team = team

    def play_remainder_of_battle(self) -> TowerState:
        """
        This function starts from the move_select screen and plays until one of these stopping conditions:
        1. the battle ends (either on a win or loss)
        2. we fail to select a move that is included in the search (but not the 'default' move)

        This is expected to be called in the _select_and_execute_move function of the SearchAgent
          and as such it expects the provided savestate to start it in move select.
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

        if self.moves < len(self.moves):
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
        savestate_file = uuid.uuid4().hex + '.dst' # remember to delete this later
        self.env.emu.savestate.save_file(savestate_file)

        state = self.state

        if self.depth >= 3:
            raise NotImplementedError("I don't currently have anything for swapping Pokemon yet")
