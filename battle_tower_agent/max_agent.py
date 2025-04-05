import os

import numpy as np
import cv2

from battle_tower_agent.agent import (
    BattleTowerAgent,
    DATA_DIR,
    REF_IMG_DIR,
    BATTLE_TOWER_SAVESTATE,
    get_opponent_pokemon_info,
    get_cur_pokemon_info, TowerState, in_battle, in_move_select, pokemon_is_fainted,
    is_next_opponent_box, at_save_battle_video, opp_pokemon_is_out, our_pokemon_is_out
)
from battle_tower_agent.battle_tower_database.interface import BattleTowerDBInterface

# for the characters in a Pokemon's nameplate, each of them are 10 pixels high 6 pixels wide (except space but... I'll get to that)
CHAR_WIDTH = 6
CHAR_HEIGHT = 10
MAX_CHARS_PER_NAME = 10

# TODO: add 0
IDX_TO_CHAR = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-'
]

CHAR_IMAGE = cv2.imread(os.path.join(REF_IMG_DIR, 'letters', 'name_chars.png'))
INDIVIDUAL_CHARS = np.stack(np.split(CHAR_IMAGE, len(IDX_TO_CHAR), axis=1))

# Okay let me explain: every character in the name is 6 pixels wide except for spaces (which is 3 pixels)
#  so the method of greedy decoding doesn't work well. However this only applies to Mr. Mine and Mime Jr. (in Gen 4)
#  which is why I make a special case for them.
NAME_PREFIX_TO_PKMN = {
    'MR.': 'MR. MIME',
    'MIME': 'MIME JR.'
}

def extract_pokemon_name(info: np.ndarray) -> str:
    """Gets the name of a Pokemon from the info bar from a frame"""
    name_chars = info[:, :CHAR_WIDTH * MAX_CHARS_PER_NAME, :]

    # I originally did some fancy NP stuff and tbh, I don't really need to do that; this is plenty fast especially since it is called so rarely
    pkmn_name = ''

    for i in range(0, CHAR_WIDTH * MAX_CHARS_PER_NAME, CHAR_WIDTH):
        # cv2.imshow('h', name_chars[:, i:i+CHAR_WIDTH, :])
        # cv2.waitKey(0)
        # cv2.imshow('h', INDIVIDUAL_CHARS[6])
        # cv2.waitKey(0)

        matching_char = (name_chars[:, i:i+CHAR_WIDTH, :] == INDIVIDUAL_CHARS).all(axis=(1,2,3))
        if matching_char.sum() == 1: # if there are no matches for this part in the info bar, we get an empty list
            pkmn_name += IDX_TO_CHAR[matching_char.argmax()]

    pkmn_name = NAME_PREFIX_TO_PKMN.get(pkmn_name, pkmn_name)
    print(pkmn_name)
    return pkmn_name

class MaxDamageAgent(BattleTowerAgent):

    def __init__(self, render=True, savestate_file=BATTLE_TOWER_SAVESTATE, db_interface: BattleTowerDBInterface = None):
        super().__init__(render=render, savestate_file=savestate_file, db_interface=db_interface)

        self.cur_pkmn_info = None
        self.cur_pkmn_name = None

        self.opp_pkmn_info = None
        self.opp_pkmn_name = None

    def _get_pokemon_names(self, frame):
        # realistically, we don't even need to get the opponent's name in move select, just in-battle will work!
        #  this does mean that multi-turn moves (e.g. Outrage) won't get a new name whenever we cause a Pokemon to faint but that should be fine
        if self.state == TowerState.BATTLE:
            # we need to keep track of both our pokemon and our opponent's
            pokemon_info_position = our_pokemon_is_out(frame)
            if pokemon_info_position:
                cur_info = get_cur_pokemon_info(frame, position=pokemon_info_position)
                if self.cur_pkmn_name is None or (cur_info != self.cur_pkmn_info).any():
                    self.cur_pkmn_info = cur_info
                    self.cur_pkmn_name = extract_pokemon_name(cur_info)

            if opp_pokemon_is_out(frame):
                opp_info = get_opponent_pokemon_info(frame)
                if self.opp_pkmn_info is None or (opp_info != self.opp_pkmn_info).any():
                    self.opp_pkmn_info = opp_info
                    self.opp_pkmn_name = extract_pokemon_name(opp_info)

        # since we're kinda hacking `wait_for`, it needs to return a bool to indicate that the check failed
        return False

    def _select_move(self) -> int:
        print(self.cur_pkmn_name)
        print(self.opp_pkmn_name)
        return 0

    def _wait_for_battle_states(self):
        return self._wait_for(
            # NOTE: this is how we keep track of the opponent's name as closely as "real time" as possible (since checks are run every frame)
            # The watcher has to look *before* any other checks b/c once a check is found, no other checks are run
            (self._get_pokemon_names, TowerState.WAITING),
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

    def _run_battle_loop(self) -> TowerState:
        state = super()._run_battle_loop()

        # this is a good place to reset the info now that we're done with the battle
        self.opp_pkmn_info = None
        self.opp_pkmn_info = None

        return state

if __name__ == '__main__':
    agent = MaxDamageAgent(render=True)
    agent.play()