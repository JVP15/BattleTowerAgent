import json
import os
import subprocess
import tempfile
import logging

import numpy as np
import cv2

from battle_tower_agent.agent import (
    BattleTowerAgent,
    DATA_DIR,
    ROM_DIR,
    REF_IMG_DIR,
    get_opponent_pokemon_info,
    get_cur_pokemon_info, TowerState, in_battle, in_move_select, pokemon_is_fainted,
    is_next_opponent_box, at_save_battle_video, opp_pokemon_is_out, our_pokemon_is_out
)
from battle_tower_agent.battle_tower_database.interface import BattleTowerDBInterface, BattleTowerServerDBInterface

SCRIPT_DIR = os.path.join(DATA_DIR, 'damage_calculator')
os.makedirs(SCRIPT_DIR, exist_ok=True)

if os.name == 'nt':
    MAX_DAMAGE_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Max Damage.dst')
else:
    MAX_DAMAGE_TEAM_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Max Damage Linux.dst')

logger = logging.getLogger('MaxPower') # it's a reference

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

    return pkmn_name

def dict_to_pokemon_sets_string(pokemon_dict):
    """
    Basically for this, I define the team as a dictionary (unlike everywhere else where it's a string)
    But I still need a string representation for logging
    """
    sets_text = ""
    for pokemon_name, pokemon_data in pokemon_dict.items():
        name_item_line = pokemon_name
        if pokemon_data.get('item'):
            name_item_line += f" @ {pokemon_data['item']}"

        ability_line = f"Ability: {pokemon_data.get('ability', '')}"
        evs_line = "EVs: "
        ev_strings = []
        for stat, amount in pokemon_data.get('evs', {}).items():
            ev_strings.append(f"{amount} {stat}")
        evs_line += " / ".join(ev_strings)

        nature_line = f"{pokemon_data.get('nature', 'Serious')} Nature" # serious is the default nature on Showdown

        moves_lines = ""
        for move in pokemon_data.get('moves', []):
            moves_lines += f"- {move}\n"

        ivs_line = ""
        if pokemon_data.get('ivs'): # Handle missing ivs
            ivs_strings = []
            for stat, amount in pokemon_data['ivs'].items():
                ivs_strings.append(f"{amount} {stat}")
            ivs_line = "IVs: " + ", ".join(ivs_strings) + "\n"

        pokemon_set_text = f"{name_item_line}\n{ability_line}\n{evs_line}\n{nature_line}\n"
        if ivs_line:
            pokemon_set_text += ivs_line
        pokemon_set_text += moves_lines.strip() + "\n\n" # strip to remove last newline from moves and add double newline to separate pokemon

        sets_text += pokemon_set_text

    return sets_text.strip() # remove trailing double newline


GARCHOMP_SUICUNE_SCIZOR_TEAM = {
    "Garchomp": {
        "item": "Focus Sash",
        "ability": "Sand Veil",
        "evs": {"hp": 4, "atk": 252, "spe": 252},
        "nature": "Jolly",
        "moves": [
            "Outrage",
            "Earthquake", # I want to put EQ first b/c it should be chosen more often but... see my explanation in _select_move
            "Flamethrower",
            "Crunch"
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
        "nature": "Calm",
        "ivs": {"atk": 0},
        "moves": [
            "Surf",
            "Ice Beam",
            "Signal Beam",
            "Extrasensory"
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

JS_TEMPLATE = """
// requires you to `npm install @pkmn/dev @pkmn/data @smogon/calc` 
const {{Dex}} = require('@pkmn/dex');
const {{Generations}} = require('@pkmn/data');
// see https://github.com/smogon/damage-calc/issues/465 
const {{calculate, Pokemon, Move}} = require('@smogon/calc/dist/adaptable.js');

const genNum = 4; // Generation
const our_pkmn_name = '{our_pkmn_name}';
const our_pkmn_details = {our_pkmn_details_json};
const opponent_pkmn_name = '{opponent_pkmn_name}';

const gens = new Generations(Dex);
const gen = gens.get(genNum);
const level = 50; // BattleTower is set at level 50

const results = {{ }}; // Store minimum damage per move

try {{
    const opponent_species = gen.species.get(opponent_pkmn_name);
    if (!opponent_species) {{ // I like to check stuff here just in case the name parser failes
        throw new Error(`Opponent species '${{opponent_pkmn_name}}' not found in Gen ${{genNum}}.`);
    }}
    // TBH I don't know what type `abilities` are but it logs like this {{0: 'Ability1', 1: 'Ability2'}}
    // calling `Object.values turns it into `[Ability1, Ability2]` though so... yay?
    const possible_opp_abilities = Object.values(opponent_species.abilities);

    // Attacker Pokemon object (created once)
    const attacker = new Pokemon(gen, our_pkmn_name, {{
        item: our_pkmn_details.item,
        ability: our_pkmn_details.ability,
        nature: our_pkmn_details.nature,
        evs: our_pkmn_details.evs,
        level: level
    }});

    for (const moveName of our_pkmn_details.moves) {{
        let min_damage_for_move = Infinity; // Track min damage for this move across *all* possible abilities

        const move = new Move(gen, moveName);

        // Opponents (like Bronzong) may have several damage-relevant abilities (stupid levitate and heat proof bronzong)
        for (const opp_ability of possible_opp_abilities) {{
            // Defender Pokemon object (created for each opponent ability)
            const defender = new Pokemon(gen, opponent_pkmn_name, {{
                ability: opp_ability,
                level: level
            }});

            const calculation_result = calculate(gen, attacker, defender, move);

            // I only really care about minimum damage here
            // calculation_result.damage can be a number (0 for immunities/wonder guard) or array (range)
            let current_min = 0;
            if (Array.isArray(calculation_result.damage) && calculation_result.damage.length > 0) {{
                current_min = calculation_result.damage[0];
            }} else if (typeof calculation_result.damage === 'number') {{
                 current_min = calculation_result.damage;
            }} else {{
                // It really *shouldn't* do this but you never know...
                console.error(`DEBUG: Move=${{moveName}}, OppAbility=${{opp_ability}}, MinDamage=${{current_min}}`);
            }}
            
            // I want the damages normalized by HP so that we can easily compare > or < than 1
            // and okay I know that this doesn't take into account EVs but... I am okay sacrificing that for simpler code
            current_min /= defender.originalCurHP;

            if (current_min < min_damage_for_move) {{
                min_damage_for_move = current_min;
            }}
        }}

        // Store the overall minimum damage for this move
        // If min_damage_for_move is still Infinity, it means no damage was calculated (e.g., immunity across all abilities)
        results[moveName] = min_damage_for_move;
    }}

    console.log(JSON.stringify(results));

}} catch (error) {{
    // Output error to stderr for Python to potentially catch
    console.error(`JavaScript Error: ${{error.message}}`);
    console.error(error.stack); // More detailed stack trace
    process.exit(1); // Signal error exit code
}}
"""



class MaxDamageAgent(BattleTowerAgent):

    strategy = 'max_damage'

    def __init__(
        self,
        render=True,
        savestate_file=MAX_DAMAGE_TEAM_SAVESTATE,
        db_interface: BattleTowerDBInterface = None,
        team=GARCHOMP_SUICUNE_SCIZOR_TEAM,
    ):
        super().__init__(render=render, savestate_file=savestate_file, db_interface=db_interface)

        self.cur_pkmn_info = None
        self.cur_pkmn_name = None

        self.opp_pkmn_info = None
        self.opp_pkmn_name = None

        self.move_damage_cache = None

        self.team = dict_to_pokemon_sets_string(team)
        self.team_dict = team

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

                    # anytime we change or our opponent changes pokemon, we have to re-calculate the move damages
                    self.move_damage_cache = None

            if opp_pokemon_is_out(frame):
                opp_info = get_opponent_pokemon_info(frame)
                if self.opp_pkmn_info is None or (opp_info != self.opp_pkmn_info).any():
                    self.opp_pkmn_info = opp_info
                    self.opp_pkmn_name = extract_pokemon_name(opp_info)

                    self.move_damage_cache = None

        # since we're kinda hacking `wait_for`, it needs to return a bool to indicate that the check failed
        return False

    def _calculate_move_damages(self):
        """
        Calculates the damage (to be conservative, I take the min) for each move against an opponent,
        considering all opponent abilities. Damages are normalized so that 0 is no damage and 1 is a full HP bar (can be > than 1)

        Returns:
            dict: A dictionary mapping move names to calculated damage or None if an error occurred.

        NOTE: I'm working w/ a dict right now b/c it's nice to see how much each move does, but this may change to an array in the future.
        """
        pkmn_name = self.cur_pkmn_name.title() # it's normally in ALL_CAPS but the names in the dict are not
        details_json = json.dumps(self.team_dict[pkmn_name])

        # Fill the JavaScript template
        js_code = JS_TEMPLATE.format(
            our_pkmn_name=pkmn_name,
            our_pkmn_details_json=details_json,
            opponent_pkmn_name=self.opp_pkmn_name
        )

        # I just found out that tempfile exists, maybe I should use it for savestates too...
        js_file = None
        damage_results = None
        try:
            # Create a temporary file with .js extension
            # delete=False is important on Windows, manually delete later
            js_file = tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8', dir=SCRIPT_DIR)
            js_filepath = js_file.name

            js_file.write(js_code)
            js_file.close()  # Close the file handle before letting node access it

            # Ensure node is in your PATH or provide the full path
            process = subprocess.run(
                ['node', js_filepath],
                capture_output=True,
                text=True,  # Decodes stdout/stderr automatically
                check=False,
                encoding='utf-8'
            )

            # the code *shouldn't* break (since it already runs) but... ya never know
            if process.returncode != 0:
                logger.warning(f"Error executing Node.js script:\n {process.stderr}")
            else:
                damage_results = json.loads(process.stdout)

        except FileNotFoundError:
            logger.error("Error: 'node' command not found. Is Node.js installed and in your PATH?")
        finally:
            # --- Cleanup ---
            if js_file and os.path.exists(js_filepath):
                try:
                    os.remove(js_filepath)
                except OSError as e:
                    logger.warning(f"Warning: Could not remove temporary file {js_filepath}: {e}")

        return damage_results


    def _select_move(self) -> int:
        if self.move_damage_cache is None:
            damage_results = self._calculate_move_damages()
            if damage_results is None: # gotta check for an error here
                return 0 # just go w/ the first and "best" move
            # calculate move damages is a dict but to speed things along, we only really need an array
            self.move_damage_cache = np.array(list(damage_results.values()))

            logger.debug(f'{self.cur_pkmn_name} vs {self.opp_pkmn_name} move damage: {damage_results}')

            # it doesn't really matter how much move we do if it'll KO the opponent (TODO: handle possible EVs)
            self.move_damage_cache[self.move_damage_cache > 1] = 1

        # okay due to the problem below this... doesn't actually get chosen but that's a bug for another day
        move = np.argmax(self.move_damage_cache)

        return move

    def _execute_move(self, move: int) -> TowerState:
        # I put some custom logic here to go down the move list in order of most-> least damaging move
        #  instead of just choosing the next move in the list
        if self.move_damage_cache is None:
            return super()._execute_move(move)

        move_cache = self.move_damage_cache
        best_to_worst_moves = np.argsort(move_cache)[::-1] # argsort goes ascending, but we need descending

        # this is a bit of a pain; numpy preserves the ordering of elements while sorting, so if 0 and 1 both have the same value we'd get ... 0, 1
        # and since we want descending, it'd go 1, 0, 3, 4... which is why even though EQ is the 2nd move on the list, it's chosen more than outrage
        # which is a good thing in practise but I don't like this interaction

        state = self.state

        advanced_game = False
        for chosen_move in best_to_worst_moves:
            self._goto_move(chosen_move)

            self._general_button_press('A')
            state = self._wait_for_battle_states()

            # any other state but MOVE_SELECT means that the move 'worked' (i.e. advanced the game)
            # and so we can handle the logic of the next turn, otherwise we have to keep looping through the moves and trying them
            if state != TowerState.MOVE_SELECT:
                advanced_game = True
                self.state = TowerState.BATTLE  # this is important b/c we need to reset the state back to BATTLE

                break

        if not advanced_game:
            self._log_error_image('could_not_make_move', state)
            raise ValueError('Could not select a move while in move select (for some reason)')

        return state


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
    logging.basicConfig(level=logging.DEBUG)

    agent = MaxDamageAgent(
        render=True,
        db_interface=BattleTowerServerDBInterface()
    )

    agent.play()

