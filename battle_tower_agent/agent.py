import pathlib
import time
from collections import deque

import cv2
import numpy as np
import os
import datetime

from battle_tower_agent.battle_tower_database.interface import BattleTowerDBInterface, BattleTowerServerDBInterface
from battle_tower_agent.pokemon_env import PokemonEnv

from enum import Enum

import logging

# I have some debug statements that I want to include while running the program, but the tidal wave of button presses
#  drowns it out, which is why I created a new, even lower level logging method
logging.BUTTON_PRESS = 5
logging.addLevelName(logging.BUTTON_PRESS, 'BUTTON_PRESS')

logger = logging.getLogger('TowerAgent')

# roms should be in BattleTowerAgent/ROMS, and we should be in BattleTowerAgent/battle_tower_agent/agent.py
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
ROM_DIR = os.path.join(ROOT_DIR, 'ROM')

if os.name == 'nt': # windows
    BATTLE_TOWER_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower.dst')
else:
    BATTLE_TOWER_SAVESTATE = os.path.join(ROM_DIR, 'Pokemon - Platinum Battle Tower Linux.dst')

SEARCH_SAVESTATE_DIR = os.path.join(ROM_DIR, 'search')
os.makedirs(SEARCH_SAVESTATE_DIR, exist_ok=True)

# 3-5 cycles is a decent time to hold down a button for any normal press, and it takes around 12 cycles to walk (which isn't needed here thankfully)
BUTTON_PRESS_DURATION = 5
AFTER_PRESS_WAIT = 30 # I have no good justification for this, we just need to spend some amount of time waiting for the game to process our input

# a full battle may take about 20k frames, and if it takes that much time while *waiting* for something,
# then something probably went wrong
MAX_WAIT_CYCLES = 20_000

# for better error handling, we'll log the last N frames (at say, 15 FPS to preserve memory and disk space)
MAX_DEBUG_VIDEO_LENGTH = 15 * 10 # 15 FPS x 10 seconds of video

DATA_DIR = os.path.join(ROOT_DIR, 'data')
REF_IMG_DIR = os.path.join(DATA_DIR, 'reference_images')

BATTLE_TOWER_STREAK_LENGTH = 7
NUM_POKEMON_IN_SINGLES = 3 # in case I ever want to do doubles, I'm setting this now

POKEMON_MAX_MOVES = 4

def ref_img(image_name) -> np.ndarray:
    return cv2.imread(os.path.join(REF_IMG_DIR, image_name))

SAVE_GAME_Q = (
    ref_img('save_game_q.png'),
    154, # upper-left hand row of the image when it should be on screen
    16, # upper left hand col
)

OVERWRITE_GAME_Q = (ref_img('overwrite_game_q.png'),155, 16,)

READY_FOR_BATTLE_TOWER = (ref_img('ready_for_battle_room.png'), 155, 17)

BATTLE_SCREEN = (ref_img('fight.png'), 271, 111)
CANCEL_BUTTON = (ref_img('cancel.png'), 365, 107) # this is the 'Cancel' text at the bottom of the move select menu; it only appears when you're in the move select menu

POKEMON_SELECT = (ref_img('pokemon_select.png'), 171, 16)

SWAP_POKEMON = (ref_img('swap_pokemon.png'), 363, 16)

# if you have a 7 win streak, you are asked to delete the prev video, or if you lose, you are also asked this
DELETE_VIDEO = (ref_img('delete_video.png'), 155, 16)

# After the beating the tycoon (and asking to delete video) you get a 'Congratulations!' (note the !, it prevents us from being confused w/ the win streak)
# and after the 7 win streak, you get a "Congratulations on your winning streak, challenger!"
# We'll use this to determine whether you actually won a streak after asking to delete the video
TYCOON_WIN = (ref_img('tycoon_win.png'), 154, 16)
SET_WIN = (ref_img('streak_win.png'), 155, 16)

# if we lose, we'll get the delete video, and then some text that says "Thank you for playing!"
LOSS_TEXT = (ref_img('loss_text.png'), 154, 16)

def check_key_pixels(frame: np.ndarray, key_pixels, frame_is_bgr=True):
    """
    This is one way that I can determine states.
    I'll basically go frame by frame, check a lot of differet pixels to see if they match what I expect.
    Also I expect the frame to be BGR (due to the way I load the data) but this can be configured.
    key_pixels is expected to be a list of list of lists that look like:
    [
        [[pixel_row, pixel_col], [pixel_r, pixel_g, pixel_b]],
        ...
    ]
    """

    for (pixel_row, pixel_col), pixel_rgb in key_pixels:
        frame_pixel = frame[pixel_row, pixel_col]
        if frame_is_bgr:
            frame_pixel = np.flip(frame_pixel)

        if (frame_pixel != np.array(pixel_rgb)).any():
            return False

    # if we got this far, then the frame *must* work
    return True

def check_screen_subset(frame: np.ndarray, reference_image: np.ndarray, row: int, col: int) -> bool:
    refence_height = reference_image.shape[0]
    reference_width = reference_image.shape[1]

    if reference_image.shape[-1] >= 4:
        reference_image = reference_image[..., :3] # we don't want any alpha in the comparison

    subset = frame[row:row + refence_height, col:col + reference_width]

    return np.array_equal(subset, reference_image)

def is_dialog_box(frame):
    # Since the text for a dialog box can change a lot, I found it easier to just chose a bunch of key pixels to check
    key_pixels = [
        ((148, 242), (240, 200, 200)), # pinkish pixel on the upper right side of the bar
        ((149, 242), (232, 152, 152)), # slightly darker pixel right below it
        ((150, 242), (216, 96, 96)), # red color for the right side of the dialog box
        ((166, 11), (40, 48, 40)), # dark gray 2nd border on the left side of the dialog box
        # 'corners' of the dialog box
        ((145, 12), (40, 48, 40)),
        ((190, 12), (40, 48, 40)),
        ((145, 235), (40, 48, 40)),
        ((190, 235), (40, 48, 40)),
    ]

    return check_key_pixels(frame, key_pixels)

def is_next_opponent_box(frame):
    # these key pixels check for the choice box that appears before the next opponent
    key_pixels = [
        ((69, 173), (72,96, 120)), # inner side of the dark blue bar on upper left hand corner
        ((138, 250), (72,96, 120)), # inner side of dark blue outline on lower right hand corner
        ((71, 175), (168, 184, 192)), # inner side of light blue outline on upper left hand corner
        ((136, 248), (168, 184, 192)), # inner side of light blue outline on lower right hand corner
    ]

    return is_dialog_box(frame) and check_key_pixels(frame, key_pixels)
    
def is_save_dialog(frame):
    return is_dialog_box(frame) and check_screen_subset(frame, *SAVE_GAME_Q)

def is_save_overwrite_dialog(frame):
    return is_dialog_box(frame) and check_screen_subset(frame, *OVERWRITE_GAME_Q)

def in_pokemon_select(frame):
    return check_screen_subset(frame, *POKEMON_SELECT)

def in_battle(frame):
    return check_screen_subset(frame, *BATTLE_SCREEN)

def in_move_select(frame):
    return check_screen_subset(frame, *CANCEL_BUTTON)

def pokemon_is_fainted(frame):
    # the `get_party_status` is probably not necessary but it's another check to make sure at least one pokemon is fainted
    return check_screen_subset(frame, *SWAP_POKEMON) and get_party_status(frame).any()

def is_ready_for_battle_tower(frame):
    return check_screen_subset(frame, *READY_FOR_BATTLE_TOWER)

def won_set(frame):
    return check_screen_subset(frame, *SET_WIN) or check_screen_subset(frame, *TYCOON_WIN)

def at_save_battle_video(frame):
    return check_screen_subset(frame, *DELETE_VIDEO)

def lost_set(frame):
    return check_screen_subset(frame, *LOSS_TEXT)

def get_battle_number(frame):
    """
    Gets the # of the next opponent, or None if it can't find the opponent number
    """
    if not is_next_opponent_box(frame):
        return None

    for opp_number in range(2, 8):
        opp_img = ref_img(f'opp_{opp_number}.png')
        if check_screen_subset(frame, opp_img, 155, 184): # all of the numbers are the same size and also start in the same row/col
            return opp_number
        
    # if we've gotten to this point, it may be the tower tycoon
    if check_screen_subset(frame, ref_img('opp_tower_tycoon.png'), 155, 16):
        return 7
    
    # if we've gotten *this* far, something went wrong b/c we should have found an opponent number by now, I should probably log when this happens
    return None

def get_party_status(frame):
    """
    This function returns an np array w/ the alive/fainted status of each member in the team.
    True means the pokemon is healthy, false means the Pokemon has fainted
    """
    
    fainted_color = (176, 88, 0)
    slot_1 = (224, 118)
    slot_2 = (225, 247)
    slot_3 = (266, 118)

    # I'm using check_key_pixels instead of a normal equals because it handles the image being bgr (plus any other logic I may implement in the future)
    fainted_status = np.array([
        check_key_pixels(frame, ((slot_1, fainted_color), )),
        check_key_pixels(frame, ((slot_2, fainted_color), )),
        check_key_pixels(frame, ((slot_3, fainted_color), )),
    ])

    return ~fainted_status

def get_selected_pokemon_in_swap_screen(frame):
    """
    This function returns the slot number (from 0-5) of the pokemon that is being hovered in the swap screen.
    Useful b/c sometimes just hitting the "right" button can be a little over-zealous and put us in an unexpected state.
    It's also possible that there is no slot selected, in which case it returns None (indicating that you have to push A first)
    """
    selected_color = (248, 0, 0)

    # there are arrows on each corner of the slot that is selected; these coordinates refer to those arrows in upper-left hand -> lower-right hand
    slot_pixels = [
        [ # this is for slot 0
            [194, 2], [194, 125], [237, 2], [237, 125]
        ],
        [ # this is for slot 1
            [201, 130], [201, 254], [245, 130], [245, 254]
        ],
        [ # slot 2
            [243, 2], [243, 125], [285, 2], [285, 125]
        ],
        [ # slot 3
            [250, 130], [250, 254], [294, 130], [294, 254]
        ],
        [ # slot 4
            [292, 2], [292, 125], [331, 2], [331, 125]
        ],
        [ # slot 5
            [298, 130], [298, 254], [341, 130], [341, 254]
        ]
    ]

    matching_slot_id = None

    for slot_id, slot_corners in enumerate(slot_pixels):
        if check_key_pixels(frame, [[corner, selected_color] for corner in slot_corners]):
            matching_slot_id = slot_id
            break

    return matching_slot_id

def our_pokemon_is_out(frame):
    """
    This function checks whether our pokemon is fully out (i.e. that we're in battle,
    there is a name/HP bar, and that it isn't fainted/we're waiting for the next pokemon
    """
    # There's no good "one" check for the our's bar, so I am checking various key pixels
    # (mostly at the far right b/c the name bar slides to the left when a pokemon faints)
    # There is *one more* complication: unlike the opponent, our pokemon "bounces" up and down and the info bar does too
    #  which means we actually have to check for three different positions

    # this handles the "top" position (i.e. the tallest it gets), each other position is just 1 pixel down
    key_pixels = [
        ([119, 132], (40, 48, 40)), # far left sticking out dark pixel in the HP arrow bar
        ([120, 129], (96, 72, 56)), # far left sticking out slightly brigher pixel in arrow bar
        ([96, 147], (40, 48, 40)), # upper left gray corner of our pkmn box
        ([133, 147], (40, 48, 40)), # lower left gray corner of our pkmn box
    ]

    is_top = check_key_pixels(frame, key_pixels)

    if is_top:
        return 1
    else:
        for pixel in key_pixels:
            pixel[0][0] += 1 # each subsequent thing is one more pixel down

        is_middle = check_key_pixels(frame, key_pixels)

        if is_middle:
            return 2
        else:
            for pixel in key_pixels:
                pixel[0][0] += 1  # each subsequent thing is one more pixel down

            is_bottom = check_key_pixels(frame, key_pixels)

            if is_bottom:
                return 3

    # abusing the notation a bit here, but 0 means false, 1 means top, 2 means middle, 3 means bottom
    return 0

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

def get_opponent_pokemon_info(frame):
    """This gets the part of the frame containing the opponents pokemon's name, gender, and level"""
    info = frame[27:37, 2:98, :]
    return info

def get_cur_pokemon_info(frame, position=1):
    """
    This gets the part of the frame containing our pokemon's name, gender, and level
    Position refers to the top (1) middle (2) and bottom (3) possible positions for the "bouncing" pkmn info
    """
    # the constants (101 and 111) account for the fact that the position starts at 1, so for the top "bounce" ymin is 102 and ymax is 112
    ymin = 101 + position
    ymax = 111 + position

    info = frame[ymin:ymax, 152:256, :]
    return info

class TowerState(Enum):
    WAITING = 0
    LOBBY = 1
    POKEMON_SELECT = 2
    BATTLE_TOWER = 3
    SWAP_POKEMON = 4
    BATTLE = 5
    MOVE_SELECT = 6

    # these aren't states to be in, per say, but they are relevant for keeping track of what happened
    WON_BATTLE = 100
    END_OF_SET = 101
    WON_SET = 102
    LOST_SET = 103

    # used just for search to keep track of the state
    STOPPED_SEARCH = 1000




class BattleTowerAgent:
    # you'll have to override these in subclasses
    strategy: str = None
    team: str = None # NOTE: this should ideally be a Showdown-compatible format

    def __init__(self, render=True, savestate_file=BATTLE_TOWER_SAVESTATE, db_interface: BattleTowerDBInterface = None):
        """
        Creates the (Abstract) Battle Tower Agent.
        The strategy is to just search over all available moves until the end of the battle.
        Args:
            render: Whether to display the battle as it's going on.
            savestate_file: The initial savestate file that the agent loads the game from.
                There is a somewhat intricate setup needed to run the agent, so I don't recommend changing this.
            db_interface: A BattleTowerDB Interface, letting the agent record it's stats to a DB as it is playing
                (by default it is None, so it won't  record anything).
        """
        self.env = PokemonEnv(
            include_bottom_screen=True,
            savestate_files=[savestate_file],
        )
        self.env.reset()

        self.render = render

        self.state = TowerState.LOBBY

        self.current_streak = 0
        self.longest_streak = 0
        self.num_attempts = 0
        self.cur_frame = self.env.step(None)

        self.debug_video = deque(maxlen=MAX_DEBUG_VIDEO_LENGTH)

        self.num_cycles = 0
        self.cur_battle_start_cycle = 0 # this is useful for keeping track of the duration of a battle (mainly for DB logging purposes)

        if db_interface is None: # the default DBInterface is a no-op
            db_interface = BattleTowerDBInterface()

        self.db_interface = db_interface

    def play(self):
        while True:
            self.play_set()

    def play_set(self) -> bool:
        """
        Plays a single 7-game set through the battle tower, returning True if we won the set and False if we lost.
        Requires the player to be directly adjacent and facing the center person; it will return the player in the same spot and direction.
        """
        # for the initial set, this helps seed the RNG and also the opponent's battle tower team
        # w/o it, it always gets the battle tower team from when the savestate was made (not the proper opponent set)
        self._general_button_press(['UP'])

        start_time = time.time()
        set_start_cycle = self.num_cycles

        if self.current_streak == 0:
            self.num_attempts += 1
            logger.info(f'Starting attempt {self.num_attempts}, current winstreak record is {self.longest_streak}.')
            self.db_interface.on_streak_start(team=self.team, strategy=self.strategy)

        # this will get us through the initial dialog (including selecting singles battle, etc)
        self.state = self._wait_for((in_pokemon_select, TowerState.POKEMON_SELECT), button_press='A')

        # we should be in Pokemon select by now
        self._select_pokemon()

        # this isn't strictly needed, but I want to check to make sure we got out of the pokemon select,
        #  passed the save step, and are ready to enter the battle tower
        self.state = self._wait_for((is_ready_for_battle_tower, TowerState.BATTLE_TOWER), button_press='A')

        self._general_button_press(button_press='A') # we need to press A to get passed the 'ready' dialog

        state = self._run_set_loop()

        if state != TowerState.END_OF_SET:
            self._log_error_image('post_battle_loop', state)
            raise ValueError("This *really* shouldn't happen, but somehow the state is", state)

        state = self._wait_for(
            (won_set, TowerState.WON_SET),
            (lost_set, TowerState.LOST_SET),
            button_press='B', # I want to skip dialog and also not accidentally re-start another dialog, so I choose B over A
        )

        won = False
        if state == TowerState.WON_SET:
            self.current_streak += 1
            logger.info(f'Won the set! The current streak is {self.current_streak}')
            self.db_interface.on_battle_end(won=True, duration=self.num_cycles - self.cur_battle_start_cycle)

            if self.current_streak > self.longest_streak:
                logger.info(
                    f'The current streak of {self.current_streak} just passed the previous longest streak of {self.longest_streak}')
                self.longest_streak = self.current_streak

            won = True
        elif state == TowerState.LOST_SET:
            logger.info(f'Lost the last game. The win streak ended on game {self.current_streak + 1}.')
            self.current_streak = 0

            self.db_interface.on_battle_end(won=False, duration=self.num_cycles - self.cur_battle_start_cycle)
            self.db_interface.on_streak_end()

        # whether we won or lost, we should be back in the lobby at this point
        self.state = TowerState.LOBBY

        end_time = time.time() # it's about 2x faster to not render stuff
        duration = end_time - start_time
        cycles = self.num_cycles - set_start_cycle
        logger.debug(f'Finished set in {duration}s, it took {cycles} cycles, which comes out to {cycles / duration} frames/s')

        return won

    def play_battle(self) -> TowerState:
        """
        Plays a single battle in the battle tower.
        Expects player to currently be in the battle tower (though not necessarily at the fight screen)
        After the fight is over, will fast-forward to either the next opponent dialog or the save video dialog (and will return the corresponding state)
        """
        if self.state != TowerState.BATTLE_TOWER and self.state != TowerState.BATTLE:
            self._log_error_image('not_in_battle_tower')
            raise ValueError(f'The agent is attempting to fight a battle but the player is not in the battle tower. Expected state to be {TowerState.BATTLE_TOWER} or {TowerState.BATTLE}, got {self.state}.')

        start_time = time.time()
        self.cur_battle_start_cycle = self.num_cycles
        logger.debug(f'Beginning battle #{self.current_streak + 1}') # +1 b/c we report streak length as 1-indexed

        self.state = self._wait_for(
            (in_battle, TowerState.BATTLE),
            button_press='B', # we may need to skip dialog
            check_first=True, # and it's possible that we're already in the fight
        )

        state = self._run_battle_loop()

        duration = time.time() - start_time
        cycles = self.num_cycles - self.cur_battle_start_cycle
        logger.debug(f'Finished game in {duration}s, it took {cycles} cycles, which comes out to {cycles / duration} frames/s')

        return state

    def _select_move(self) -> int:
        """
        This function is what makes the agent an agent. It is how we control which moves (or switches) that the agent takes.
        It expects that we're in move select, and it returns the TowerState that we're in after successfully selecting a move/advanced the game in one way or another.
        """
        raise NotImplementedError("The BattleTowerAgent class is an abstract class, you need to subclass it and implement _select_move yourself.")

    def _execute_move(self, move: int) -> TowerState:
        """
        This function takes a move number (0-3, most likely from the output of `_select_move`)
          and executes that move in the Pokemon client.
        It expects that we're in move select, and it returns the TowerState that we're in after successfully
          selecting a move/advanced the game in one way or another.

        There's one slight snag, we may or may not be able to select the move (e.g. due to torment, choice specs)
          but you are *still* in move select, unlike certain other conditions
        There's no (good) way to know until after we click it, so we've just got to keep trying until we get it
        It's a tad inefficient, but it *is* compatible w/ searching and choice moves b/c:
        1. if we are searching over a set of moves that are different, we break
        2. if we are searching the same move consecutively, then even if it isn't the first move, we'll eventually goto the choice selected move.
        """
        state = self.state

        advanced_game = False
        for i in range(POKEMON_MAX_MOVES):
            # i is 0 at first, if the first move is successful, we'll never have to go to the next one
            chosen_move = move + i
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


    def _act(self, action: str | None = None) -> np.ndarray:
        """This function is basically a wrapper for the env.step but it also handles render logic"""
        frame = self.env.step(action)
        self.num_cycles += 1

        if self.num_cycles % 4 == 0:
            self.debug_video.append(frame)

        if self.render:
            # for display purposes, I want the screen to be 2x bigger
            display_frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_NEAREST)

            cv2.imshow('Pokemon Platinum', display_frame)
            cv2.waitKey(1)

        return frame

    def _wait_for(self, *checks: tuple[callable, TowerState], button_press: str | None = None, check_first=False) -> TowerState:
        """
        Waits for the environment to reach a predefined state from a (list of) predefined states and their corresponding 'true' conditions.
        While waiting for that state, it will continuously press the button (if provided).
        If multiple checks are true on the same frame, returns the state corresponding to the earlier check (the order they appear makes a difference)
        If `check_first` is True, the checks will be run first and it may return a state before pressing any buttons

        NOTE: you may encounter problems waiting for two states that are back to back (e.g. BATTLE and MOVE_SELECT)
            because it may hit A, it takes longer than a single frame for the screen to advance, and the check sees it as the first state,
            but by the time A is finished processing, we've moved onto the 2nd state.
            This can be avoided by using `check_first` and implementing some of the logic yourself
        """

        reached_state = None
        starting_cycle = self.num_cycles

        def run_checks(frame):
            for check, state in checks:
                if check(frame):
                    logger.debug(f'Reached {state}')
                    return state
            return None

        if check_first:
            reached_state = run_checks(self.cur_frame)

        while reached_state is None:
            logger.log(msg=f'Pressing {button_press}', level=logging.BUTTON_PRESS)

            # this makes sure we hold down the button long enough to take an action (making sure we don't just 'tap' A for 1 frame)
            for _ in range(BUTTON_PRESS_DURATION):
                self.cur_frame = self._act(button_press)
                reached_state = run_checks(self.cur_frame)
                if reached_state:
                    break

            # after any # press, even if we do reach the state, we'll always wait the full duration so that we can let the game process things
            for _ in range(AFTER_PRESS_WAIT):
                self.cur_frame = self._act()

                if not reached_state:
                    reached_state = run_checks(self.cur_frame)

            if self.num_cycles - starting_cycle > MAX_WAIT_CYCLES:
                error_str = f'Timeout in `_wait_for`. It has been {self.num_cycles - starting_cycle} cycles with no change in state. Checking these conditions:'

                for check, state in checks:
                    check_name = check.__name__ if hasattr(check, '__name__') else 'lambda'
                    error_str += f'\n- {check_name}: {state}'

                self._log_error_image(message='wait_for_timeout')
                raise ValueError(error_str)

        return reached_state

    def _wait_for_battle_states(self):
        """
        Whenever we're in a battle, these are the possible states we could reach after clicking any move
        This is a special case of _wait_for that we'll tend to use in the battle loop
        """
        return self._wait_for(
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


    def _select_pokemon(self):
        """Handles the Pokemon select screen by selecting the first 3 Pokemon in the party."""
        if not in_pokemon_select(self.cur_frame) or not self.state == TowerState.POKEMON_SELECT:
            self._log_error_image(message='pokemon_select', state=self.state)
            raise ValueError(f"Something is out of order here, `_select_pokemon` was called but the Pokemon select screen isn't up right now or the state wasn't properly set (it is currently {self.state}).")

        # TODO: right now, I'll only select the first 3 Pokemon, but in the future, I should do more advanced logic to support variable teams

        # It takes 1 A presses to select and confirm a Pokemon, and we can always just go right to get to the next Pokemon
        for slot_num in range(NUM_POKEMON_IN_SINGLES):
            if slot_num != 0:
                self._general_button_press('RIGHT') # we can't do this at the end b/c that will move the selection away from 'confirm'

            self._general_button_press('A')
            self._general_button_press('A')

        self._general_button_press('A') # to confirm the team

    def _run_set_loop(self) -> TowerState:
        """
        This function goes through the 'standard' battle loop of the battle tower within a single set,
        only breaking when a set ends with either a win or a loss (which is returned as a state).
        It expects the player to be past any pre-tower dialog and actually in (or at least being escorted into) the battle tower.
        """

        # now that we're in the battle tower proper, we'll handle the pokemon battles
        #  and between/post battle situations. There's pretty much 2 cases:
        # 1. if we win battle 1-6, the dialog for the next opponent will show up and we'll use that dialog to know we've won
        # 2. if lose a battle or win the set, we'll be prompted to save the last video. Then, we can determine whether we won or not by:
        #     * if we get a congratulations, we won the set
        #     * if we get a 'Thank you for playing' then we lost the battle

        logger.debug('Starting set loop')

        state = self.state

        while not state == TowerState.END_OF_SET:
            state = self.play_battle()

            if state == TowerState.WON_BATTLE:
                self.current_streak += 1
                logger.info(f'Won a battle. The current streak is {self.current_streak}')

                # this is a check for my logic to make sure that I'm capturing wins properly, it *should* always be true, but my code may be buggy so...
                next_opp_number = get_battle_number(self.cur_frame)
                if self.current_streak % 7 != next_opp_number - 1:
                    self._log_error_image(message=f'streak_mismatch_{self.current_streak}_{next_opp_number}')
                    raise ValueError(
                        'Mismatch between the recorded current streak and the next opponent.'
                        f' The current streak is {self.current_streak} and we expected to fight opponent {self.current_streak % 7 + 1}'
                        f" but instead we're fighting opponent {next_opp_number}"
                    )

                if self.current_streak > self.longest_streak:
                    logger.info(f'The current streak of {self.current_streak} just passed the previous longest streak of {self.longest_streak}')
                    self.longest_streak = self.current_streak

                # if we just wait for the next state, it'll press A, but it'll also check the frame and show the opponent # on the same cycle we hit the button
                #  so we need to manually press A here (it avoids a bug of counting the win twice)
                self._general_button_press('A')

                self.db_interface.on_battle_end(won=True, duration=self.num_cycles - self.cur_battle_start_cycle)

        # we've lost the game or won the whole set, either way, the battle loop is finished... and we're also back in the Battle Tower proper
        self.state = TowerState.BATTLE_TOWER

        return state

    def _run_battle_loop(self) -> TowerState:
        """
        This is where the battle logic gets handled.
        Expects the game to be in a battle when the function is called.
        Supports a single move or a list of moves.
        - If there is only a single action, it always chooses that
        - If there are a list of actions, performs them 1-by-1
        If a chosen move is blocked for some reason, it'll choose the next move (based on position) once, then go back to the normal order (however the move will be consumed from the move selection list)
        Loops until the battle is over (either we're in the winning state or the end of set state)
        """
        if not in_battle(self.cur_frame) and self.state != TowerState.BATTLE:
            self._log_error_image(message='not_in_battle', state=self.state)
            raise ValueError(f'Expected to be in a battle with the Fight screen up when calling `_run_battle_loop`, but the current state is {self.state}')

        logger.debug(f'Starting battle loop')

        state = self.state

        while state != TowerState.WON_BATTLE and state != TowerState.END_OF_SET:

            # NOTE: this works even on the first turn when you have to highlight the fight button b/c the while loop goes back to here
            if state == TowerState.BATTLE:
                self._general_button_press('A')

            # NOTE: w/ struggle (and maybe some other effect), you don't get to go into move select, you hit Fight and it happens automatically
            #  so you'll either go to the next turn (w/ the Fight screen), you'll have to swap a pokemon, or the battle will end
            #  which is why I still have to `_wait_for` after entering the move screen

            state = self._wait_for_battle_states()

            if state == TowerState.MOVE_SELECT:
                self.state = state
                move = self._select_move()
                state = self._execute_move(move)

            # I'm not using elif b/c this handles the cases where we don't go into move select (e.g. struggle) and faint,
            #  *or* we went into move select, chose a move, then fainted afterwards
            if state == TowerState.SWAP_POKEMON:
                self.state = state
                self._swap_to_next_pokemon()
                self.state = TowerState.BATTLE

        return state


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

    def _goto_move(self, move_idx: int):
        """
        When in the fight screen, moves the cursor to the chose move (zero-indexed at the upper-left hand corner)
        0 for upper-left, 1 for upper-right, 2 for lower-left, 3 for lower-right)
        NOTE: this function does not select the move, it just goes to it
        """

        if not in_move_select(self.cur_frame) or not self.state == TowerState.MOVE_SELECT:
            self._log_error_image(message='move_select', state=self.state)
            raise ValueError(f'Called `goto_move` but not in move select, and the current state is {self.state}')

        # b/c the previously selected move is saved, we don't know which move we're on, so I'll be safe and always move back up to the first move
        self._general_button_press(['UP', 'LEFT'])

        # now that we're at the upper-right hand corner, we can easily go to any other slot

        if move_idx == 1 or move_idx == 3:
            self._general_button_press('RIGHT')
        if move_idx == 2 or move_idx == 3:
            self._general_button_press('DOWN')


    def _general_button_press(self, button_press: str | list[str] | None = None):
        """
        This is a button press that we can use when we don't need any special waiting logic
        It handles the durations 'n stuff
        Supports:
        - single button press
        - no button presses (use `None`)
         - a list of button presses (including a list of no button, but why would you do that?)
        """

        if not isinstance(button_press, list):
            button_press = [button_press]

        for button in button_press:
            logger.log(msg=f'Pressing {button}', level=logging.BUTTON_PRESS)

            for _ in range(BUTTON_PRESS_DURATION):
                self.cur_frame = self._act(button)
            for _ in range(AFTER_PRESS_WAIT):
                self.cur_frame = self._act()

    def _log_error_image(self, message='', state=None):
        """Whenever there is an error, we log a video of the last part of gameplay"""
        log_dir = os.path.join(ROOT_DIR, 'log', 'debug')
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        log_fname = f'{now}-ERROR'

        if message:
            log_fname += f'_{message}'

        if state:
            log_fname += '_' + str(state).replace('.', '-') # TowerState strs look like TowerState.Battle but we want to get rid of the .

        video_dir = os.path.join(log_dir, log_fname)
        os.makedirs(video_dir, exist_ok=True) # this probably shouldn't happen but b/c of the time foolery stuff I do on linux, we should be careful
        for i, frame in enumerate(self.debug_video):
            cv2.imwrite(os.path.join(video_dir, f'frame_{i}.jpg'), frame)


GARCHOM_SUICUNE_SCIZOR_TEAM = """Garchomp @ Focus Sash  
Ability: Sand Veil  
EVs: 4 HP / 252 Atk / 252 Spe  
Jolly Nature  
- Outrage  

Suicune @ Choice Specs  
Ability: Pressure  
EVs: 252 HP / 4 Def / 252 SpA  
Modest Nature  
- Surf  

Scizor @ Choice Band  
Ability: Technician  
EVs: 252 HP / 252 Atk / 4 SpD  
Adamant Nature  
- Bullet Punch  
"""

class BattleTowerAAgent(BattleTowerAgent):
    strategy = 'A'
    team = GARCHOM_SUICUNE_SCIZOR_TEAM

    def _select_move(self) -> int:
        # the 'A' agent always tries to select the first move
        return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    agent = BattleTowerAAgent(
        render=True,
        #db_interface=BattleTowerServerDBInterface()
    )

    agent.play()

    # # NOTE: this is my debug stuff
    # from pokemon_env import *
    # import keyboard
    # import win32api
    # import win32gui
    # import time
    #
    # emu = DeSmuME()
    # emu.open(ROM_FILE)
    # emu.savestate.load_file(r'..\ROM\Name Save 2.dst')
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
    #     our_pokemon_is_out,
    #     opp_pokemon_is_out,
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
    #         image_path = os.path.join(DATA_DIR, 'Decision Making', input('Enter image path:') + '.PNG')
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
    #         print('Currently selecting slot # ', get_selected_pokemon_in_swap_screen(screen))
    #
    #     if is_next_opponent_box(screen):
    #         print('Next opp:', get_battle_number(screen))
    #
    #     emu.cycle()
    #     window.draw()