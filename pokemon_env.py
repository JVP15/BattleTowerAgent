import numpy as np
import cv2
import os

from desmume.emulator import DeSmuME, SCREEN_PIXEL_SIZE, SCREEN_PIXEL_SIZE_BOTH, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_HEIGHT_BOTH
from desmume.controls import Keys, keymask

ROM_FILE = os.path.join('ROM', 'Pokemon - Platinum.nds')
SAVESTATE_FILES = [
    os.path.join('ROM', 'Pokemon - Platinum.ds1')
]

CYCLES_PER_STEP = 60 # the game goes at 60 FPS, so this means each step will last at minimum 1 second to handle any effects
CYCLES_PER_ACTION = 15 # this means each individual action (press a button, wait some time for effects) will last 15 frames
CYCLES_PER_BUTTON_PRESS = 9 # this means we hold down a button press for 9 frames

class PokemonEnv():
    """
    Creates an environment to run the Pokemon Platinum. 
    Note: while it does have many similarities to the gym spec, it is not compatible (specifically due to changes w/ how rendering works)
    """

    button_to_key = {
        'A': Keys.KEY_A,
        'B': Keys.KEY_B,
        'X': Keys.KEY_X,
        'Y': Keys.KEY_Y,
        'UP': Keys.KEY_UP,
        'DOWN': Keys.KEY_DOWN,
        'LEFT': Keys.KEY_LEFT,
        'RIGHT': Keys.KEY_RIGHT,
        'START': Keys.KEY_START,
        'SELECT': Keys.KEY_SELECT,
        'L': Keys.KEY_L,
        'R': Keys.KEY_R
    }


    def __init__(
            self, 
            include_bottom_screen=False,
            rom_file=ROM_FILE, 
            savestate_files=SAVESTATE_FILES, 
        ):
        
        self.emu = DeSmuME()
        self.emu.volume_set(0) # even headless, desmume makes noise

        self.emu.open(rom_file)

        self.savestate_files = savestate_files

        # TODO: this gives us a starting point for the LLM, gotta figure out some other save states and points to work with
        self.load_savestate(self.savestate_files[0])

        self.include_botton_screen = include_bottom_screen

    def load_savestate(self, savestate_file):
        self.emu.savestate.load_file(savestate_file)

    def step(self, action: str | None = None) -> np.ndarray:
        # it's just easier to clear all of the keys first and then set the ones we want instead of trying to figure out which ones are already set
        self.emu.input.keypad_rm_key(Keys.NO_KEY_SET)

        if action:
            action = action.strip().upper()

            if action not in self.button_to_key:
                raise ValueError(f"Invalid action: {action}, expected one of: {self.button_to_key.keys()}")

            self.emu.input.keypad_add_key(keymask(self.button_to_key[action]))

        self.emu.cycle()

        return self.get_state()

    def get_state(self):
        screen = self._get_screen()
        
        if not self.include_botton_screen:
            screen = screen[:SCREEN_HEIGHT]
        
        return screen
        
    def reset(self):
        self.load_savestate(self.savestate_files[0])
        return self.get_state()

    def _get_screen(self):
        # see https://py-desmume.readthedocs.io/en/latest/quick_start.html#custom-drawing
        screen_buffer = self.emu.display_buffer_as_rgbx()
        screen_pixels = np.frombuffer(screen_buffer, dtype=np.uint8)

        screen = screen_pixels[:SCREEN_PIXEL_SIZE_BOTH * 4] 
        screen = screen.reshape((SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 4))[..., :3] # drop the alpha channel

        return screen

class PokemonClient:
    """
    This is a wrapper around the Pokemon env that handles some things, in particular holding down a button for long enough for an action to complete
    (since you need to hold the D-pad or else it will only rotate the character instead of moving the character)
    """

    def __init__(
        self, 
        render_screen=False, 
        include_bottom_screen=False,
        rom_file=ROM_FILE, 
        savestate_files=SAVESTATE_FILES, 
        cycles_per_step=CYCLES_PER_STEP,
        cycles_per_action=CYCLES_PER_ACTION,
        cycles_per_button_press=CYCLES_PER_BUTTON_PRESS
    ):
        """
        The 'step' refers to the minimum time it takes to wait for the result of an action or list of actions.
        The 'action' refers to the time it takes to press a button and then wait for the results of that button press (this is useful for quickly skipping dialog)
        The 'button_press' refers to how many frames to hold down a button
        """

        self.env = PokemonEnv(
            include_bottom_screen=include_bottom_screen,
            rom_file=rom_file,
            savestate_files=savestate_files,
        )
        
        if cycles_per_step <= 0:
            raise ValueError(f"cycles_per_step must be > 0, got {cycles_per_step}")
        if cycles_per_action <= 0:
            raise ValueError(f"cycles_per_action must be > 0, got {cycles_per_action}")
        if cycles_per_button_press <= 0:
            raise ValueError(f'cycles_per_button_press must be > 0, got {cycles_per_button_press}')

        self.render_screen = render_screen
        self.cycles_per_step = cycles_per_step
        self.cycles_per_action = cycles_per_action
        self.cycles_per_button_press = cycles_per_button_press

        self.tmp_cycles_per_step = cycles_per_step
        self.tmp_cycles_per_action = cycles_per_action
        self.tmp_cycles_per_button_press = cycles_per_button_press

    def step(self, actions: str | list[str] | None):
        if actions is None:
            actions = []
        elif isinstance(actions, str):
            actions = [actions]

        for action in actions:
            action = action.strip().upper()

            for _ in range(self.cycles_per_button_press):
                self._cycle(action)

            for _ in range(self.cycles_per_action - self.cycles_per_button_press):
                self._cycle()

        for _ in range(self.cycles_per_step - self.cycles_per_action * len(actions)):
            self._cycle()

        return self.env.get_state()

    def reset(self):
        self.env.reset()

        return self.step(None)
    
    def _cycle(self, action: str | None = None):
        self.env.step(action)

        if self.render_screen:
            self._render()
        
    def _render(self):
        frame = self.env.get_state()

        # for display purposes, I want the screen to be 2x bigger
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_NEAREST)

        cv2.imshow('Pokemon Platinum', frame)
        cv2.waitKey(3)




if __name__ == '__main__':
    # env = PokemonClient(render_screen=True, include_bottom_screen=False, savestate_files=['ROM\Pokemon - Platinum Battle Tower.dst'])
    # env.reset()
    #
    # while True:
    #     # env.step('RIGHT')
    #     action = input('Enter Action:')
    #
    #     if action.upper().strip() == 'Q':
    #         break
    #
    #     env.step(action)

    # from desmume.emulator import DeSmuME
    #
    # emu = DeSmuME()
    # emu.open(ROM_FILE)
    # emu.savestate.load_file('ROM\Pokemon - Platinum Battle Tower.dst')

    # Create the window for the emulator
    # window = emu.create_sdl_window()
    # import time
    # # Run the emulation as fast as possible until quit
    # while not window.has_quit():
    #     emu.input.keypad_rm_key(Keys.NO_KEY_SET)
    #
    #     window.process_input()  # Controls are the default DeSmuME controls, see below.
    #     emu.cycle()
    #     window.draw()
    #
    #     time.sleep(.01)
    import keyboard
    import win32api
    import win32gui

    emu = DeSmuME()
    emu.open(ROM_FILE)
    emu.savestate.load_file('ROM\Pokemon - Platinum Battle Tower.dst')
    emu.volume_set(0)

    # Create the window for the emulator
    window = emu.create_sdl_window()

    # Get handle for desmume sdl window
    window_handle = win32gui.FindWindow(None, "Desmume SDL")

    CONTROLS = {
        "enter": Keys.KEY_START,
        "right shift": Keys.KEY_SELECT,
        "q": Keys.KEY_L,
        "w": Keys.KEY_R,
        "a": Keys.KEY_Y,
        "s": Keys.KEY_X,
        "x": Keys.KEY_A,
        "z": Keys.KEY_B,
        "up": Keys.KEY_UP,
        "down": Keys.KEY_DOWN,
        "right": Keys.KEY_RIGHT,
        "left": Keys.KEY_LEFT,
    }

    while not window.has_quit():
        # Check if any buttons are pressed and process them
        for key, emulated_button in CONTROLS.items():
            if keyboard.is_pressed(key):
                emu.input.keypad_add_key(keymask(emulated_button))
            else:
                emu.input.keypad_rm_key(keymask(emulated_button))

        if keyboard.is_pressed('t'):
            image_path = os.path.join('images', 'Decision Making', input('Enter image path:') + '.PNG')
            screen_buffer = emu.display_buffer_as_rgbx()
            screen_pixels = np.frombuffer(screen_buffer, dtype=np.uint8)
            screen = screen_pixels[:SCREEN_PIXEL_SIZE_BOTH * 4]
            screen = screen.reshape((SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 4))[..., :3]  # drop the alpha channel

            cv2.imwrite(image_path, screen)

        # Check if touch screen is pressed and process it
        if win32api.GetKeyState(0x01) < 0:
            # Get coordinates of click relative to desmume window
            x, y = win32gui.ScreenToClient(window_handle, win32gui.GetCursorPos())
            # Adjust y coord to account for clicks on top (non-touch) screen
            y -= SCREEN_HEIGHT

            if x in range(0, SCREEN_WIDTH) and y in range(0, SCREEN_HEIGHT):
                emu.input.touch_set_pos(x, y)
            else:
                emu.input.touch_release()
        else:
            emu.input.touch_release()

        emu.cycle()
        window.draw()

