import logging
from multiprocessing import Queue

from battle_tower_agent import BattleTowerAAgent, BATTLE_TOWER_SAVESTATE, BUTTON_PRESS_DURATION, AFTER_PRESS_WAIT
from battle_tower_database.interface import BattleTowerDBInterface

logger = logging.getLogger('TwitchAgent')

class BattleTowerTwitchAgent(BattleTowerAAgent): # TODO: make this work w/ the search agents
    def __init__(self, frame_queue: Queue, render=True, savestate_file=BATTLE_TOWER_SAVESTATE, db_interface: BattleTowerDBInterface = None):
        super().__init__(render, savestate_file, db_interface)

        self.frame_queue = frame_queue

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
                self.frame_queue.put(self.cur_frame)
            for _ in range(AFTER_PRESS_WAIT):
                self.cur_frame = self._act()
                self.frame_queue.put(self.cur_frame)