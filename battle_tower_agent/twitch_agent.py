import logging
import time
from queue import Queue

import numpy as np

from battle_tower_agent.agent import BattleTowerAAgent, BATTLE_TOWER_SAVESTATE, BUTTON_PRESS_DURATION, AFTER_PRESS_WAIT
from battle_tower_agent.battle_tower_database.interface import BattleTowerDBInterface

logger = logging.getLogger('TwitchAgent')

TARGET_FPS = 60
TARGET_FRAME_TIME = 1 / TARGET_FPS  # ~0.01667 seconds

class BattleTowerTwitchAgent(BattleTowerAAgent): # TODO: make this work w/ the search agents (IDEA: create a 'get-agent' function that takes a class? and then we subclass that way?)
    def __init__(self, frame_queue: Queue, result_queue: Queue, render=False, savestate_file=BATTLE_TOWER_SAVESTATE, db_interface: BattleTowerDBInterface = None):
        super().__init__(render=render, savestate_file=savestate_file, db_interface=db_interface)
        # render is a no-op now due to how we subclassed _act

        # this lets us pass the frames back to the UI process
        self.frame_queue = frame_queue

        # this lets us pass stuff like the current streak length back to the UI process
        self.result_queue = result_queue

        # we actually want the game to play audio now for the twitch stream
        self.env.emu.volume_set(100)

    def _act(self, action: str | None = None) -> np.ndarray:
        """
        This function wraps env.step() and handles render logic.
        It enforces a relatively consistent frame rate by sleeping the remainder
        of the frame period after processing.
        """
        start_time = time.perf_counter()

        # Get the frame by taking an action
        frame = self.env.step(action)
        self.num_cycles += 1

        # I only want to display this at 30 FPS, so we can just skip every other frame
        if self.num_cycles % 2 == 0:
            self.frame_queue.put(frame)

        # okay the 'correct' place to put this is actually every time we update the events, current streak, longest streak, etc.
        # but that's spread across a fair bit of code and it's okay to only update the results periodically (
        # in this case, ~ 1/second so as to not add too much extra compute
        if self.num_cycles % 60 == 0:
            current_results = {
                'num_attempts': self.num_attempts,
                'current_streak': self.current_streak,
                'longest_streak': self.longest_streak,
            }
            try:
                self.result_queue.put(current_results, block=False) # no problem if we can't put it in the queue this time
            except:
                pass

        # Calculate how long this cycle took
        elapsed = time.perf_counter() - start_time
        remaining = TARGET_FRAME_TIME - elapsed

        # Sleep if there's time remaining within the target frame time.
        if remaining > 0:
            time.sleep(remaining)
        else:
            # Optionally log a warning if processing took longer than the target frame period.
            logger.warning("Frame processing exceeded target time by {:.4f} seconds".format(-remaining))

        return frame

if __name__ == '__main__':
    # import numpy as np
    # import cv2
    #
    # size = 720 * 16 // 9, 720
    # duration = 15
    # fps = 25
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    # for _ in range(fps * duration):
    #     data = np.random.randint(0, 256, size, dtype='uint8')
    #     out.write(data)
    # out.release()
    frame_queue = Queue()
    agent = BattleTowerTwitchAgent(render=True, frame_queue=frame_queue)
    agent.play()