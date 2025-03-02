import logging
import time
from queue import Queue

import numpy as np

from battle_tower_agent.agent import BattleTowerAAgent

logger = logging.getLogger('DisplayAgent')

TARGET_FPS = 60
TARGET_FRAME_TIME = 1 / TARGET_FPS  # ~0.01667 seconds

def create_battle_tower_display_agent(frame_queue: Queue, result_queue: Queue, agent_cls = BattleTowerAAgent, *agent_args, **agent_kwargs, ):
    """
    Creates a Display Agent (i.e. an agent that works with the current commentator display) that subclasses a given BattleTower agent class.

    Args:
        frame_queue: Queue for sending frames to the UI.
        result_queue: Queue for sending results to the UI.
        agent_cls: The base BattleTowerAgent class to subclass (default: BattleTowerAAgent).
        *agent_args: Positional arguments to pass to the base class constructor.
        **agent_kwargs: Keyword arguments to pass to the base class constructor.
    """

    class BattleTowerDisplayAgent(agent_cls):
        def __init__(
                self,
                frame_queue: Queue,
                result_queue: Queue,
                *args,
                **kwargs
        ):
            super().__init__(*args, **kwargs)

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

    agent = BattleTowerDisplayAgent(
        *agent_args,
        frame_queue=frame_queue,
        result_queue=result_queue,
        **agent_kwargs
    )

    return agent

if __name__ == '__main__':

    frame_queue = Queue()
    result_queue = Queue()
    agent = create_battle_tower_display_agent(frame_queue, result_queue)
    agent.play()