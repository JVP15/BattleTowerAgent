"""

This is a nice display for the Battle Tower Agent as it does its runs,
 with an added bonus that the runs are commentated by Google's Gemini.
Gemini acts as two commentators, Castor and Pollux, and they'll go back and forth about... whatever they want really.

The general layout looks like:

##############################################
#  Result Bar (i.g. current streak length)   #
##############################################
#                           #   Chat Feed    #
#                           #                #
#                           #  Msg1.         #
#         Top DS            #          Msg2. #
#         Screen            #  Msg3.         #
#                           #                #
#                           ##################
#                           #                #
#                           #   Bottom DS    #
#                           #     Screen     #
#                           #                #
##############################################

The tricky part was the chat feed because it has a lot of nice features like:
- text streaming (where each message will be printed 1 character per frame)
- Left/right justification (based on the commentator)
- Text warping (so that you can see the full message)
- Conversation history (older messages will rise higher on the chat feed)
- Messages are surrounded by a bubble (like a rounded rectangle, which needed custom code).
and more!

This code was heavily written by OpenAI o3-mini (mainly because I don't enjoy doing UI stuff). 
"""

import datetime

import cv2
import numpy as np
import threading
import queue
import time
import random
import os
from playsound import playsound

from battle_tower_agent.agent import DATA_DIR
from battle_tower_agent.display.gemini_commentator import GeminiCommentator
from battle_tower_agent.twitch_agent import BattleTowerTwitchAgent

from desmume.emulator import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_HEIGHT_BOTH

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Display Settings

# Window Settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Chat bubble settings:
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = .5
THICKNESS = 1
LINE_SPACING = 5
MARGIN = 10  # margin within chat region

BUBBLE_PADDING = 8     # padding inside each bubble
BUBBLE_SPACING = 10    # vertical spacing between bubbles
BUBBLE_RADIUS = 10

# Result bar settings:
RESULT_FONT = cv2.FONT_HERSHEY_DUPLEX
RESULT_FONT_SCALE = .8
RESULT_THICKNESS = 1
RESULTS_BAR_HEIGHT = 40  # height for the status/results bar at the top


# Color Settings
CASTOR_BUBBLE_COLOR = (70, 70, 70)
POLLUX_BUBBLE_COLOR = (90, 90, 90)
CHAT_BACKGROUND_COLOR = (50, 50, 50)
CHAT_BUBBLE_BORDER_COLOR = (200, 200, 200)

# Video Settings
VIDEO_FPS = 30
FIRST_VIDEO_LENGTH = VIDEO_FPS * 5 # idk, 5 seconds seems like a good time to wait

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Audio Settings
CASTOR_AUDIO = os.path.join(DATA_DIR, 'audio', 'castor.wav')
POLLUX_AUDIO = os.path.join(DATA_DIR, 'audio', 'pollux.wav')

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Simulate asynchronous arrival of message sets.
def simulate_incoming_messages(msg_queue):
    sample_message_sets = [
        [  # Set 1
            {"role": "Castor", "content": "Hello, Pollux! How are you doing today?"},
            {"role": "Pollux", "content": "I'm fine, thanks Castor. How about yourself?"},
        ],
        [  # Set 2
            {"role": "Castor",
             "content": "I'm doing well. I've been pondering the complexities of our universe and the little details that make it so fascinating."},
            {"role": "Pollux",
             "content": "The universe is indeed an enigma—a tapestry of mysteries that leads us to question everything."},
            {"role": "Castor",
             "content": "Sometimes a few words can inspire a million thoughts, and every exchange sparks a new idea."},
        ],
        [  # Set 3: Longer texts.
            {"role": "Pollux", "content": ("I recently read an article about quantum computing. "
                                           "The possibilities are infinite, and I wonder how soon we might see a "
                                           "radical shift in technology due to these emerging concepts. It is truly exciting!")},
            {"role": "Castor", "content": ("Quantum computing holds great promise, but it also challenges our "
                                           "current understanding of digital logic and data processing. "
                                           "It's not only about speed—it’s a new paradigm that forces us to rethink everything.")},
        ],
        [  # Set 4
            {"role": "Castor",
             "content": ("By the way, did you see the latest project update? It includes bug fixes, performance "
                         "improvements and several new features that I think you'll find very exciting. "
                         "The interface has been refined, and we've made behind-the-scenes optimizations as well.")},
            {"role": "Pollux", "content": (
                "Yes, I caught a glimpse of it. The improvements in real-time processing are particularly impressive, "
                "and I'm looking forward to testing the new functionality when I get a chance.")},
        ],
        [  # Set 5
            {"role": "Pollux",
             "content": ("Have you ever considered how artificial intelligence might transform our daily lives? "
                         "From automating mundane tasks to enabling groundbreaking scientific discoveries, "
                         "it's a field full of surprises and potential.")},
            {"role": "Castor",
             "content": ("Absolutely. AI isn’t just about efficiency—it’s about challenging our assumptions "
                         "and redefining creativity. Its influence stretches far beyond simple automation.")},
            {"role": "Pollux",
             "content": ("Sometimes, in our pursuit of progress, we might overlook the beauty of spontaneity, "
                         "the unexpected moments that ultimately shape our experiences.")},
        ]
    ]

    for msg_set in sample_message_sets:
        time.sleep(random.uniform(3, 6))
        msg_queue.put(msg_set)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Wrap text into multiple lines based on a maximum pixel width.
def wrap_text(text, max_width, font, font_scale, thickness):
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = word if current_line == "" else current_line + " " + word
        (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if line_width > max_width and current_line != "":
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    return lines


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# For a given message dictionary return a block (list of text lines),
# with the role on its own first line.
# When streaming only the first curr_progress characters of the content are shown.
def get_message_block(msg, curr_progress=None, available_width=200):  # available_width to be passed in
    block = []
    block.append(msg["role"] + ":")
    if curr_progress is None:
        text_to_wrap = msg["content"]
    else:
        text_to_wrap = msg["content"][:curr_progress]
    wrapped_lines = wrap_text(text_to_wrap, available_width, FONT, FONT_SCALE, THICKNESS)
    block.extend(wrapped_lines)
    return block


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Helper to draw a filled rounded rectangle (using cv2 primitives).
def draw_rounded_rect(img, top_left, bottom_right, color, radius):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness=-1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness=-1)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness=-1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness=-1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness=-1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness=-1)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Draw a single chat bubble.
def draw_chat_bubble(img, bubble):
    x = bubble["x"]
    y = bubble["y"]
    w = bubble["w"]
    h = bubble["h"]
    style = bubble["style"]
    bg_color = bubble["bg_color"]
    border_color = bubble["border_color"]

    if style == "rect":
        cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, thickness=-1)
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, thickness=1)
    elif style == "rounded":
        draw_rounded_rect(img, (x, y), (x + w, y + h), border_color, BUBBLE_RADIUS)
        draw_rounded_rect(img, (x + 1, y + 1), (x + w - 1, y + h - 1), bg_color, BUBBLE_RADIUS)

    current_y = y + BUBBLE_PADDING
    for line in bubble["lines"]:
        (text_size, baseline) = cv2.getTextSize(line, FONT, FONT_SCALE, THICKNESS)
        cv2.putText(img, line, (x + BUBBLE_PADDING, current_y + text_size[1]),
                    FONT, FONT_SCALE, (255, 255, 255), THICKNESS)
        current_y += text_size[1] + LINE_SPACING


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Draw the entire chat feed.
# This is the bread and butter of this program because it handles text streaming,
#  bubble placement, conversation history, etc
def draw_chat_feed(chat_img, finished_groups, current_conv, current_stream):
    """
    Draws the chat feed.
    Args:
        chat_img: the part of the current frame designated for the chat messages.
        finished_groups: a list of conversations that have already been fully processed
        current_conv: the current conversation (i.e. [["role": "Castor", "content": "Hi"}, ...]
        current_stream: the current message that we're processing with, optionally, an additional key "progress"
            that represents how many characters from the message that we've printed so far.

    """
    chat_img[:] = CHAT_BACKGROUND_COLOR # fill with dark gray
    # Determine available width dynamically:
    chat_width = chat_img.shape[1]
    bubble_max_width = int(chat_width * 0.8)

    # Helper: process one conversation group and return a list of bubble dictionaries.
    def process_conv_group(conv_group):
        group_bubbles = []
        for msg in conv_group:
            if msg["role"] == "Castor":
                alignment = "left"
                style = "rect"
                bg_color = CASTOR_BUBBLE_COLOR
            else:  # Pollux
                alignment = "right"
                style = "rounded"
                bg_color = POLLUX_BUBBLE_COLOR
            lines = get_message_block(msg, curr_progress=None, available_width=bubble_max_width - 2 * BUBBLE_PADDING)
            group_bubbles.append({"lines": lines, "alignment": alignment, "style": style,
                                  "bg_color": bg_color, "border_color": CHAT_BUBBLE_BORDER_COLOR})
        return group_bubbles

    bubbles = []
    # Process finished conversation groups.
    for conv in finished_groups:
        conv_bubbles = process_conv_group(conv)
        bubbles.extend(conv_bubbles)
        bubbles.append({"gap": True, "height": BUBBLE_SPACING})

    # Process the current conversation group.
    current_group = []
    if current_conv:
        current_group = process_conv_group(current_conv)

    # currnt_stream is the message that we are streaming
    if current_stream is not None:
        if current_stream["role"] == "Castor":
            alignment = "left"
            style = "rect"
            bg_color = CASTOR_BUBBLE_COLOR
        else:
            alignment = "right"
            style = "rounded"
            bg_color = POLLUX_BUBBLE_COLOR
        lines = get_message_block(current_stream, curr_progress=current_stream.get("progress", 0),
                                  available_width=bubble_max_width - 2 * BUBBLE_PADDING)
        current_group.append({"lines": lines, "alignment": alignment, "style": style,
                              "bg_color": bg_color, "border_color": CHAT_BUBBLE_BORDER_COLOR})
    bubbles.extend(current_group)

    # Determine bubble dimensions.
    (sample_text_size, _) = cv2.getTextSize("Tg", FONT, FONT_SCALE, THICKNESS)
    line_height = sample_text_size[1] + LINE_SPACING

    for b in bubbles:
        if b.get("gap", False):
            continue
        b["w"] = bubble_max_width
        num_lines = len(b["lines"])
        bubble_height = num_lines * line_height + 2 * BUBBLE_PADDING
        b["h"] = bubble_height

    # Position bubbles (the most recent ones at the bottom).
    total_height = 0
    for b in bubbles:
        if b.get("gap", False):
            total_height += b["height"]
        else:
            total_height += b["h"] + BUBBLE_SPACING
    start_y = chat_img.shape[0] - total_height if total_height > chat_img.shape[0] else 0

    current_y = start_y
    for b in bubbles:
        if b.get("gap", False):
            current_y += b["height"]
        else:
            if b["alignment"] == "left":
                b["x"] = MARGIN
            else:
                b["x"] = chat_width - bubble_max_width - MARGIN
            b["y"] = current_y
            draw_chat_bubble(chat_img, b)
            current_y += b["h"] + BUBBLE_SPACING


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Launch the Battle Tower Agent
def start_battle_tower_agent(frame_queue: queue.Queue, result_queue: queue.Queue):
    agent = BattleTowerTwitchAgent(
        frame_queue=frame_queue,
        result_queue=result_queue,
        render=False
    )
    agent.play()

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Handle the Commentator Thread
def run_commentator_loop(video_queue: queue.Queue, message_queue: queue.Queue):
    """
    Runs a loop waiting for videos (i.e. paths to a folder with images in them),
    calls the GeminiCommentators with that path, and then puts the result in the message queue.

    If the video path is ever None, quits.
    """
    commentators = GeminiCommentator()

    while True:
        video_path = video_queue.get(block=True)

        if video_path is None:
            break

        conversation = commentators(video_path)
        if conversation is not None:
            message_queue.put(conversation)

def create_video_dir() -> str:
    """Creates a new directory to store the frames and returns the full path"""

    video_file = f'pkmn_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    video_path = os.path.join(DATA_DIR, 'video', video_file)
    os.makedirs(video_path)

    return video_path


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Main loop.
def main():
    # –––––––––––––––––––––––––––––––––––
    # Conversation state tracking
    finished_message_groups = []  # list of past conversations from Gemini (TODO: prune it as the list gets longer)
    current_conv_messages = []  # finished messages in the current group
    current_message_set = None # the latest conversation from Gemini
    current_msg_index = 0 # which message in the conversation that we're processing
    current_char_index = 0 # we stream characters in the chat window, so this keeps track of which char we're on in the current message
    current_message_finished = False

    # once we're done with a message, we wait to process another message, this is a countdown for the # of cycles until we process the next message
    message_delay_counter = 0

    # –––––––––––––––––––––––––––––––––––
    # Starting/Communicating with the threads

    # A thread-safe queue for incoming conversations from Gemini
    message_queue = queue.Queue()

    # These receive the frames/current scores from the Battle Tower Agent
    frame_queue = queue.Queue(maxsize=5)
    result_queue = queue.Queue()
    video_queue = queue.Queue()

    gemini_thread = threading.Thread(
        target=run_commentator_loop,
        args=(video_queue, message_queue,),
        daemon=True
    )
    gemini_thread.start()

    agent_thread = threading.Thread(
        target=start_battle_tower_agent,
        args=(frame_queue, result_queue),
        daemon=True
    )
    agent_thread.start()

    # –––––––––––––––––––––––––––––––––––
    # Keeping track of run statistics
    num_attempts = 0
    current_streak = 0
    longest_streak = 0

    # –––––––––––––––––––––––––––––––––––
    # Managing the video (to send to Gemini)

    # we'll reset this whenever we start on a new video
    frame_counter = 0

    video_path = create_video_dir()
    prev_video_path = None # this lets us clean up videos that we've already used

    is_initial_video = True

    # –––––––––––––––––––––––––––––––––––
    # Main display loop

    cv2.namedWindow("BattleTowerAgent")
    # we need to have an init frame or else there will be trouble
    frame = np.zeros((SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 3), dtype=np.uint8)

    while True:
        frame_counter += 1

        # Get a frame from the emulator.
        try:
            # the timeout ensures that, even if something freezes up, the UI processing stuff (half a second is probably too much but w/e)
            frame = frame_queue.get(block=True, timeout=.5)
        except:
            pass

        frame_path = os.path.join(video_path, f'frame_{frame_counter:06}.jpg')
        cv2.imwrite(frame_path, frame)

        # the first video needs special processing
        if is_initial_video and frame_counter == FIRST_VIDEO_LENGTH:
            is_initial_video = False
            video_queue.put(video_path) # this starts the whole video processing + sending to Gemini thing
            prev_video_path = video_path
            video_path = create_video_dir()
            frame_counter = 0

        upper_screen = frame[:SCREEN_HEIGHT]
        lower_screen = frame[SCREEN_HEIGHT:]

        # Process any new stats from the results queue.
        try:
            result_dict = result_queue.get(block=False)
            num_attempts = result_dict['num_attempts']
            current_streak = result_dict['current_streak']
            longest_streak = result_dict['longest_streak']
        except Exception:
            pass

        # Build overall canvas.
        canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

        # ––– Draw the results bar at the very top.
        results_bar_color = (30, 30, 30)
        cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, RESULTS_BAR_HEIGHT), results_bar_color, -1)
        status_text = f"Today's Attempts: {num_attempts:<4} Current Streak: {current_streak:<4} Today's Longest Streak: {longest_streak}"
        text_size, baseline = cv2.getTextSize(status_text, RESULT_FONT, RESULT_FONT_SCALE, RESULT_THICKNESS)
        text_x = MARGIN
        text_y = (RESULTS_BAR_HEIGHT + text_size[1]) // 2
        cv2.putText(canvas, status_text, (text_x, text_y), RESULT_FONT, RESULT_FONT_SCALE, (255, 255, 255), RESULT_THICKNESS)

        # ––– Compute dynamic scaling for the DS upper screen. TODO: a lot of this could be precomputed
        available_height = WINDOW_HEIGHT - RESULTS_BAR_HEIGHT
        # Scale so that the upper screen fills all available vertical space.
        upper_scale = available_height / SCREEN_HEIGHT
        U_disp_w = int(SCREEN_WIDTH * upper_scale)
        U_disp_h = available_height
        upper_resized = cv2.resize(upper_screen, (U_disp_w, U_disp_h), interpolation=cv2.INTER_NEAREST)
        canvas[RESULTS_BAR_HEIGHT:RESULTS_BAR_HEIGHT + U_disp_h, 0:U_disp_w] = upper_resized

        # ––– Right column (for messages and inset lower screen).
        RIGHT_COL_X = U_disp_w
        RIGHT_COL_WIDTH = WINDOW_WIDTH - U_disp_w

        # Scale lower screen to fill the right column’s width.
        lower_scale = RIGHT_COL_WIDTH / SCREEN_WIDTH
        L_disp_w = int(SCREEN_WIDTH * lower_scale)  # should equal RIGHT_COL_WIDTH
        L_disp_h = int(SCREEN_HEIGHT * lower_scale)
        lower_resized = cv2.resize(lower_screen, (L_disp_w, L_disp_h), interpolation=cv2.INTER_NEAREST)
        lower_x = WINDOW_WIDTH - L_disp_w
        lower_y = WINDOW_HEIGHT - L_disp_h
        canvas[lower_y:WINDOW_HEIGHT, lower_x:WINDOW_WIDTH] = lower_resized

        # ––– The message region occupies the remaining area in the right column:
        # from x = RIGHT_COL_X to WINDOW_WIDTH, from y = RESULTS_BAR_HEIGHT to y = lower_y.
        msg_region_top = RESULTS_BAR_HEIGHT
        msg_region_bottom = lower_y
        msg_region_left = RIGHT_COL_X
        msg_region_right = WINDOW_WIDTH
        chat_region = canvas[msg_region_top:msg_region_bottom, msg_region_left:msg_region_right]

        # draw divider lines.
        cv2.line(canvas, (U_disp_w, RESULTS_BAR_HEIGHT), (U_disp_w, WINDOW_HEIGHT), (200, 200, 200), 2)
        cv2.line(canvas, (msg_region_left, lower_y), (WINDOW_WIDTH, lower_y), (200, 200, 200), 2)

        # ––– Process incoming messages (after we've processed all the current messages)
        if current_message_set is None:
            # TODO: what happens if there are a lot of frames?
            try:
                current_message_set = message_queue.get(block=False)

                # reset everything related to the conversation
                current_msg_index = 0
                current_char_index = 0
                current_message_finished = False
                current_conv_messages = []  # start a new conversation group
                print("Received new message set:", current_message_set)

                # now we can also send the current video to Gemini
                video_queue.put(video_path)

                if os.path.exists(prev_video_path): # it's polite to clean up the video dir
                    os.removedirs(prev_video_path)

                prev_video_path = video_path
                video_path = create_video_dir()
                frame_counter = 0

            except:
                pass

        current_stream = None
        if current_message_set is not None:
            current_msg = current_message_set[current_msg_index]
            current_msg["progress"] = current_char_index

            if not current_message_finished:
                if current_char_index < len(current_msg["content"]):
                    # Play an audio blip on every other alphanumeric character (kinda like Phoenix Right).
                    if current_char_index % 2 and current_msg["content"][current_char_index].isalnum():
                        if current_msg["role"] == "Castor":
                            audio_file = CASTOR_AUDIO
                        elif current_msg["role"] == "Pollux":
                            audio_file = POLLUX_AUDIO
                        else:
                            audio_file = None
                        if audio_file:
                            playsound(audio_file, block=False)
                    current_char_index += 1
                if current_char_index >= len(current_msg["content"]):
                    current_message_finished = True
                    message_delay_counter = 15  # pause before next message
            else:
                # this controls how many frames we wait before we begin printing the next message
                if message_delay_counter > 0:
                    message_delay_counter -= 1
                else:
                    current_conv_messages.append(current_msg)
                    current_msg_index += 1
                    current_char_index = 0
                    current_message_finished = False
                    if current_msg_index >= len(current_message_set):
                        finished_message_groups.append(current_conv_messages)
                        current_conv_messages = []
                        current_message_set = None

            if current_message_set is not None:
                current_stream = current_message_set[current_msg_index]

        # Draw the chat feed into the message region.
        draw_chat_feed(chat_region, finished_message_groups, current_conv_messages, current_stream)

        cv2.imshow("BattleTowerAgent", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    agent_thread.join()
    gemini_thread.join()
    

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
