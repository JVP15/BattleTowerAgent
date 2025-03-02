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
import multiprocessing
from collections import deque
from functools import partial

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
from battle_tower_agent.display_agent import create_battle_tower_display_agent

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

CHARS_PER_FRAME = 3  # Number of characters to display per frame (higher = faster text)
AUDIO_BLIP_FREQUENCY = 8  # Play a blip every N characters (must be >= 1, keeps the audio from playing rapid-fire)

# Result bar settings
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
FIRST_VIDEO_LENGTH = VIDEO_FPS * 7 # idk, 7 seconds seems like a good intro video to send to Gemini
MAX_VIDEO_LENGTH = 20 * VIDEO_FPS # this limits the length of the video we'll send to Gemini, which makes sure it only covers the most recent, relevant parts.

# Video/Commentary Synchronization Settings
COMMENTARY_SYNC_DELAY_SECONDS = 9  # Seconds to delay frame display (gives Gemini time to respond to the initial video and go from there)

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Audio Settings
CASTOR_AUDIO = os.path.join(DATA_DIR, 'audio', 'castor.wav')
POLLUX_AUDIO = os.path.join(DATA_DIR, 'audio', 'pollux.wav')

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
# Multitreading/Processing stuff

def start_battle_tower_agent(frame_queue: queue.Queue, result_queue: queue.Queue, battle_tower_agent_is_ready: threading.Event):
    """
    Launched the Battle Tower Agent. When DeSmuME is initialized (which takes a second or two, will set
    `battle_tower_agent_is_ready` (which is useful to synchronize the audio, DeSmuME has a variable init time).
    """
    agent = create_battle_tower_display_agent(
        frame_queue=frame_queue,
        result_queue=result_queue,
    )

    battle_tower_agent_is_ready.set()
    agent.play()

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

def play_chat_audio_process(audio_queue: multiprocessing.Queue):
    """
    Okay this takes some explaining. Basically, I can delay the video, but not the audio with DeSmuME.
    But when I use OBS to stream this stuff, I can specify application audio sources and apply a delay specificaly to it.
    Unfortunately, I need to actually have a different "application" (read: process) for OBS, and it also needs its
    own window. But this basically allows me to apply a 7 second delay to the Pokemon game in this code, and then a
    7-second delay to the audio in OBS, leaving the video and audio for the chat feed untouched.

    Also, OBS needs a window to latch onto, which is why I need to create a persistent window here.
    I don't like that window, so when I'm not using this for twitch streaming, I just use playsound like normal.
    """
    cv2.namedWindow("BattleTowerAgentChatAudio")

    while True:
        audio_file = audio_queue.get()
        playsound(audio_file, block=False)

def create_video_dir() -> str:
    """Creates a new directory to store the frames and returns the full path"""

    video_file = f'pkmn_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    video_path = os.path.join(DATA_DIR, 'video', video_file)
    os.makedirs(video_path)

    return video_path


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Main loop.
def main(display_for_twitch_streaming=True):
    """
    Runs the commentator display.
    If `display_for_twitch_streaming` is True, then we apply some little QoL things like delaying the video so that
    Gemini can respond in more "real-time" (note: real-time streaming w/ Gemini doesn't fix this).
    Since that causes the audio to go out of sync, there is additional support for OBS to capture the game and chat feed
    audio separately.

    If it's false, then we just play things as they are.
    """

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
    # Starting/Communicating with the threads/processes

    # This handle incoming conversations from Gemini
    message_queue = queue.Queue()

    # These receive the frames/current scores from the Battle Tower Agent
    frame_queue = queue.Queue(maxsize=5)
    result_queue = queue.Queue()
    video_queue = queue.Queue()

    # DeSmuME takes a while (and variable amount of time) to init, and this helps us stay synchronized
    #  (at least when we are using another program to process the audio and video separately)
    battle_tower_agent_is_set = threading.Event()

    gemini_thread = threading.Thread(
        target=run_commentator_loop,
        args=(video_queue, message_queue,),
        daemon=True
    )
    gemini_thread.start()

    agent_thread = threading.Thread(
        target=start_battle_tower_agent,
        args=(frame_queue, result_queue, battle_tower_agent_is_set),
        daemon=True
    )
    agent_thread.start()

    battle_tower_agent_is_set.wait()

    # okay this takes some explaining, but which you can find in the `play_chat_audio_process` fn

    if display_for_twitch_streaming:
        # audio_queue = multiprocessing.Queue()
        # audio_process = multiprocessing.Process(
        #     target=play_chat_audio_process,
        #     args=(audio_queue, ),
        # )
        #
        # audio_process.start()
        #
        # play_chat_audio = audio_queue.put
        # TODO: the text blip sound really messes w/ Twitch streaming, causing the audio and video to buffer a lot
        # Instead of playing an audio blip on chat, I'll play a message notification sound, but that is for future me.
        play_chat_audio = lambda audio: None
    else:
        play_chat_audio = partial(playsound, block=False)

    # –––––––––––––––––––––––––––––––––––
    # Keeping track of run statistics
    num_attempts = 0
    current_streak = 0
    longest_streak = 0

    # –––––––––––––––––––––––––––––––––––
    # Managing the video (to send to Gemini)

    frame_counter = 0 # we reset this whenever we start on a new video

    video_path = create_video_dir()
    prev_video_path = None # this lets us clean up videos that we've already used

    is_initial_video = True

    # depending on the application, we may apply a delay before actually showing the video of the Agent
    # this is to let Gemini have some "catch up time" so that it's outputs don't reference stuff 10 seconds ago
    frame_buffer = deque()
    delay_frames_needed = COMMENTARY_SYNC_DELAY_SECONDS * VIDEO_FPS if display_for_twitch_streaming else 0

    # we also need the results to be in-sync, so a simple solution is to just use a results buffer too
    result_buffer = deque([None] * delay_frames_needed)

    # –––––––––––––––––––––––––––––––––––
    # Main display loop

    cv2.namedWindow("BattleTowerAgent")
    # we need to have an init frame or else there will be trouble
    frame = np.zeros((SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 3), dtype=np.uint8)

    while True:
        start_time = time.perf_counter()

        # Get a frame from the emulator.
        try:
            new_frame = frame_queue.get(block=True, timeout=.5)

            # Always save the frame for the video (b/c we need these frames for Gemini)
            frame_path = os.path.join(video_path, f'frame_{frame_counter:06}.jpg')
            cv2.imwrite(frame_path, new_frame)
            frame_counter += 1

            # I don't want the videos being sent to Gemini to be too long (with the current set up, it *shouldn't* happen, but this will enforce that)
            if frame_counter > MAX_VIDEO_LENGTH:
                old_frame_path = os.path.join(video_path, f'frame_{frame_counter - MAX_VIDEO_LENGTH:06}.jpg')
                if os.path.exists(old_frame_path):
                    os.remove(old_frame_path)

            # to accomondate for the video delay, we actually get the frame from the frame buffer
            # NOTE: this is dependant on the video FPS being relatively stable
            frame_buffer.append(new_frame)

            if len(frame_buffer) > delay_frames_needed:
                frame = frame_buffer.popleft()

        except:
            pass

        # the first video needs special processing
        if is_initial_video and frame_counter == FIRST_VIDEO_LENGTH:
            is_initial_video = False
            video_queue.put(video_path) # this starts the whole video processing + sending to Gemini thing
            prev_video_path = video_path
            video_path = create_video_dir()
            frame_counter = 0

        upper_screen = frame[:SCREEN_HEIGHT]
        lower_screen = frame[SCREEN_HEIGHT:]

        try:
            new_result_dict = result_queue.get(block=False)

            # instead of immediately updating the results, we use the result buffer (which automatically handles the delay)
            result_buffer.append(new_result_dict)
        except Exception:
            result_buffer.append(None)

        result_dict = result_buffer.popleft()

        if result_dict:
            num_attempts = result_dict['num_attempts']
            current_streak = result_dict['current_streak']
            longest_streak = result_dict['longest_streak']

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
                    # Figure out how many chars to add this frame
                    chars_to_add = min(CHARS_PER_FRAME, len(current_msg["content"]) - current_char_index)

                    # Check if we should play an audio blip (based on character positions)
                    for i in range(chars_to_add):
                        char_pos = current_char_index + i
                        if (
                                char_pos % AUDIO_BLIP_FREQUENCY == 0
                                and char_pos < len(current_msg["content"])
                                and current_msg["content"][char_pos].isalnum()
                        ):
                            if current_msg["role"] == "Castor":
                                audio_file = CASTOR_AUDIO
                            elif current_msg["role"] == "Pollux":
                                audio_file = POLLUX_AUDIO
                            else:
                                audio_file = None
                            if audio_file:
                                play_chat_audio(audio_file) # this works w/ both the process and plain-ol playsound method
                                break  # Only play one blip per frame

                    current_char_index += chars_to_add
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
        elapsed_s = time.perf_counter() - start_time
        remaining_ms = max(1, int( (1 / VIDEO_FPS - elapsed_s) * 1000))
        # this keeps the FPS relatively stable, which is vital because text streaming is tied to FPS
        key = cv2.waitKey(remaining_ms) & 0xFF
        if key == ord("q"):
            break

    agent_thread.join()
    gemini_thread.join()

    if display_for_twitch_streaming:
        #audio_process.join() # see audio TODO above
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(display_for_twitch_streaming=True)
