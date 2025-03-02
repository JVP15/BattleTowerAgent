import base64
import os
import pathlib
import queue
import random
import time
import ffmpeg

import dotenv
from google import genai
from google.genai import types

from pydantic import BaseModel

import logging

logger = logging.getLogger('GeminiAPI')
logger.setLevel(logging.DEBUG)

# we expect the .env in BattleTowerAgent/.env
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
dotenv.load_dotenv(os.path.join(ROOT_DIR, '.env'))


# we'll put the logs in BattleTowerAgent/logs/gemini/commentator.log
LOG_DIR = os.path.join(ROOT_DIR, 'log', 'gemini')
LOG_FILE = os.path.join(LOG_DIR, 'commentator.log')
os.makedirs(LOG_DIR, exist_ok=True)

file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
# by default, asctime includes miliseconds, but I don't want them so... boo
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


MODEL = 'gemini-2.0-flash'
COMMENTATORS = ['Pollux', 'Castor']
MAX_CONVERSATION_LEN = 5
MIN_CONVERSATION_LEN = 2

SYSTEM_PROMPT = """Simulate a sports commentary between two personalities, Castor and Pollux, in response to a video snippet from a Pokémon game, following the structure of a sports newscast discussion.

- Castor and Pollux are knowledgeable about Pokémon battles and esports and their interactions should reflect this expertise.
- Each comment should be concise, styled like a sports commentator's quick take, consisting of 1-2 sentences.
- The conversation will alternate between Castor and Pollux, beginning with the specified commentator.
- Respond to the immediate events or actions depicted in the video, maintaining focus on the game.

# Steps

1. Watch or analyze the video snippet provided.
2. Generate a short (2 sentences max) summary  that focuses on the factual events that happened in the video.
3. Identify key actions, strategies, or moments in the Pokémon battle that would excite or interest a sports audience.
4. Begin the conversation with the specified commentator, Castor or Pollux.
5. Alternate sentences between Castor and Pollux for the specified number of messages.
6. Ensure each sentence is reactive to either the video or the preceding comment to maintain a dynamic dialogue.

# Output Format

The commentary should be output as a JSON object with a list of messages, where each entry consists of the role ('Castor' or 'Pollux') and the content of their message.

# Examples

**Example 1:**
- Video: [Description of the video clip]
- First Commentator: Pollux
- Number of Messages: 4

**Output:**
{
    'summary': "A brief 1-2 sentence factual summary of the important events that occured in the video."
	'messages': 
	[
	  {'role': 'Pollux', 'content': 'Rookie trainer is opting for an aggressive start!'}
	  {'role': 'Castor', 'content': 'Charizard responding with a defensive maneuver.'}
	  {'role': 'Pollux', 'content': 'The crowd is clearly on edge after that move!'}
	  {'role': 'Castor', 'content': 'Indeed, Pollux. The stakes are sky-high in this battle.'}
	]
}


# Notes

- Ensure each message logically follows the previous one and pertains to the events in the video.
- The focus should be on providing engaging, sports-style commentary that keeps viewers entertained and informed.
- Adjust tone and excitement level according to the video content while maintaining professionalism as commentators.
"""

USER_PROMPT = """First Commentator: {commentator}
Number of Messages: {num_messages}"""

# Ideas to improve commentary:
# 1. Include history (i.e. the summaries from the last N, 5?? videos)
# 2. Instruct the LLM to go off on a tangent w/ a small probability (e.g. 5%)
# 3. Maybe instruct the LLM to talk about other stuff during downtime/inbetween battles (random topics? Quantum Physics? Math? Whatever)


class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    summary: str
    messages: list[Message]



class GeminiCommentator:
    def __init__(self, model_name=MODEL, user_prompt=USER_PROMPT, system_prompt=SYSTEM_PROMPT):
        self.model_name = model_name
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt

        self.history = []

        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        self.generation_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=Conversation,
            system_instruction=self.system_prompt
        )


    def prepare_and_upload_video(self, video_path) -> types.File:
        start = time.time()
        video_filepath = os.path.join(video_path, 'video.mp4')

        out, err = (
            ffmpeg
            .input(os.path.join(video_path, 'frame_%06d.jpg'), framerate=30)
            .output(video_filepath, loglevel="quiet") # thank you https://stackoverflow.com/a/76146125/8095352
            .run(overwrite_output=True)
        )
        if err:  # TODO: figure out what to do if there is an error with the video procesisng and upload
            logger.error(err)

        video_remote = self.client.files.upload(file=video_filepath)

        # we have to wait for the video to be ready for processing
        while video_remote.state.name == "PROCESSING":
            time.sleep(.5)
            video_remote = self.client.files.get(name=video_remote.name)

        logger.info(f'Took {time.time() - start:.3f} seconds to process and upload {video_path}')

        return video_remote

    def __call__(self, video_path) -> list[dict] | None:
        """
        Prompts Gemini to create commentary between Castor and Pollux on the provided video.

        Args:
            video_path: A directory containing images with the format frame_000001.jpg, frame_000002.jpg, ...
                This will turn those images into a video and upload it to Google's servers, then send it to Gemini.
        Returns:
             A list of dictionaries like [{'role': 'Castor', 'content': '...'}, {'role': 'Pollux', ...}]
            If an error occurs anywhere in this process, will return None instead
        """
        try:
            video_file = self.prepare_and_upload_video(video_path)
        except Exception as e:
            logger.exception('Error uploading file') # thank you https://stackoverflow.com/questions/5191830/how-do-i-log-a-python-error-with-debug-information
            return None

        num_messages = random.randint(MIN_CONVERSATION_LEN, MAX_CONVERSATION_LEN)
        role = random.choice(COMMENTATORS)

        prompt = self.user_prompt.format(commentator=role, num_messages=num_messages)

        contents = [
            video_file,
            prompt
        ]

        start = time.time()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self.generation_config,
        )

        logger.info(
            f' Took {time.time() - start:.3f} seconds to call {self.model_name}.'
            f' Input tokens: {response.usage_metadata.prompt_token_count}.'
            f' Output tokens: {response.usage_metadata.candidates_token_count}.'
        )

        logger.debug(f'Prompt:\n{prompt}\nResponse:\n{response.text}')
        json_output = response.parsed.dict()

        conversation = json_output.get('messages')

        # TODO: add history support (commenting this out to avoid
        # summary = json_output.get('summary')
        # if summary:
        #     self.history.append(summary)

        return conversation


if __name__ == '__main__':
    video_path = r'C:\Users\jorda\Documents\Python\CynthAI\GeminiPlaysPokemon\data\video\pkmn_20250228-205616'

    commentator = GeminiCommentator()
    out = commentator(video_path)

    print(out)

