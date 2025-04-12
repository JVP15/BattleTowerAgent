import pathlib
import re
import sqlite3
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
import os

import dotenv

# we expect the .env in BattleTowerAgent/.env
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
SCRIPT_DIR = os.path.join(ROOT_DIR, 'data', 'tmp')
os.makedirs(SCRIPT_DIR, exist_ok=True)

DB_PATH = os.path.join(ROOT_DIR, 'battle_tower_agent', 'battle_tower_database', 'battle_tower.db')

dotenv.load_dotenv(os.path.join(ROOT_DIR, '.env'))

MODEL_NAME = 'gemini-2.0-flash-thinking-exp-01-21'
#MODEL_NAME = 'gemini-2.5-pro-preview-03-25'

# What % of streaks with the strategy "A" make it to battle 22? How about 50? What about for the strategy "max_damage"?
SCHEMA = """TABLES:
streaks:
    - id: INTEGER, primary key, auto-incrementing
    - start_timestamp: TIMESTAMP
    - end_timestamp: TIMESTAMP

battles:
    - id: INTEGER, primary key, auto-incrementing
    - streak_id: INTEGER, foreign key referencing streaks(id)
    - battle_number: INTEGER
    - battle_duration: INTEGER
    - pokemon_team: TEXT
    - strategy: TEXT
    - win: BOOLEAN
    - timestamp: TIMESTAMP
"""

SYSTEM_PROMPT = f"""You are an AI data scientist that helps users calculate and visualize statistics based on a database. Here are some rules/guidelines for you to follow:
* You can write Python code, and this code will be executed in a Python interpreter.
* When you write code, you must write everything that is necessary to run the code without errors or exceptions. 
* There are a number of standard data science packages installed in the Python environment that you may use, but you must import them.
  * Matplotlib and Numpy are already imported in the environment do not re-import them.    
* It is VERY IMPORTANT that you don't re-define those variables in your code or replace them with placeholder values; you *MUST* assume that they are there and as described.
* The underlying data is stored in an sqlite3 database, and you must interact with it and retrieve data through your Python code.
  * There will *always* be a `cursor` object in the Python environment that will let you execute SQL queries in the database and you must use it.
  * The database has the following schema: {SCHEMA}
  * **IMPORTANT: DO NOT OVERWRITE THE CURSOR**
* Surround any python code you write with ```python ```. NOTE: only the very last ```python ``` block will be executed, so you must include all of your written code in that final block.
* If a user wants the data visualized, always use matplotlib and be sure to always display the plot after you create it.
  * Not all queries require a plot to be visualized; don't create a plot unless the user's request requires it.
  * It's okay to display multiple plots at once depending on the user's request.
"""


PROMPT = """Here is the user's request: {user_prompt}

Write a python script that loads the necessary data from the database (using the `cursor` object and a SQL query), performs any transformations or statistics on the data (if the user requested it) and displays the data using MatPlotLib.
"""

CODE_IMPORTS_AND_CURSOR = """import matplotlib.pyplot as plt
import numpy as np
import sqlite3
db_path = r'{db_path}'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
"""


def execute_code_in_subprocess(code_string, db_path=DB_PATH):
    code_string = CODE_IMPORTS_AND_CURSOR.format(db_path=db_path) + '\n' + code_string
    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=SCRIPT_DIR, encoding='utf-8')
    tmp_file.write(code_string)
    tmp_file_path = tmp_file.name
    tmp_file.close() # close so that it gets written

    process = subprocess.run(
        ["python", tmp_file_path],  # Assumes 'python' is in PATH
        capture_output=True,
        text=True,  # Decode stdout and stderr as text
    )

    stdout = process.stdout
    stderr = process.stderr

    #if os.path.exists(tmp_file_path):
    #    os.remove(tmp_file_path)

    return stdout, stderr


def run_terminal_loop(model_name, db_path=DB_PATH):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    print(f'{ROOT_DIR=}')
    print(os.path.join(ROOT_DIR, '.env'))

    if GEMINI_API_KEY is None:
        raise ValueError('You must have `GEMINI_API_KEY` set in the environment (or .env file) to let Gemini query the DB.')

    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_PROMPT)
    chat = model.start_chat()

    print(f'Connected to database, {model_name} is ready to respond to your queries. Enter (q)uit to exit the loop.')

    user_prompt = input('What would you like to see? ')

    while user_prompt.lower().strip() not in ('quit', 'q'):

        prompt = PROMPT.format(user_prompt=user_prompt)

        response = chat.send_message(content=prompt).text

        print(model_name,  'response:\n', response)

        scripts = re.findall('```python(.+?)```', response, re.DOTALL)

        if not scripts:
            print('Model did not generate any valid scripts')
        else:
            script = scripts[-1]

            try:
                stdout_output, stderr_output = execute_code_in_subprocess(script, db_path=db_path)
                print(stdout_output)

                if stderr_output:
                    raise ValueError(stderr_output)
            except Exception as e:
                print('While executing script, got exception:', e)

        print('\n', flush=True)

        user_prompt = input('What would you like to see? ')


if __name__ == '__main__':
    run_terminal_loop(model_name=MODEL_NAME)
