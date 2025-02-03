import re
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
import os

import dotenv
dotenv.load_dotenv('../.env')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = 'gemini-2.0-flash-thinking-exp-01-21'
MODEL_NAME = 'gemini-exp-1206'

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
* When you write code, you write everything that is necessary to run the code without errors or exceptions. 
* There are a number of standard data science packages installed in the Python environment that you may use, but you must import them.
  * Matplotlib and Numpy are already imported in the environment (through `import matplotlib.pyplot as plt` and `import numpy as np` respectively), do not re-import them.  
* It is VERY IMPORTANT that you don't re-define those variables in your code or replace them with placeholder values; you *MUST* assume that they are there and as described.
* The underlying data is stored in an sqlite3 database, and you must interact with it and retrieve data through your Python code.
  * There will *always* be a `cursor` object in the Python environment that will let you execute SQL queries in the database and you must use it.
  * **IMPORTANT: DO NOT CLOSE THIS CURSOR**
  * The database has the following schema: {SCHEMA}
* Surround any python code you write with ```python ```. NOTE: only the very last ```python ``` block will be executed, so you must include all of your written code in that final block.
* If a user wants the data visualized, always use matplotlib and be sure to always display the plot after you create it.
  * Not all queries require a plot to be visualized.
  * It's okay to display multiple plots at once depending on the user's request.
"""


PROMPT = """Here is the user's request: {user_prompt}

Write a python script that loads the necessary data from the database (using the `cursor` object and a SQL query), performs any transformations or statistics on the data (if the user requested it) and displays the data using MatPlotLib.
"""


def run_terminal_loop(cursor):

    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    chat = model.start_chat()

    user_prompt = input('What would you like to see? ')

    while user_prompt.lower().strip() not in ('quit', 'q'):

        prompt = PROMPT.format(user_prompt=user_prompt)

        response = chat.send_message(content=prompt).text

        print(MODEL_NAME,  'response:\n', response)

        scripts = re.findall('```python(.+?)```', response, re.DOTALL)

        if not scripts:
            print('Model did not generate any valid scripts')
        else:
            script = scripts[-1]

            try:
                exec(script)
            except Exception as e:
                print('While executing script, got exception:', e)

        print('\n', flush=True)


        user_prompt = input('What would you like to see? ')


if __name__ == '__main__':
    db_path = 'battle_tower.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    run_terminal_loop(cursor)
