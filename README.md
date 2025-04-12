# Battle Tower Agent

![image](data/battle_tower.png)

This is an AI Agent (using "AI" as in "video game AI" not "neural network" AI) that can play Pokémon Platinum's Singles Battle Tower .
It controls Pokémon Platinum using [py-desmume](https://github.com/SkyTemple/py-desmume), and you'll have to supply your own ROM file (see Setup).

The goal is to see how far we can go in the Battle Tower. The longest streak I've done is 135 wins. 
The longest streak that my one of my Agents have done is **1227 wins**. That would (unofficially) place it in 3rd on the 
[Pokemon Pokémon Battle Tower Singles leaderboard](https://www.smogon.com/forums/threads/4th-generation-battle-facilities-discussion-and-records.3663294/)
But that's not enough. I want to shoot for the #1 spot, and it still has a ways to go.

## Running the Agents

---

After your environment is set up (see the Setup section below), the simplest way to get started is to run the following command: 
which will launch the A Agent and render it while the agent is running.

```bash
python run.py --strategy a --render
```

* `--strategy a`: This launches the Battle Tower Agent using the Only A strategy, which only ever hits the A button in the battle tower.. You can swap 'a' for `search_v1` or `search_v2` to try the other agents (see Agents for details).
* `--render`: By default, the agents are headless, i.e. there is no UI, but this renders the battles. NOTE: this slows down battling by about 50%, so if you are just going for a long streak, don't use it.

### Database Server:

Each Battle Tower Agent keeps track of its current streak length and best streak, but when the agent stops, all progress is lost. 
The agent can save its progress to a database, which can be later queried for stats and plots. To run the database server, do:
```bash
python run.py --serve
```

This will start up a flask server for the DB. Now, in another terminal, run the agent with the `--log-db` flag like:

```bash
python run.py --strategy search_v2 --log-db
```

### Visualizing the Results:

Now one *could* open up a SQL terminal or python script, write a bunch of queries yourself, and then code up some plots to visualize the results of the Battle Tower Agents,
but that's a lot of work that I don't want to do. So I wrote a shell that uses Gemini to do it for you. You can run:
```bash
python run.py --visualize
```

And then enter your queries like: "give me the average # of battles in each streak". 
Gemini will write and execute the code and print the results or display them using MatPlotLib. 
It keeps track of the chat history, so you can reference previous commands like: "the agent has been running for a while, do that again."
You can exit the loop by typing `quit` or just `q`.

*NOTE 1:* You must supply a Gemini API key using the `GEMINI_API_KEY` environment variable to do this. 
You can either set `GEMINI_API_KEY` directly in the environment 
or create a `.env` file in the root directory (i.e. under `BattleTowerAgent`, same folder as `run.py`) or set `GE

*NOTE 2:* This works under the hood by Gemini generating code and then running that code using `exec`. 
It is possible that Gemini can write harmful code (although I haven't seen anything like that in testing) 
that *gets automatically executed* so just keep that in mind as you prompt it.

### Other Useful Options:
* `--log-level`: The Battle Tower Agent has verbose logging (based on `logging` levels, including an extra one called `BUTTON_PRESS` that logs *every single button press by the agent).
By default it is `INFO`, but set it to `DEBUG` if you want to see extra info about each state the agent reaches.
* `--gemini-model-name`: If you want to specify the Gemini model that you're using to run stats and visualize the DB. 
* `--db-path`: If you want the server or agent to point to a different database, you can use this option (though I wouldn't recommend it unless you have a good reason).

*NOTE:* `--strategy`, `--visualize`, and `--serve` are all incompatible with each other, so you must run each of them on a seperate process/terminal.

## Setup

---

```bash
git clone https://github.com/JVP15/BattleTowerAgent.git
cd BattleTowerAgent
pip install -e .
```

I've tested this on a Windows 10 and 11 machine with Python 3.10 and 3.11, so if your setup looks like that, you're good to go.
Linux support *is* coming (slowly). I've created the savestates for Linux but there are still some problems my multiprocessing code on Linux.

*NOTE:* you have to supply your own ROM, I won't include one (or a link to any) in this repo. 
It needs to be a US copy of Pokémon Platinum to work with the savestates I've created.
Once you have the ROM, put it in the `ROM` folder and name it `Pokemon - Platinum.nds` (so should look like `ROM/Pokemon - Platinum.nds`).

## Agents

---

Right now, this repo has 3 strategies/agents. 

| Strategy  | Win Streak | Usage                  | Notes                                                                                                                                                             |
|-----------|------------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A         | 115        | '--strategy a'         | Throughout the entire Battle Tower, this agent only hits the 'A' button. There is no decision making and yet it still handily beats Palmer                        |
| Search v1 | 383        | '--strategy search_v1' | This is a 'basic' search that chooses each possible move, plays the game until the end, and then chooses the move that won, or at least ended the game the quickest |
| Search v2 | 1227[^1]   | '--strategy search_v2' | This uses an improved search algorithm that tracts damage dealt, searches over which Pokémon to swap to, and has a number of efficiency improvements.             |
| Max Damage| 147 | '--strategy max_damage' | This is the classic "pick the strongest move" strategy                                                                                                            |

[^1]: When the v2 Search agent got the 1227 winstreak, searching over the next Pokémon to swap to wasn't available.

## Pokémon Teams

---

I use a Garchomp, Suicune, and Scizor team for my agents. 
There is definitely room for improvement but this team has led me well both in my personal games and for these agents.
You can check out the exact team builds here:
* [Only A](https://pokepast.es/cdad5c488ee8329d)
* [Search Agents](https://pokepast.es/db129847a7a0e6d5)

