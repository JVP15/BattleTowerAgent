# Battle Tower Agent

This is an AI Agent (using "AI" as in "video game AI" not "neural network" AI) that can play Pokémon Platinum's Battle Tower (Singles).
It controls Pokémon Platinum using [py-desmume](https://github.com/SkyTemple/py-desmume), and you'll have to supply your own ROM file (see Setup).

The goal is to see how far we can go in the Battle Tower. The longest streak I've done is 135 wins. 
The longest streak that my one of my Agents have done is **1227 wins**. That would (unofficially) place it in 3rd on the 
[Pokemon Pokémon Battle Tower Singles leaderboard](https://www.smogon.com/forums/threads/4th-generation-battle-facilities-discussion-and-records.3663294/)
But that's not enough. I want to shoot for the #1 spot and we still have a ways to go.

## Running the Agents

---

TODO

## Setup

---

```bash
git clone https://github.com/JVP15/BattleTowerAgent.git
cd BattleTowerAgent
pip install -e .
```

I've tested this on a Windows 10 and 11 machine with Python 3.10 and 3.11, so if your setup looks like that, you're good to go.
Linux support *is* coming (slowly). I've created the savestates for Linux but there are still some problems my multiprocessing code on Linux.

NOTE: you have to supply your own ROM, I won't include one (or a link to any) in this repo. 
It needs to be a US copy of Pokémon Platinum to work with the savestates I've created.
Once you have the ROM, put it in the `ROM` folder and name it `Pokemon - Platinum.nds` (so should look like `ROM/Pokemon - Platinum.nds`).

## Agents

---

Right now, this repo has 3 strategies/agents. 

| Strategy  | Win Streak | Usage                  | Notes                                                                                                                                                               |
|-----------|------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A         | 115        | '--strategy a'         | Throughout the entire Battle Tower, this agent only hits the 'A' button. There is no decision making and yet it still handily beats Palmer                          |
| Search v1 | 383        | '--strategy search_v1' | This is a 'basic' search that chooses each possible move, plays the game until the end, and then chooses the move that won, or at least ended the game the quickest |
| Search v2 | 1227[^1]   | '--strategy search_v2' | This uses an improved search algorithm that tracts damage dealt, searches over which Pokémon to swap to, and has a number of efficiency improvements.               |

[^1]: When the v2 Search agent got the 1227 winstreak, searching over the next Pokémon to swap to wasn't enabled.

## Pokémon Teams

---

I use a Garchomp, Suicune, and Scizor team for my agents. 
There is definitely room for improvement but this team has led me well both in my personal games and for these agents.
You can check out the exact team builds here:
* [Only A](https://pokepast.es/cdad5c488ee8329d)
* [Search Agents](https://pokepast.es/db129847a7a0e6d5)

