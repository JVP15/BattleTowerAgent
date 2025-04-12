import argparse
import logging
import sqlite3

# this also sets the BUTTON_PRESS log level
from battle_tower_agent.agent import BattleTowerAAgent
from battle_tower_agent.battle_tower_database.visualize import run_terminal_loop
from battle_tower_agent.search_agent import BattleTowerSearchAgent
from battle_tower_agent.search_agent_v2 import BattleTowerSearchV2Agent
from battle_tower_agent.max_agent import BattleTowerMaxDamageAgent

from battle_tower_agent.battle_tower_database.interface import BattleTowerServerDBInterface

from battle_tower_agent.battle_tower_database.server import create_app
from battle_tower_agent.battle_tower_database.database import DB_PATH

def setup_logging(log_level_str):
    """Sets up logging based on the provided log level string."""
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "BUTTON_PRESS": logging.BUTTON_PRESS,
    }

    level = log_levels.get(log_level_str.upper(), logging.INFO) # Default to INFO if invalid
    logging.basicConfig(level=level)

def parse_args():
    parser = argparse.ArgumentParser(description="Running the Battle Tower Agent")

    # Mutually exclusive group for the main modes
    group = parser.add_mutually_exclusive_group(required=True) # Ensure one of these is required
    group.add_argument('--strategy', type=str, choices=['a', 'search_v1', 'search_v2', 'max_damage'],
                       help='Runs the Battle Tower Agent with using the provided strategy (choose from: a, search_v1, search_v2, max_damage).')

    group.add_argument('--server', action='store_true', help= ("Runs the database server."
                                                               " This lets the agent save it's stats (battle #s, streak lengths, etc)"
                                                               " to a database that can be queried later.")
    )

    group.add_argument('--visualize', action='store_true', help= ('Opens a terminal chat with Gemini that lets you query the database,'
                                                                  ' run statistics, and generate plots using only natural language.'))

    # Optional arguments
    parser.add_argument('--render', action='store_true', help='Renders the Battle Tower Agent while it is running.')
    parser.add_argument('--log-db', action='store_true', help=("Causes the Battle Tower Agent to log it's results to the database. "
                                                               "NOTE: you must already have the database server running"
                                                               " i.e. by calling `python run.py --serve` in a different terminal.")
    )

    parser.add_argument('--savestate-file', type=str, help='Custom path to a Pokemon Platinum DeSmuME game state.')
    parser.add_argument('--db-path', type=str, help='Custom path to a different database file.')
    parser.add_argument(
        '--log-level', type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "BUTTON_PRESS"],
        default="INFO", help=('Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, BUTTON_PRESS). Default: INFO'
                               ' The Battle Tower Agents have a lot of logging, especially with DEBUG mode.')
    )

    parser.add_argument('--gemini-model-name', type=str, default='gemini-2.0-flash-thinking-exp-01-21', help='The Gemini model that you will chat with when you run `--visualize`')

    # TODO: add more args like server port and url 'n stuff, also additional arguments specific to different strategies like this:

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    setup_logging(args.log_level)

    if args.strategy:
        savestate_kwargs = {}
        if args.savestate_file:
            savestate_kwargs['savestate_file'] = args.savestate_file

        db_interface = BattleTowerServerDBInterface() if args.log_db else None
        if args.strategy == 'a':
            agent = BattleTowerAAgent(render=args.render, db_interface=db_interface, **savestate_kwargs)
        elif args.strategy == 'search_v1':
            agent = BattleTowerSearchAgent(render=args.render, db_interface=db_interface, **savestate_kwargs)
        elif args.strategy == 'search_v2':
            agent = BattleTowerSearchV2Agent(render=args.render, db_interface=db_interface, **savestate_kwargs)
        elif args.strategy == 'max_damage':
            agent = BattleTowerMaxDamageAgent(render=args.render, db_interface=db_interface, **savestate_kwargs)
        else:
            raise ValueError('Invalid strategy provided. Supported strategies are [a, search_v1, search_v2')

        agent.play()

    elif args.server:
        app = create_app(db_path=args.db_path if args.db_path else DB_PATH)
        app.run(debug=True)

    elif args.visualize:
        conn = sqlite3.connect(args.db_path if args.db_path else DB_PATH)
        cursor = conn.cursor()

        run_terminal_loop(cursor, args.gemini_model_name)
    else:
        raise ValueError('One of `--strategy`, `--visualize`, or `--serve` must be provided.')

if __name__ == "__main__":
    main()