import sqlite3
import datetime
import os
import pathlib

# I like the idea that the default DB path just points to this folder
DATA_DIR = pathlib.Path(__file__).parent.resolve()
DB_PATH = os.path.join(DATA_DIR, 'battle_tower.db')

# this class was created w/ AI assistance b/c I don't know databases very well (and I *really* don't know SQL)
class BattleTowerDatabase:
    def __init__(self, db_name=DB_PATH):
        self.db_name = db_name
        self.initialize_schema()

    def _connect(self):
        """Connect to the database and return a cursor."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        return conn, cursor

    def _close(self, conn, cursor):
        """Close the database connection."""
        conn.commit()
        cursor.close()
        conn.close()

    def initialize_schema(self):
        """
        Initializes the database schema by creating the necessary tables
        if they do not already exist.
        """
        conn, cursor = self._connect()

        # Create the 'streaks' table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS streaks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_timestamp TIMESTAMP,
                end_timestamp TIMESTAMP
            )
        """)

        # Create the 'battles' table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS battles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                streak_id INTEGER,
                battle_number INTEGER,
                battle_duration INTEGER,
                pokemon_team TEXT,
                strategy TEXT,
                win BOOLEAN,
                timestamp TIMESTAMP,
                FOREIGN KEY (streak_id) REFERENCES streaks(id)
            )
        """)

        self._close(conn, cursor)

    def start_streak(self):
        """Starts a new streak and returns the streak ID."""
        conn, cursor = self._connect()
        start_timestamp = datetime.datetime.now()
        cursor.execute(
            "INSERT INTO streaks (start_timestamp) VALUES (?)",
            (start_timestamp, )
        )
        streak_id = cursor.lastrowid
        self._close(conn, cursor)
        return streak_id

    def add_battle(self, streak_id, battle_number, battle_duration, pokemon_team, strategy, win):
        """Adds a battle record to the database."""
        conn, cursor = self._connect()
        timestamp = datetime.datetime.now()
        cursor.execute(
            "INSERT INTO battles (streak_id, battle_number, battle_duration, pokemon_team, strategy, win, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (streak_id, battle_number, battle_duration, pokemon_team, strategy, win, timestamp)
        )
        self._close(conn, cursor)

    def end_streak(self, streak_id):
        """Ends a streak by updating the end_timestamp"""
        conn, cursor = self._connect()
        end_timestamp = datetime.datetime.now()
        cursor.execute(
            "UPDATE streaks SET end_timestamp = ? WHERE id = ?",
            (end_timestamp, streak_id)
        )
        self._close(conn, cursor)