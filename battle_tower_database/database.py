import sqlite3
import datetime
import os
import pathlib

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

    def get_average_streak_length(self):
        """Calculates the average streak length in battles."""
        conn, cursor = self._connect()
        cursor.execute("""
            SELECT AVG(battle_count)
            FROM (
                SELECT streak_id, COUNT(*) as battle_count
                FROM battles
                GROUP BY streak_id
            )
        """)
        result = cursor.fetchone()
        self._close(conn, cursor)
        return result[0] if result and result[0] is not None else 0

    def get_streaks_at_least(self, min_length):
        """Calculates the amount of streaks with at least a min_length"""
        conn, cursor = self._connect()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM (
                SELECT streak_id, COUNT(*) as battle_count
                FROM battles
                GROUP BY streak_id
                HAVING battle_count >= ?
            )
            """,
            (min_length,)
        )
        result = cursor.fetchone()
        self._close(conn, cursor)
        return result[0] if result and result[0] is not None else 0

    def get_longest_streak(self):
        """Gets the longest streak in battle_count"""
        conn, cursor = self._connect()
        cursor.execute(
            """
            SELECT streak_id, COUNT(*) as battle_count
            FROM battles
            GROUP BY streak_id
            ORDER BY battle_count DESC
            LIMIT 1
            """
        )
        result = cursor.fetchone()
        self._close(conn, cursor)
        return result[1] if result and result and result[1] is not None else 0


if __name__ == '__main__':
    # Example usage:
    db = BattleTowerDatabase()

    # Start a new streak
    streak_id = db.start_streak()
    print(f"Started streak with ID: {streak_id}")

    # Add a few battle records
    pokemon = "Pikachu, Charizard, Blastoise"
    db.add_battle(streak_id, 1, 120, pokemon, True, 'A')  # Battle 1, Won, duration 120s
    db.add_battle(streak_id, 2, 150, pokemon, True, 'A')  # Battle 2, Won, duration 150s
    db.add_battle(streak_id, 3, 100, pokemon, False, 'A')  # Battle 3, Lost, duration 100s

    # End a streak
    db.end_streak(streak_id)
    print(f"Ended streak with ID: {streak_id}")

    # Example usage with a new streak
    streak_id = db.start_streak()
    db.add_battle(streak_id, 1, 200, pokemon, True, 'A')
    db.add_battle(streak_id, 2, 200, pokemon, True, 'A')
    db.add_battle(streak_id, 3, 200, pokemon, True, 'A')
    db.add_battle(streak_id, 4, 200, pokemon, True, 'A')
    db.add_battle(streak_id, 5, 200, pokemon, True, 'A')
    db.add_battle(streak_id, 6, 200, pokemon, True, 'A')
    db.add_battle(streak_id, 7, 200, pokemon, True, 'A')
    db.add_battle(streak_id, 8, 200, pokemon, True, 'A')
    db.add_battle(streak_id, 9, 200, pokemon, True, 'A')
    db.add_battle(streak_id, 10, 200, pokemon, True, 'A')
    db.end_streak(streak_id)

    # Example usage for statistics
    average_streak_length = db.get_average_streak_length()
    print(f"Average streak length: {average_streak_length}")

    streaks_at_least_21 = db.get_streaks_at_least(21)
    print(f"Streaks with at least 21 battles: {streaks_at_least_21}")

    longest_streak = db.get_longest_streak()
    print(f"Longest streak: {longest_streak}")