import sqlite3
import datetime
import os

DB_PATH = os.path.join('data', 'battle_tower.db')

class BattleTowerDBInterface:
    """
    Use this class to interface w/ the Battle Tower database (instead of calling functions or making API calls directly)
    This class is useful b/c I may want to switch to a Google Cloud VM and I don't want to change the interface
    """

    battle_num: int = None
    current_team: str = None
    current_strategy: int = None

    def on_streak_start(self, team, strategy):
        self.battle_num = 1 # battles start at 1
        self.current_team = team
        self.current_strategy = strategy

    def on_streak_end(self):
        self.battle_num = None
        self.current_team = None
        self.current_strategy = None

    def on_battle_start(self):
        pass

    def on_battle_end(self, won: bool, duration: int):
        """Call this when you win/lose the battle, duration is the # of cycles that the battle took to complete"""
        # TODO: implement checks for the battle num and strategy 'n stuff
        pass

class BattleTowerServerDBInterface(BattleTowerDBInterface):
    streak_id: int = None

    def __init__(self, url='127.0.0.1:500'):
        self.url = url


# this class was created w/ AI assistance b/c I don't know databases very well (and I *really* don't know SQL)
class BattleTowerDatabase:
    def __init__(self, db_name='data/battle_tower.db'):
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
                pokemon_team TEXT,
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
                strategy TEXT,
                win BOOLEAN,
                timestamp TIMESTAMP,
                FOREIGN KEY (streak_id) REFERENCES streaks(id)
            )
        """)

        self._close(conn, cursor)

    def start_streak(self, pokemon_team):
        """Starts a new streak and returns the streak ID."""
        conn, cursor = self._connect()
        start_timestamp = datetime.datetime.now()
        cursor.execute(
            "INSERT INTO streaks (pokemon_team, start_timestamp) VALUES (?, ?)",
            (pokemon_team, start_timestamp)
        )
        streak_id = cursor.lastrowid
        self._close(conn, cursor)
        return streak_id

    def add_battle(self, streak_id, battle_number, battle_duration, win, strategy):
        """Adds a battle record to the database."""
        conn, cursor = self._connect()
        timestamp = datetime.datetime.now()
        cursor.execute(
            "INSERT INTO battles (streak_id, battle_number, battle_duration, strategy, win, timestamp) VALUES (?, ?, ?, ?, ?)",
            (streak_id, battle_number, battle_duration, strategy, win, timestamp)
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
    streak_id = db.start_streak("Pikachu, Charizard, Blastoise")
    print(f"Started streak with ID: {streak_id}")

    # Add a few battle records
    db.add_battle(streak_id, 1, 120, True)  # Battle 1, Won, duration 120s
    db.add_battle(streak_id, 2, 150, True)  # Battle 2, Won, duration 150s
    db.add_battle(streak_id, 3, 100, False)  # Battle 3, Lost, duration 100s

    # End a streak
    db.end_streak(streak_id)
    print(f"Ended streak with ID: {streak_id}")

    # Example usage with a new streak
    streak_id = db.start_streak("Pikachu, Snorlax, Gyarados")
    db.add_battle(streak_id, 1, 200, True)
    db.add_battle(streak_id, 2, 200, True)
    db.add_battle(streak_id, 3, 200, True)
    db.add_battle(streak_id, 4, 200, True)
    db.add_battle(streak_id, 5, 200, True)
    db.add_battle(streak_id, 6, 200, True)
    db.add_battle(streak_id, 7, 200, True)
    db.add_battle(streak_id, 8, 200, True)
    db.add_battle(streak_id, 9, 200, True)
    db.add_battle(streak_id, 10, 200, True)
    db.end_streak(streak_id)

    # Example usage for statistics
    average_streak_length = db.get_average_streak_length()
    print(f"Average streak length: {average_streak_length}")

    streaks_at_least_21 = db.get_streaks_at_least(21)
    print(f"Streaks with at least 21 battles: {streaks_at_least_21}")

    longest_streak = db.get_longest_streak()
    print(f"Longest streak: {longest_streak}")