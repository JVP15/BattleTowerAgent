import requests

import logging
logger = logging.getLogger('BattleTowerDatabase')

class BattleTowerDBInterface:
    """
    Use this class to interface w/ the Battle Tower database (instead of calling functions or making API calls directly)
    This class is useful b/c I may want to switch to a Google Cloud VM and I don't want to change the interface
    """

    battle_num: int | None = None
    current_team: str | None = None
    current_strategy: int | None = None

    def on_streak_start(self, team, strategy):
        self.battle_num = 1 # battles start at 1
        self.current_team = team
        self.current_strategy = strategy

    def on_streak_end(self):
        self.battle_num = None
        self.current_team = None
        self.current_strategy = None

    def on_battle_end(self, won: bool, duration: int):
        """Call this when you win/lose the battle, duration is the # of cycles that the battle took to complete"""
        # TODO: implement checks for the battle num and strategy 'n stuff

        self.battle_num += 1

class BattleTowerServerDBInterface(BattleTowerDBInterface):
    HEADERS: dict = {'Content-Type': 'application/json'}
    streak_id: int = None

    def __init__(self, base_url='http://127.0.0.1:5000'):
        self.base_url = base_url

    def on_streak_start(self, team, strategy):
        super().on_streak_start(team, strategy)
        url = f"{self.base_url}/streaks"
        try:
            response = requests.post(url)
            response.raise_for_status()
            self.streak_id = response.json()['streak_id']
        except requests.exceptions.RequestException as e:
            logger.error(f"Error starting streak: {e}")

    def on_battle_end(self, won: bool, duration: int):
        url = f"{self.base_url}/battle"

        data = {
            'streak_id': self.streak_id,
            'battle_number': self.battle_num,
            'battle_duration': duration,
            'win': won,
            'pokemon_team': self.current_team,
            'strategy': self.current_strategy
        }

        try:
            response = requests.post(url, json=data, headers=self.HEADERS)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error adding battle: {e}")

        super().on_battle_end(won, duration)

    def on_streak_end(self):
        url = f"{self.base_url}/streaks/{self.streak_id}/end"

        try:
            response = requests.put(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error ending streak: {e}")

        self.streak_id = None