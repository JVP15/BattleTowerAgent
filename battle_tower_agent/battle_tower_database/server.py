from flask import Flask, request, jsonify
from battle_tower_agent.battle_tower_database.database import BattleTowerDatabase, DB_PATH

def create_app(db_path=DB_PATH):
    app = Flask(__name__)
    app.config['DB_PATH'] = db_path

    with app.app_context():
        app.db = BattleTowerDatabase(app.config['DB_PATH'])  # Initialize DB using app's db_path

    # Error handler for common issues
    @app.errorhandler(400)  # Bad Request
    def bad_request(error):
        return jsonify({'error': 'Bad request', 'message': str(error)}), 400

    @app.errorhandler(404)  # Not Found
    def not_found(error):
        return jsonify({'error': 'Not found', 'message': str(error)}), 404

    @app.errorhandler(500)  # Internal Server Error
    def internal_error(error):
        return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

    # Route to start a new streak
    @app.route('/streaks', methods=['POST'])
    def start_streak():
        db = app.db # Access database from app context
        streak_id = db.start_streak()
        return jsonify({'streak_id': streak_id}), 201

    # Route to add a new battle
    @app.route('/battle', methods=['POST'])
    def add_battle():
        db = app.db # Access database from app context
        data = request.get_json()
        required_fields = ['streak_id', 'battle_number', 'battle_duration', 'win', 'pokemon_team', 'strategy']
        if not data or any(field not in data for field in required_fields):
            return bad_request('Missing required fields in request data')

        try:
            streak_id = int(data['streak_id'])
            battle_number = int(data['battle_number'])
            battle_duration = int(data['battle_duration'])
            win = bool(data['win'])
            pokemon_team = data['pokemon_team']
            strategy = data['strategy']
        except ValueError:
            return bad_request('Invalid data types for required fields')

        db.add_battle(streak_id, battle_number, battle_duration, pokemon_team, strategy, win)
        return jsonify({'message': 'Battle added successfully'}), 201

    # Route to end a streak
    @app.route('/streaks/<int:streak_id>/end', methods=['PUT'])
    def end_streak(streak_id):
        db = app.db # Access database from app context
        try:
            db.end_streak(streak_id)
            return jsonify({'message': 'Streak ended successfully'}), 200
        except Exception as e:
            return internal_error(e)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

