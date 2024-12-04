from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from utils.Log import Log
from routes.backtest_routes import backtest_bp

app = Flask(__name__)
sock = SocketIO(app)

# 注册Blueprint
app.register_blueprint(backtest_bp)

@app.route('/')
def index():
    return render_template('index.html')

@sock.on('connect')
def handle_connect():
    Log.debug('Client connected')

@sock.on('disconnect')
def handle_disconnect():
    Log.debug('Client disconnected')

if __name__ == '__main__':
    sock.run(app=app, debug=True)
