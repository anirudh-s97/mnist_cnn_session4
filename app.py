from flask import Flask, render_template, jsonify
import json
from pathlib import Path
import time

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data/training')
def get_training_data():
    try:
        with open('logs/training_metrics.json', 'r') as f:
            return jsonify([json.loads(line) for line in f])
    except FileNotFoundError:
        return jsonify([])

@app.route('/data/test')
def get_test_data():
    try:
        with open('logs/test_metrics.json', 'r') as f:
            return jsonify([json.loads(line) for line in f])
    except FileNotFoundError:
        return jsonify([])

@app.route('/data/results')
def get_test_results():
    try:
        with open('logs/test_results.json', 'r') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True, port=5000)