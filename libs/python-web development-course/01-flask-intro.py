"""
01-flask-intro.py

Introduction to Flask web development.

Overview:
---------
Flask is a lightweight web framework for Python.

Example: Basic web server
-------------------------
"""

from flask import Flask # pip install flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

if __name__ == "__main__":
    app.run(debug=True)
