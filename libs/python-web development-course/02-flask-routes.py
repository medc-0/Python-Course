"""
02-flask-routes.py

Multiple routes in Flask.
"""

from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Home Page"

@app.route("/about")
def about():
    return "About Page"

@app.route("/user/<username>")
def user(username):
    return f"Hello, {username}!"

if __name__ == "__main__":
    app.run(debug=True)
