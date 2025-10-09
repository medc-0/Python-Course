"""
02-flask-routes.py

Routing Deep Dive
-----------------
Explore dynamic segments, type converters, methods, path segments, variable
parts, and route building with url_for. Includes small demo endpoints.
"""

from __future__ import annotations

from typing import Optional

from flask import Flask, jsonify, redirect, request, url_for


app = Flask(__name__)


@app.route("/")
def home() -> str:
    return (
        "<h1>Routes</h1>"
        "<ul>"
        "<li><a href='/about'>/about</a></li>"
        "<li><a href='/user/Ada'>/user/&lt;username&gt;</a></li>"
        "<li><a href='/post/42'>/post/&lt;int:id&gt;</a></li>"
        "<li><a href='/files/path/a/b/c.txt'>/files/path/&lt;path:p&gt;</a></li>"
        "<li><a href='/search?q=flask'>/search?q=flask</a></li>"
        "<li><a href='/submit'>/submit (GET form)</a></li>"
        "</ul>"
    )


@app.route("/about")
def about() -> str:
    return "<p>About Page</p>"


# Dynamic string segment
@app.route("/user/<username>")
def user(username: str) -> str:
    return f"<p>Hello, {username}!</p>"


# Type converter for integers
@app.route("/post/<int:post_id>")
def post(post_id: int) -> str:
    return f"<p>Showing post #{post_id}</p>"


# Path converter for slashes within the segment
@app.route("/files/path/<path:subpath>")
def files_path(subpath: str) -> str:
    return f"<pre>Requested path: {subpath}</pre>"


# Query string access (?q=...)
@app.route("/search")
def search() -> str:
    q: Optional[str] = request.args.get("q")
    return f"<p>Search: {q or 'nothing supplied'}</p>"


# Methods demo: GET shows a simple form, POST handles submission
@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        data = request.form.to_dict() or request.get_json(silent=True) or {}
        return jsonify({"status": "ok", "received": data})
    form = (
        "<form method='post'>"
        "<input name='name' placeholder='Name'/>"
        "<button type='submit'>Send</button>"
        "</form>"
    )
    return form


# Building URLs with url_for
@app.route("/go-home")
def go_home():
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
