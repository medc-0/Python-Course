"""
01-flask-intro.py

Flask Introduction
------------------
Flask is a lightweight, flexible Python web framework. This lesson shows you
how to create an app, define routes, return different response types, use
request data, and handle errors.

What you'll learn
-----------------
1) Creating a Flask app and a simple route
2) Dynamic routes and query string parameters
3) Returning HTML and JSON responses
4) Request lifecycle hooks (before/after)
5) Basic error handling and configuration

Docs
----
- Project: https://flask.palletsprojects.com/
- Quickstart: https://flask.palletsprojects.com/en/latest/quickstart/
- API: https://flask.palletsprojects.com/en/latest/api/
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from flask import Flask, jsonify, redirect, request, url_for


def create_app() -> Flask:
    app = Flask(__name__)

    # Example configuration values
    app.config.update(
        ENV="development",
        DEBUG=True,
        SECRET_KEY="dev-secret-key",  # for sessions/CSRF later lessons
        JSON_SORT_KEYS=False,
    )

    # Request lifecycle hooks (for logging/timing)
    @app.before_request
    def _start_timer() -> None:  # Called before each request
        request.start_time = datetime.utcnow()  # type: ignore[attr-defined]

    @app.after_request
    def _log_response(response):  # Called after each request
        start: datetime | None = getattr(request, "start_time", None)  # type: ignore[attr-defined]
        if start:
            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            response.headers["X-Response-Time-ms"] = str(duration_ms)
        response.headers["X-App"] = "Flask Intro Lesson"
        return response

    # Basic route
    @app.route("/")
    def home() -> str:
        return (
            "<h1>Flask Intro</h1>"
            "<p>Welcome! Try <a href='/hello/World'>/hello/World</a>, "
            "<a href='/greet?name=Alice'>/greet?name=Alice</a>, "
            "<a href='/json'>/json</a>, or <a href='/redirect'>/redirect</a>.</p>"
        )

    # Dynamic route segment
    @app.route("/hello/<name>")
    def hello(name: str) -> str:
        return f"<h2>Hello, {name}!</h2>"

    # Query string parameters (?name=...&title=...)
    @app.route("/greet")
    def greet() -> str:
        name = request.args.get("name", "friend")
        title = request.args.get("title")
        label = f"{title} {name}" if title else name
        return f"<p>Greetings, {label}.</p>"

    # JSON response
    @app.route("/json")
    def json_example():
        payload: Dict[str, Any] = {
            "message": "JSON payload",
            "items": [1, 2, 3],
            "meta": {"path": request.path, "args": request.args.to_dict()},
        }
        return jsonify(payload)

    # Redirect and url_for usage
    @app.route("/redirect")
    def go_somewhere():
        return redirect(url_for("docs"))

    @app.route("/docs")
    def docs() -> str:
        return (
            "<h2>Docs</h2>"
            "<ul>"
            "<li><a href='https://flask.palletsprojects.com/'>Flask Documentation</a></li>"
            "<li><a href='/'>Back Home</a></li>"
            "</ul>"
        )

    # Error handling examples
    @app.errorhandler(404)
    def not_found(_):
        return ("<h3>Not Found</h3><p>The requested URL was not found.</p>", 404)

    @app.errorhandler(500)
    def server_error(e):
        return (
            f"<h3>Server Error</h3><pre>{e!s}</pre>",
            500,
        )

    return app


app = create_app()


if __name__ == "__main__":
    # Run the development server with debug reloader
    app.run(debug=True)
