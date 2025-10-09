"""
18-wsgi_middleware.py

WSGI Middleware
---------------
Wrap a Flask app with a tiny WSGI middleware that adds a header and logs path.
"""

from __future__ import annotations

from typing import Callable, Iterable

from flask import Flask


def header_middleware(app: Callable) -> Callable:
    def wrapper(environ, start_response):
        path = environ.get("PATH_INFO", "-")
        # Wrap start_response to inject a header
        def custom_start_response(status, headers, exc_info=None):
            headers = list(headers) + [("X-Path", path)]
            return start_response(status, headers, exc_info)

        return app(environ, custom_start_response)

    return wrapper


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def root():
        return {"message": "ok"}

    return app


app = create_app()
app.wsgi_app = header_middleware(app.wsgi_app)  # type: ignore[assignment]


if __name__ == "__main__":
    app.run(debug=True)


