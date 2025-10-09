"""
19-cors_security.py

CORS & Security Headers
----------------------
Add CORS to allow cross-origin requests and set basic security headers.
"""

from __future__ import annotations

from flask import Flask, jsonify
from flask_cors import CORS  # pip install flask-cors


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
        return resp

    @app.get("/api/data")
    def data():
        return jsonify({"message": "ok"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)


