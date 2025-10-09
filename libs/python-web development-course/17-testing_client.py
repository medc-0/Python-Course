"""
17-testing_client.py

Testing with Flask Test Client
------------------------------
Demonstrates how to write simple tests for routes using app.test_client().
Run this file directly to execute the tests.
"""

from __future__ import annotations

import json
from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def root():
        return {"message": "ok"}

    @app.post("/echo")
    def echo():
        from flask import request

        data = request.get_json() or {}
        return {"you_sent": data}

    return app


def run_tests() -> None:
    app = create_app()
    client = app.test_client()

    # GET test
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.get_json()["message"] == "ok"

    # POST test
    payload = {"x": 1, "y": 2}
    resp = client.post("/echo", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    assert resp.get_json()["you_sent"] == payload

    print("All tests passed!")


if __name__ == "__main__":
    run_tests()


