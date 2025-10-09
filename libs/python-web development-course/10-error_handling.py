"""
10-error_handling.py

Error Handling & Logging
-----------------------
Custom error pages and simple logging.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, jsonify, render_template_string


app = Flask(__name__)


ERROR_404 = """
<!doctype html><html><body>
  <h1>404 Not Found</h1>
  <p>The requested resource was not found.</p>
</body></html>
"""


ERROR_500 = """
<!doctype html><html><body>
  <h1>500 Internal Server Error</h1>
  <p>Something went wrong. Try again later.</p>
</body></html>
"""


@app.errorhandler(404)
def not_found(_):
    return render_template_string(ERROR_404), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.exception("Unhandled exception: %s", e)
    return render_template_string(ERROR_500), 500


@app.route("/crash")
def crash():
    # Force an exception to test 500 page and logging
    raise RuntimeError("Boom!")


@app.route("/api")
def api():
    return jsonify({"message": "ok"})


def setup_logging() -> None:
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    handler = RotatingFileHandler(logs / "app.log", maxBytes=500_000, backupCount=3)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("App startup")


if __name__ == "__main__":
    setup_logging()
    app.run(debug=True)


