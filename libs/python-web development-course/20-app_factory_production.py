"""
20-app_factory_production.py

Production-ready App Factory
----------------------------
Combine config, blueprints, logging, and CLI in a single app factory.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask


def create_app(config_object: str | None = None) -> Flask:
    app = Flask(__name__)
    if config_object:
        app.config.from_object(config_object)
    else:
        app.config.update(SECRET_KEY="prod-secret", JSON_SORT_KEYS=False)

    @app.route("/")
    def health():
        return {"status": "ok"}

    # Logging
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    handler = RotatingFileHandler(logs / "app.log", maxBytes=1_000_000, backupCount=5)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("App created")

    # Example CLI command
    @app.cli.command("status")
    def status():  # pragma: no cover
        click = __import__("click")
        click.echo("App is healthy")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)


