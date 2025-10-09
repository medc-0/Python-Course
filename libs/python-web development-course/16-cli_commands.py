"""
16-cli_commands.py

Custom CLI Commands
-------------------
Add Flask CLI commands for tasks like creating an admin user or seeding data.
"""

from __future__ import annotations

import click
from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def hi():
        return {"message": "ok"}

    @app.cli.command("hello")
    @click.option("--name", default="world")
    def hello(name: str):
        """Print a friendly greeting."""
        click.echo(f"Hello, {name}!")

    @app.cli.command("seed")
    def seed():
        """Seed database with initial data (demo)."""
        click.echo("Seeded sample data")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)


