"""
12-configuration.py

Configuration Patterns
----------------------
Use config classes and environment variables. Show per-env settings and a
factory that loads the right configuration.
"""

from __future__ import annotations

import os
from flask import Flask


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev")
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///config.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


class ProductionConfig(Config):
    pass


CONFIGS = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def create_app(env: str | None = None) -> Flask:
    app = Flask(__name__)
    env_name = env or os.getenv("FLASK_ENV", "development")
    cfg = CONFIGS.get(env_name, DevelopmentConfig)
    app.config.from_object(cfg)

    @app.route("/")
    def info():
        return {
            "env": env_name,
            "debug": app.debug,
            "testing": app.testing,
            "database": app.config["SQLALCHEMY_DATABASE_URI"],
        }

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)


