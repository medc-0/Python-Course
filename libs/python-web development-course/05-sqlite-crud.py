"""
05-sqlite-crud.py

SQLite CRUD (no ORM)
--------------------
Minimal persistence using sqlite3 and Flask app context. Demonstrates creating
tables, inserting, listing, and deleting rows.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable

from flask import Flask, g, jsonify, redirect, render_template_string, request, url_for


DB_PATH = Path("flask_app.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS todos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  done INTEGER NOT NULL DEFAULT 0
);
"""


LIST_TEMPLATE = """
<!doctype html><html><body>
  <h1>Todos</h1>
  <form method="post" action="{{ url_for('add_todo') }}">
    <input name="title" placeholder="New todo"/>
    <button type="submit">Add</button>
  </form>
  <ul>
  {% for t in todos %}
    <li>
      {{ t['title'] }}
      <a href="{{ url_for('delete_todo', todo_id=t['id']) }}">[delete]</a>
    </li>
  {% else %}
    <li>No items</li>
  {% endfor %}
  </ul>
</body></html>
"""


def get_db() -> sqlite3.Connection:
    db = getattr(g, "_db", None)
    if db is None:
        db = g._db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.executescript(SCHEMA)
        db.commit()


def query(sql: str, args: Iterable | None = None):
    cur = get_db().execute(sql, args or [])
    rows = cur.fetchall()
    cur.close()
    return rows


def execute(sql: str, args: Iterable | None = None) -> None:
    db = get_db()
    db.execute(sql, args or [])
    db.commit()


app = Flask(__name__)


@app.before_first_request
def _ensure_db():
    init_db()


@app.teardown_appcontext
def _close_db(_):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()


@app.route("/")
def index():
    todos = query("SELECT id, title, done FROM todos ORDER BY id DESC")
    return render_template_string(LIST_TEMPLATE, todos=todos)


@app.route("/add", methods=["POST"])
def add_todo():
    title = request.form.get("title", "").strip()
    if title:
        execute("INSERT INTO todos (title, done) VALUES (?, 0)", (title,))
    return redirect(url_for("index"))


@app.route("/delete/<int:todo_id>")
def delete_todo(todo_id: int):
    execute("DELETE FROM todos WHERE id=?", (todo_id,))
    return redirect(url_for("index"))


# JSON endpoints (for API-minded learners)
@app.route("/api/todos")
def api_list():
    rows = query("SELECT id, title, done FROM todos ORDER BY id DESC")
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    app.run(debug=True)


