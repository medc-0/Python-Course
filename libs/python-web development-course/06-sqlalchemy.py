"""
06-sqlalchemy.py

SQLAlchemy ORM
--------------
Define models, create tables, and perform CRUD using SQLAlchemy.
"""

from __future__ import annotations

from flask import Flask, jsonify, redirect, render_template_string, request, url_for
from flask_sqlalchemy import SQLAlchemy  # pip install flask_sqlalchemy


app = Flask(__name__)
app.config.update(
    SQLALCHEMY_DATABASE_URI="sqlite:///orm_app.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SECRET_KEY="dev",
)
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    posts = db.relationship("Post", backref="author", lazy=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.username}>"


class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Post {self.title}>"


@app.before_first_request
def _init():
    db.create_all()


LIST = """
<!doctype html><html><body>
  <h1>Users</h1>
  <form method="post" action="{{ url_for('add_user') }}">
    <input name="username" placeholder="username"/>
    <input name="email" placeholder="email"/>
    <button type="submit">Add</button>
  </form>
  <ul>
    {% for u in users %}
      <li>{{ u.username }} &lt;{{ u.email }}&gt; ({{ u.posts|length }} posts)</li>
    {% endfor %}
  </ul>
  <h2>New Post</h2>
  <form method="post" action="{{ url_for('add_post') }}">
    <input name="username" placeholder="author username"/>
    <input name="title" placeholder="title"/>
    <input name="content" placeholder="content"/>
    <button type="submit">Publish</button>
  </form>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(LIST, users=User.query.all())


@app.route("/add_user", methods=["POST"])
def add_user():
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    if username and email:
        db.session.add(User(username=username, email=email))
        db.session.commit()
    return redirect(url_for("index"))


@app.route("/add_post", methods=["POST"])
def add_post():
    username = request.form.get("username", "").strip()
    title = request.form.get("title", "").strip()
    content = request.form.get("content", "").strip()
    if username and title and content:
        user = User.query.filter_by(username=username).first()
        if user:
            db.session.add(Post(title=title, content=content, author=user))
            db.session.commit()
    return redirect(url_for("index"))


# JSON endpoints
@app.route("/api/users")
def api_users():
    users = User.query.all()
    return jsonify([
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "posts": [p.id for p in u.posts],
        }
        for u in users
    ])


if __name__ == "__main__":
    app.run(debug=True)


