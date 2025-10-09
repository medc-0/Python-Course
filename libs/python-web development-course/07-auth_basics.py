"""
07-auth_basics.py

Authentication Basics
---------------------
Register/login/logout with hashed passwords and session storage.
"""

from __future__ import annotations

from flask import Flask, flash, redirect, render_template_string, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)
app.config.update(
    SECRET_KEY="dev-secret",
    SQLALCHEMY_DATABASE_URI="sqlite:///auth.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


@app.before_first_request
def _init_db():
    db.create_all()


BASE = """
<!doctype html><html><body>
  {% for m in get_flashed_messages() %}<p style="background:#ffd;">{{ m }}</p>{% endfor %}
  {% block content %}{% endblock %}
</body></html>
"""


HOME = """
{% extends base %}
{% block content %}
  <h1>Home</h1>
  {% if session.get('uid') %}
    <p>Logged in as <b>{{ session.username }}</b>. <a href='{{ url_for('logout') }}'>Logout</a></p>
  {% else %}
    <p><a href='{{ url_for('login') }}'>Login</a> | <a href='{{ url_for('register') }}'>Register</a></p>
  {% endif %}
{% endblock %}
"""


FORM = """
{% extends base %}
{% block content %}
  <h1>{{ title }}</h1>
  <form method="post">
    <input name="username" placeholder="username" value="{{ username or '' }}"/>
    <input name="password" type="password" placeholder="password"/>
    <button type="submit">{{ title }}</button>
  </form>
{% endblock %}
"""


def _login_user(user: User) -> None:
    session["uid"] = user.id
    session["username"] = user.username


@app.route("/")
def home():
    return render_template_string(HOME, base=BASE)


@app.route("/register", methods=["GET", "POST"])
def register():
    username = request.form.get("username", "")
    if request.method == "POST":
        password = request.form.get("password", "")
        if not username or not password:
            flash("Both fields required")
        elif User.query.filter_by(username=username).first():
            flash("Username already taken")
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            _login_user(user)
            flash("Registered and logged in")
            return redirect(url_for("home"))
    return render_template_string(FORM, base=BASE, title="Register", username=username)


@app.route("/login", methods=["GET", "POST"])
def login():
    username = request.form.get("username", "")
    if request.method == "POST":
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            _login_user(user)
            flash("Welcome back!")
            return redirect(url_for("home"))
        flash("Invalid credentials")
    return render_template_string(FORM, base=BASE, title="Login", username=username)


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)


