"""
03-templates.py

Templates (Jinja2)
------------------
Render HTML using Jinja2. Demonstrates template inheritance, variables,
loops/conditionals, and `url_for`. Uses render_template_string to stay
self-contained (no external template files required).

Docs
----
- Templates: https://flask.palletsprojects.com/en/2.3.x/templating/
"""

from __future__ import annotations

from flask import Flask, render_template_string, url_for


app = Flask(__name__)


BASE = """
{% macro nav() -%}
  <nav>
    <a href="{{ url_for('home') }}">Home</a>
    <a href="{{ url_for('users') }}">Users</a>
    <a href="{{ url_for('about') }}">About</a>
  </nav>
{%- endmacro %}

<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{% block title %}Flask Templates{% endblock %}</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 2rem; }
      nav a { margin-right: 1rem; }
      .card { border: 1px solid #ccc; padding: 1rem; border-radius: .5rem; }
    </style>
  </head>
  <body>
    {{ nav() }}
    {% block content %}{% endblock %}
    <footer><small>Powered by Jinja2 • <a href="https://flask.palletsprojects.com/">Docs</a></small></footer>
  </body>
</html>
"""


HOME = """
{% extends base %}
{% block title %}Home • Templates{% endblock %}
{% block content %}
  <h1>Welcome</h1>
  <p>Hello, {{ user.name }}!</p>
  <div class="card">
    <h3>Recent Posts</h3>
    <ul>
      {% for post in posts %}
        <li>
          <a href="{{ url_for('post', post_id=post.id) }}">{{ post.title }}</a>
          {% if post.tags %}<small>— tags: {{ post.tags|join(', ') }}</small>{% endif %}
        </li>
      {% else %}
        <li>No posts yet.</li>
      {% endfor %}
    </ul>
  </div>
{% endblock %}
"""


USERS = """
{% extends base %}
{% block title %}Users • Templates{% endblock %}
{% block content %}
  <h1>Users</h1>
  <ul>
    {% for u in users %}
      <li>{{ u.username }} <small>&lt;{{ u.email }}&gt;</small></li>
    {% endfor %}
  </ul>
{% endblock %}
"""


ABOUT = """
{% extends base %}
{% block title %}About • Templates{% endblock %}
{% block content %}
  <h1>About</h1>
  <p>This page is rendered with Jinja template inheritance.</p>
{% endblock %}
"""


POST = """
{% extends base %}
{% block title %}Post #{{ post.id }} • Templates{% endblock %}
{% block content %}
  <h1>{{ post.title }}</h1>
  <p>{{ post.body }}</p>
  <p><a href="{{ url_for('home') }}">Back</a></p>
{% endblock %}
"""


@app.route("/")
def home():
    user = {"name": "Reader"}
    posts = [
        {"id": 1, "title": "Hello Flask", "tags": ["flask", "jinja"]},
        {"id": 2, "title": "Templates 101", "tags": ["templates"]},
        {"id": 3, "title": "No Tags", "tags": []},
    ]
    return render_template_string(HOME, base=BASE, user=user, posts=posts)


@app.route("/users")
def users():
    people = [
        {"username": "ada", "email": "ada@example.com"},
        {"username": "linus", "email": "linus@example.com"},
        {"username": "guido", "email": "guido@example.com"},
    ]
    return render_template_string(USERS, base=BASE, users=people)


@app.route("/about")
def about():
    return render_template_string(ABOUT, base=BASE)


@app.route("/post/<int:post_id>")
def post(post_id: int):
    post_obj = {"id": post_id, "title": f"Post #{post_id}", "body": "Lorem ipsum..."}
    return render_template_string(POST, base=BASE, post=post_obj)


if __name__ == "__main__":
    app.run(debug=True)


