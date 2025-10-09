"""
11-blueprints.py

Blueprints
----------
Structure a modular application using Blueprints. This example defines
two blueprints (auth and blog) and registers them with URL prefixes.
"""

from __future__ import annotations

from flask import Blueprint, Flask, abort, redirect, render_template_string, request, url_for


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "dev"

    # Auth blueprint
    auth = Blueprint("auth", __name__, url_prefix="/auth")

    @auth.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            user = request.form.get("username", "")
            if not user:
                return render_template_string("<p>Missing username</p>"), 400
            return redirect(url_for("blog.index"))
        return render_template_string(
            """
            <h1>Login</h1>
            <form method="post">
              <input name="username" placeholder="username" />
              <button>Login</button>
            </form>
            """
        )

    @auth.route("/logout")
    def logout():
        return redirect(url_for("blog.index"))

    # Blog blueprint
    blog = Blueprint("blog", __name__, url_prefix="/blog")
    posts = [
        {"id": 1, "title": "Hello", "body": "First post"},
        {"id": 2, "title": "Blueprints", "body": "Modular apps"},
    ]

    @blog.route("/")
    def index():
        items = "".join(f"<li><a href='/blog/{p['id']}'>{p['title']}</a></li>" for p in posts)
        return render_template_string(
            f"""
            <h1>Blog</h1>
            <p><a href='{{{{ url_for('auth.login') }}}}'>Login</a></p>
            <ul>{items}</ul>
            """
        )

    @blog.route("/<int:post_id>")
    def detail(post_id: int):
        for p in posts:
            if p["id"] == post_id:
                return render_template_string(
                    """
                    <h1>{{ p.title }}</h1>
                    <p>{{ p.body }}</p>
                    <a href="{{ url_for('blog.index') }}">Back</a>
                    """,
                    p=p,
                )
        abort(404)

    # Register the blueprints
    app.register_blueprint(auth)
    app.register_blueprint(blog)

    @app.route("/")
    def home():
        return render_template_string(
            """
            <h1>Home</h1>
            <ul>
              <li><a href="{{ url_for('blog.index') }}">Blog</a></li>
              <li><a href="{{ url_for('auth.login') }}">Login</a></li>
            </ul>
            """
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)


