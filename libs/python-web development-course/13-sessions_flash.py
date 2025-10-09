"""
13-sessions_flash.py

Sessions and Flash Messages
---------------------------
Store small bits of data in signed cookies (session) and display one-time
notifications (flash messages).
"""

from __future__ import annotations

from flask import Flask, flash, redirect, render_template_string, request, session, url_for


app = Flask(__name__)
app.secret_key = "dev"


TEMPLATE = """
<!doctype html><html><body>
  {% for m in get_flashed_messages() %}<p style="background:#ffd;">{{ m }}</p>{% endfor %}
  <h1>Preferences</h1>
  <form method="post">
    <label>Theme: <input name="theme" value="{{ session.get('theme','light') }}"/></label>
    <button>Save</button>
  </form>
  <p>Current theme: <b>{{ session.get('theme','light') }}</b></p>
  <p><a href="{{ url_for('reset') }}">Reset</a></p>
</body></html>
"""


@app.route("/", methods=["GET", "POST"])
def prefs():
    if request.method == "POST":
        session["theme"] = request.form.get("theme", "light")
        flash("Preferences saved")
        return redirect(url_for("prefs"))
    return render_template_string(TEMPLATE)


@app.route("/reset")
def reset():
    session.clear()
    flash("Preferences cleared")
    return redirect(url_for("prefs"))


if __name__ == "__main__":
    app.run(debug=True)


