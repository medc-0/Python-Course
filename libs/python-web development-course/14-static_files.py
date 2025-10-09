"""
14-static_files.py

Static Files
------------
Serve CSS/JS/images from the static folder and reference them in templates.
"""

from __future__ import annotations

from flask import Flask, render_template_string, url_for


app = Flask(__name__)


TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Static Files</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}" />
  </head>
  <body>
    <h1>Static Assets</h1>
    <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo" width="120"/>
    <script src="{{ url_for('static', filename='app.js') }}"></script>
    <p>Open devtools network tab to see static assets served.</p>
  </body>
</html>
"""


@app.route("/")
def page():
    return render_template_string(TEMPLATE)


if __name__ == "__main__":
    app.run(debug=True)


