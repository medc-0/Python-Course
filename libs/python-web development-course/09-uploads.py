"""
09-uploads.py

File Uploads
------------
Upload files with validation and list uploaded files.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, flash, redirect, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED = {"png", "jpg", "jpeg", "gif", "txt", "pdf"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = "dev"


TEMPLATE = """
<!doctype html><html><body>
  {% for m in get_flashed_messages() %}<p style="background:#ffd;">{{ m }}</p>{% endfor %}
  <h1>Uploads</h1>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="file"/>
    <button type="submit">Upload</button>
  </form>
  <h2>Files</h2>
  <ul>
    {% for f in files %}
      <li><a href="{{ url_for('file', name=f) }}">{{ f }}</a></li>
    {% else %}
      <li>No files</li>
    {% endfor %}
  </ul>
</body></html>
"""


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("No file chosen")
        elif not allowed(file.filename):
            flash("Invalid file type")
        else:
            filename = secure_filename(file.filename)
            file.save(UPLOAD_DIR / filename)
            flash("Uploaded")
        return redirect(url_for("upload"))

    files = sorted([p.name for p in UPLOAD_DIR.iterdir() if p.is_file()])
    return render_template_string(TEMPLATE, files=files)


@app.route("/files/<path:name>")
def file(name: str):
    return send_from_directory(UPLOAD_DIR, name)


if __name__ == "__main__":
    app.run(debug=True)


