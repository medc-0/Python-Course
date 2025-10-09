"""
04-forms.py

Forms and Validation
--------------------
Handle GET/POST forms, validate inputs, show flash messages, and redirect.
Self-contained using render_template_string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from flask import Flask, flash, redirect, render_template_string, request, url_for


app = Flask(__name__)
app.secret_key = "dev-secret-key"


TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Forms</title>
    <style>
      body { font-family: system-ui; margin: 2rem; }
      form { display: grid; gap: .6rem; max-width: 320px; }
      input, textarea { padding: .5rem; }
      .flash { background: #ffeaa7; padding: .4rem .6rem; border-radius: .4rem; }
      .error { color: #d63031; }
    </style>
  </head>
  <body>
    <h1>Contact Form</h1>
    {% for m in get_flashed_messages() %}
      <div class="flash">{{ m }}</div>
    {% endfor %}
    <form method="post" action="{{ url_for('contact') }}">
      <input name="name" placeholder="Name" value="{{ form.name }}"/>
      {% if errors.name %}<div class="error">{{ errors.name }}</div>{% endif %}
      <input name="email" placeholder="Email" value="{{ form.email }}"/>
      {% if errors.email %}<div class="error">{{ errors.email }}</div>{% endif %}
      <textarea name="message" placeholder="Message">{{ form.message }}</textarea>
      {% if errors.message %}<div class="error">{{ errors.message }}</div>{% endif %}
      <button type="submit">Send</button>
    </form>
  </body>
  </html>
"""


@dataclass
class ContactForm:
    name: str = ""
    email: str = ""
    message: str = ""


def validate(form: ContactForm) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    if not form.name.strip():
        errors["name"] = "Name is required"
    if "@" not in form.email:
        errors["email"] = "Valid email required"
    if len(form.message.strip()) < 5:
        errors["message"] = "Message must be at least 5 characters"
    return errors


@app.route("/contact", methods=["GET", "POST"])
def contact():
    form = ContactForm(
        name=request.form.get("name", ""),
        email=request.form.get("email", ""),
        message=request.form.get("message", ""),
    )

    if request.method == "POST":
        errors = validate(form)
        if not errors:
            # Pretend to send email or store in DB
            flash("Message sent successfully!")
            return redirect(url_for("contact"))
        return render_template_string(TEMPLATE, form=form, errors=errors)

    # GET
    return render_template_string(TEMPLATE, form=form, errors={})


if __name__ == "__main__":
    app.run(debug=True)


