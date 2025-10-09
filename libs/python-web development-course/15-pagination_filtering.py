"""
15-pagination_filtering.py

Pagination & Filtering
----------------------
Implement simple pagination and filtering parameters on a list view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from flask import Flask, render_template_string, request, url_for


app = Flask(__name__)


@dataclass
class Product:
    id: int
    name: str
    category: str


PRODUCTS: List[Product] = [
    Product(i, f"Item {i}", "A" if i % 2 == 0 else "B") for i in range(1, 101)
]


TEMPLATE = """
<!doctype html><html><body>
  <h1>Products</h1>
  <form>
    <label>Category:
      <select name="cat">
        <option value="">All</option>
        <option value="A" {% if cat=='A' %}selected{% endif %}>A</option>
        <option value="B" {% if cat=='B' %}selected{% endif %}>B</option>
      </select>
    </label>
    <button>Filter</button>
  </form>
  <ul>
    {% for p in page_items %}<li>{{ p.name }} ({{ p.category }})</li>{% endfor %}
  </ul>
  <p>
    {% if page>1 %}<a href="{{ url_for('products', page=page-1, cat=cat) }}">Prev</a>{% endif %}
    Page {{ page }} / {{ pages }}
    {% if page<pages %}<a href="{{ url_for('products', page=page+1, cat=cat) }}">Next</a>{% endif %}
  </p>
</body></html>
"""


@app.route("/")
def products():
    page = max(1, int(request.args.get("page", 1)))
    per_page = 10
    cat = request.args.get("cat", "")

    items = PRODUCTS
    if cat in ("A", "B"):
        items = [p for p in items if p.category == cat]

    total = len(items)
    pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    page_items = items[start : start + per_page]
    return render_template_string(
        TEMPLATE, page_items=page_items, page=page, pages=pages, cat=cat
    )


if __name__ == "__main__":
    app.run(debug=True)


