"""
08-rest_api.py

REST API with Flask
-------------------
Build simple JSON endpoints with proper status codes and error handling.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from flask import Flask, jsonify, request


app = Flask(__name__)


@dataclass
class Item:
    id: int
    name: str
    price: float


items: Dict[int, Item] = {
    1: Item(id=1, name="Keyboard", price=49.9),
    2: Item(id=2, name="Mouse", price=19.5),
}
next_id = 3


@app.get("/api/items")
def list_items():
    return jsonify([asdict(i) for i in items.values()])


@app.post("/api/items")
def create_item():
    global next_id
    data = request.get_json(force=True, silent=True) or {}
    name = str(data.get("name", "")).strip()
    price = float(data.get("price", 0.0))
    if not name or price <= 0:
        return jsonify({"error": "name and positive price required"}), 400
    item = Item(id=next_id, name=name, price=price)
    items[item.id] = item
    next_id += 1
    return jsonify(asdict(item)), 201


@app.get("/api/items/<int:item_id>")
def get_item(item_id: int):
    item = items.get(item_id)
    if not item:
        return jsonify({"error": "not found"}), 404
    return jsonify(asdict(item))


@app.put("/api/items/<int:item_id>")
def update_item(item_id: int):
    item = items.get(item_id)
    if not item:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(force=True, silent=True) or {}
    if "name" in data:
        item.name = str(data["name"]).strip() or item.name
    if "price" in data:
        try:
            price = float(data["price"])
            if price > 0:
                item.price = price
        except Exception:
            pass
    return jsonify(asdict(item))


@app.delete("/api/items/<int:item_id>")
def delete_item(item_id: int):
    if item_id in items:
        del items[item_id]
        return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)


