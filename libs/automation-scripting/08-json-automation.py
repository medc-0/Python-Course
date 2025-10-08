"""
08-json-automation.py

Automate reading and writing JSON files.

Overview:
---------
JSON (JavaScript Object Notation) is a lightweight format for storing and exchanging data.
It's widely used for APIs, configuration files, and data storage.
Python's json module lets you easily convert between Python objects and JSON.

How JSON works in Python:
-------------------------
- Python dicts, lists, strings, numbers, booleans, and None map directly to JSON objects, arrays, strings, numbers, true/false, and null.
- Use json.dump() to write Python objects to a file as JSON.
- Use json.load() to read JSON data from a file and convert it to Python objects.
- Use json.dumps() and json.loads() for working with JSON strings.

Examples:
---------
"""

import json # Import json pre-built python module

data = {"name": "Bob", "age": 30, "is_student": False, "skills": ["Python", "Automation"]}

# Write to JSON file
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)  # indent for pretty printing

# Read from JSON file
with open("data.json", "r") as f:
    loaded = json.load(f)
    print("Loaded from file:", loaded)

# Convert Python object to JSON string
json_str = json.dumps(data)
print("JSON string:", json_str)

# Convert JSON string to Python object
parsed = json.loads(json_str)
print("Parsed from string:", parsed)

# Clean up
# import os
# os.remove("data.json")

"""
Tips:
-----
- Use json.dump() and json.load() for files.
- Use json.dumps() and json.loads() for strings.
- JSON is great for saving settings, exchanging data, and working with APIs.
- For more info: https://docs.python.org/3/library/json.html
"""
