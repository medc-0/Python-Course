"""
26-jsonHandling.py

Beginner's guide to JSON handling in Python.

Overview:
---------
JSON (JavaScript Object Notation) is a lightweight data interchange format. Python provides built-in support for working with JSON data through the `json` module.

What is JSON?
-------------
JSON is a text format for storing and transporting data. It's easy for humans to read and write, and easy for machines to parse and generate.

Key Concepts:
--------------
- JSON is a text format
- Data is represented as key-value pairs
- Supports strings, numbers, booleans, arrays, and objects
- Widely used for APIs and configuration files
- Human-readable and machine-parseable

Examples:
---------
"""

import json

# 1. Basic JSON data
# Python dictionary
data = {
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "is_student": False,
    "hobbies": ["reading", "swimming", "coding"],
    "address": {
        "street": "123 Main St",
        "zipcode": "10001"
    }
}

print("Original Python data:")
print(data)

# 2. Converting Python to JSON
# Serialize Python object to JSON string
json_string = json.dumps(data)
print("\nJSON string:")
print(json_string)

# Pretty print JSON
json_pretty = json.dumps(data, indent=2)
print("\nPretty JSON:")
print(json_pretty)

# 3. Converting JSON to Python
# Parse JSON string to Python object
parsed_data = json.loads(json_string)
print("\nParsed data:")
print(parsed_data)
print(f"Type: {type(parsed_data)}")

# 4. Working with JSON files
# Writing to JSON file
with open("data.json", "w") as file:
    json.dump(data, file, indent=2)

print("\nData written to data.json")

# Reading from JSON file
with open("data.json", "r") as file:
    loaded_data = json.load(file)

print("Data loaded from file:")
print(loaded_data)

# 5. Handling different data types
# Numbers
numbers_data = {
    "integer": 42,
    "float": 3.14159,
    "negative": -10
}

json_numbers = json.dumps(numbers_data)
print(f"\nNumbers JSON: {json_numbers}")

# Booleans and None
boolean_data = {
    "true_value": True,
    "false_value": False,
    "null_value": None
}

json_booleans = json.dumps(boolean_data)
print(f"Booleans JSON: {json_booleans}")

# 6. Working with arrays
# List of dictionaries
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]

json_students = json.dumps(students, indent=2)
print(f"\nStudents JSON:\n{json_students}")

# 7. Custom serialization
# Custom encoder for non-standard types
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def to_dict(self):
        return {"name": self.name, "age": self.age}

person = Person("Jane", 25)

# Convert to dictionary first
person_dict = person.to_dict()
json_person = json.dumps(person_dict)
print(f"\nPerson JSON: {json_person}")

# 8. Handling JSON errors
# Invalid JSON
invalid_json = '{"name": "John", "age": 30,}'  # Trailing comma

try:
    parsed = json.loads(invalid_json)
    print(f"Parsed: {parsed}")
except json.JSONDecodeError as e:
    print(f"JSON Error: {e}")

# 9. JSON with different encodings
# UTF-8 encoding
unicode_data = {
    "name": "José",
    "city": "São Paulo",
    "message": "Hello, 世界!"
}

json_unicode = json.dumps(unicode_data, ensure_ascii=False)
print(f"\nUnicode JSON: {json_unicode}")

# 10. Working with nested JSON
# Complex nested structure
company_data = {
    "company": "Tech Corp",
    "employees": [
        {
            "id": 1,
            "name": "John",
            "department": "Engineering",
            "skills": ["Python", "JavaScript", "SQL"],
            "contact": {
                "email": "john@techcorp.com",
                "phone": "+1-555-0123"
            }
        },
        {
            "id": 2,
            "name": "Jane",
            "department": "Marketing",
            "skills": ["SEO", "Content Writing", "Analytics"],
            "contact": {
                "email": "jane@techcorp.com",
                "phone": "+1-555-0124"
            }
        }
    ],
    "departments": {
        "Engineering": {"head": "John", "size": 5},
        "Marketing": {"head": "Jane", "size": 3}
    }
}

json_company = json.dumps(company_data, indent=2)
print(f"\nCompany JSON:\n{json_company}")

# 11. Filtering and transforming JSON
# Load and filter data
with open("data.json", "r") as file:
    data = json.load(file)

# Filter students with high grades
high_grades = [student for student in data.get("students", []) if student.get("grade", 0) >= 80]
print(f"\nHigh grades: {high_grades}")

# 12. JSON validation
def validate_json(json_string):
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

# Test validation
valid_json = '{"name": "John", "age": 30}'
invalid_json = '{"name": "John", "age": 30,}'

print(f"\nValid JSON: {validate_json(valid_json)}")
print(f"Invalid JSON: {validate_json(invalid_json)}")

# 13. Working with JSON arrays
# Array of objects
products = [
    {"id": 1, "name": "Laptop", "price": 999.99, "in_stock": True},
    {"id": 2, "name": "Mouse", "price": 29.99, "in_stock": False},
    {"id": 3, "name": "Keyboard", "price": 79.99, "in_stock": True}
]

# Save products to JSON
with open("products.json", "w") as file:
    json.dump(products, file, indent=2)

# Load and process products
with open("products.json", "r") as file:
    loaded_products = json.load(file)

# Find products in stock
in_stock = [product for product in loaded_products if product["in_stock"]]
print(f"\nProducts in stock: {len(in_stock)}")

# 14. JSON with custom separators
data = {"name": "John", "age": 30, "city": "New York"}

# Custom separators
json_custom = json.dumps(data, separators=(",", ":"))
print(f"\nCustom separators: {json_custom}")

# 15. Working with JSON from APIs
# Simulate API response
api_response = {
    "status": "success",
    "data": {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ],
        "pagination": {
            "page": 1,
            "per_page": 10,
            "total": 2
        }
    },
    "message": "Users retrieved successfully"
}

# Process API response
if api_response["status"] == "success":
    users = api_response["data"]["users"]
    print(f"\nAPI Response - Users: {len(users)}")
    for user in users:
        print(f"  {user['name']} ({user['email']})")

"""
JSON Data Types:
----------------
Python -> JSON
- dict -> object
- list -> array
- str -> string
- int -> number
- float -> number
- True -> true
- False -> false
- None -> null

JSON -> Python
- object -> dict
- array -> list
- string -> str
- number -> int/float
- true -> True
- false -> False
- null -> None

Common JSON Operations:
----------------------
- json.dumps(): Convert Python to JSON string
- json.loads(): Convert JSON string to Python
- json.dump(): Write Python to JSON file
- json.load(): Read JSON file to Python
- json.dumps(obj, indent=2): Pretty print
- json.dumps(obj, ensure_ascii=False): Unicode support

When to Use JSON:
----------------
- API communication
- Configuration files
- Data storage
- Interchange between systems
- Web applications

When NOT to Use JSON:
--------------------
- Complex data structures
- Binary data
- Large datasets (use databases)
- When performance is critical
- When you need schema validation

Tips:
-----
- Use indent parameter for readable JSON
- Handle JSONDecodeError exceptions
- Use ensure_ascii=False for Unicode
- Validate JSON before processing
- Use custom encoders for complex objects
- Consider using json-schema for validation

"""
