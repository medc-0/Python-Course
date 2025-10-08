"""
27-csvHandling.py

Beginner's guide to CSV handling in Python.

Overview:
---------
CSV (Comma-Separated Values) is a simple file format for storing tabular data. Python provides built-in support for working with CSV files through the `csv` module.

What is CSV?
------------
CSV is a text format where each line represents a row, and values are separated by commas. It's widely used for data exchange between applications.

Key Concepts:
--------------
- CSV files contain rows and columns
- Values are separated by delimiters (usually commas)
- First row often contains headers
- Text values may be quoted
- Simple and widely supported format

Examples:
---------
"""

import csv
import os

# 1. Basic CSV writing
# Create sample data
data = [
    ["Name", "Age", "City", "Email"],
    ["John Doe", 30, "New York", "john@example.com"],
    ["Jane Smith", 25, "Los Angeles", "jane@example.com"],
    ["Bob Johnson", 35, "Chicago", "bob@example.com"]
]

# Write to CSV file
with open("people.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV file created: people.csv")

# 2. Basic CSV reading
# Read the CSV file
with open("people.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# 3. Reading with headers
# Read CSV with headers
with open("people.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(f"Name: {row['Name']}, Age: {row['Age']}, City: {row['City']}")

# 4. Writing with headers
# Create data as list of dictionaries
people_data = [
    {"Name": "Alice Brown", "Age": 28, "City": "Boston", "Email": "alice@example.com"},
    {"Name": "Charlie Wilson", "Age": 32, "City": "Seattle", "Email": "charlie@example.com"},
    {"Name": "Diana Davis", "Age": 27, "City": "Miami", "Email": "diana@example.com"}
]

# Write with headers
with open("people_dict.csv", "w", newline="") as file:
    fieldnames = ["Name", "Age", "City", "Email"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(people_data)

print("\nCSV file with headers created: people_dict.csv")

# 5. Reading with custom delimiter
# Create CSV with semicolon delimiter
with open("semicolon.csv", "w", newline="") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerows(data)

# Read with semicolon delimiter
with open("semicolon.csv", "r") as file:
    reader = csv.reader(file, delimiter=";")
    for row in reader:
        print(f"Semicolon CSV: {row}")

# 6. Handling quoted fields
# Data with commas in fields
quoted_data = [
    ["Product", "Description", "Price"],
    ["Laptop", "High-performance laptop, 16GB RAM", "$999.99"],
    ["Mouse", "Wireless mouse, ergonomic design", "$29.99"],
    ["Keyboard", "Mechanical keyboard, RGB lighting", "$79.99"]
]

# Write with quoting
with open("products.csv", "w", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerows(quoted_data)

print("\nQuoted CSV created: products.csv")

# 7. Reading and filtering data
# Read and filter people over 30
with open("people.csv", "r") as file:
    reader = csv.DictReader(file)
    adults = [row for row in reader if int(row["Age"]) >= 30]
    
print("\nPeople 30 and older:")
for person in adults:
    print(f"  {person['Name']} ({person['Age']})")

# 8. Working with different encodings
# Create CSV with UTF-8 encoding
unicode_data = [
    ["Name", "City", "Country"],
    ["José", "São Paulo", "Brazil"],
    ["François", "Paris", "France"],
    ["李小明", "Beijing", "China"]
]

with open("unicode.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(unicode_data)

# Read with UTF-8 encoding
with open("unicode.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        print(f"Unicode CSV: {row}")

# 9. CSV with different quote characters
# Data with quotes in fields
quote_data = [
    ["Text", "Description"],
    ["Hello", "He said 'Hello' to me"],
    ["World", "She said \"World\" to him"],
    ["Python", "It's a 'snake' language"]
]

# Write with custom quote character
with open("quotes.csv", "w", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL, quotechar="'")
    writer.writerows(quote_data)

print("\nCSV with custom quotes created: quotes.csv")

# 10. Appending to CSV file
# Append new data to existing file
new_person = ["Eve Wilson", 29, "Portland", "eve@example.com"]

with open("people.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(new_person)

print("New person appended to people.csv")

# 11. Working with large CSV files
# Create a large CSV file
def create_large_csv(filename, num_rows):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Name", "Value"])
        for i in range(num_rows):
            writer.writerow([i, f"Item_{i}", i * 10])

# Create large file
create_large_csv("large_data.csv", 1000)
print("Large CSV file created: large_data.csv")

# Read large file efficiently
def read_large_csv(filename):
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        count = 0
        for row in reader:
            count += 1
            if count <= 5:  # Show first 5 rows
                print(f"Row {count}: {row}")
        print(f"Total rows processed: {count}")

read_large_csv("large_data.csv")

# 12. CSV validation
def validate_csv(filename):
    try:
        with open(filename, "r") as file:
            reader = csv.reader(file)
            headers = next(reader)
            print(f"Headers: {headers}")
            
            row_count = 0
            for row in reader:
                row_count += 1
                if len(row) != len(headers):
                    print(f"Row {row_count}: Invalid number of columns")
                    return False
            
            print(f"CSV is valid with {row_count} data rows")
            return True
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

validate_csv("people.csv")

# 13. Converting between formats
# Convert CSV to list of dictionaries
def csv_to_dict_list(filename):
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        return list(reader)

# Convert list of dictionaries to CSV
def dict_list_to_csv(data, filename):
    if not data:
        return
    
    with open(filename, "w", newline="") as file:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Example conversion
people_list = csv_to_dict_list("people.csv")
print(f"\nConverted to list: {len(people_list)} people")

# 14. CSV with custom formatting
# Create CSV with custom formatting
formatted_data = [
    ["Product", "Price", "Stock"],
    ["Laptop", "999.99", "5"],
    ["Mouse", "29.99", "50"],
    ["Keyboard", "79.99", "25"]
]

# Write with custom formatting
with open("inventory.csv", "w", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
    writer.writerows(formatted_data)

# Read and process inventory
with open("inventory.csv", "r") as file:
    reader = csv.DictReader(file)
    total_value = 0
    for row in reader:
        value = float(row["Price"]) * int(row["Stock"])
        total_value += value
        print(f"{row['Product']}: {row['Stock']} units @ ${row['Price']} = ${value:.2f}")

print(f"Total inventory value: ${total_value:.2f}")

# 15. Error handling in CSV operations
def safe_csv_read(filename):
    try:
        with open(filename, "r") as file:
            reader = csv.reader(file)
            rows = list(reader)
            print(f"Successfully read {len(rows)} rows from {filename}")
            return rows
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

# Test error handling
safe_csv_read("nonexistent.csv")
safe_csv_read("people.csv")

# Clean up created files
files_to_remove = [
    "people.csv", "people_dict.csv", "semicolon.csv", "products.csv",
    "unicode.csv", "quotes.csv", "large_data.csv", "inventory.csv"
]

for filename in files_to_remove:
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed {filename}")

"""
CSV Module Functions:
---------------------
- csv.reader(): Read CSV data
- csv.writer(): Write CSV data
- csv.DictReader(): Read CSV as dictionaries
- csv.DictWriter(): Write CSV from dictionaries
- csv.QUOTE_ALL: Quote all fields
- csv.QUOTE_MINIMAL: Quote only when necessary
- csv.QUOTE_NONE: Never quote fields
- csv.QUOTE_NONNUMERIC: Quote non-numeric fields

Common CSV Operations:
---------------------
- Reading CSV files
- Writing CSV files
- Filtering data
- Converting formats
- Handling different delimiters
- Working with headers
- Error handling

When to Use CSV:
----------------
- Simple tabular data
- Data exchange between applications
- Import/export operations
- Configuration files
- Log files

When NOT to Use CSV:
--------------------
- Complex nested data
- Binary data
- Large datasets (use databases)
- When you need relationships
- When performance is critical

Tips:
-----
- Always use newline="" when writing CSV
- Handle encoding issues with UTF-8
- Use DictReader/DictWriter for named columns
- Validate data before writing
- Handle exceptions properly
- Consider using pandas for complex operations

"""
