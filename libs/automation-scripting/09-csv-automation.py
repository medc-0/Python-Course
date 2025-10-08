"""
09-csv-automation.py

Automate reading and writing CSV files.

Overview:
---------
CSV (Comma Separated Values) files are simple text files for storing tabular data (like spreadsheets).
Python's csv module makes it easy to read and write CSV files for data analysis, reporting, and automation.

How CSV works in Python:
------------------------
- Use csv.writer to write rows to a CSV file.
- Use csv.reader to read rows from a CSV file.
- Use csv.DictWriter and csv.DictReader for working with dictionaries.

Examples:
---------
"""

import csv

# Write to CSV
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 30])
    writer.writerow(["Bob", 25])

# Read from CSV
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print("Row:", row)

# Write using DictWriter
with open("data_dict.csv", "w", newline="") as f:
    fieldnames = ["Name", "Age"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"Name": "Charlie", "Age": 22})
    writer.writerow({"Name": "Dana", "Age": 28})

# Read using DictReader
with open("data_dict.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print("Dict row:", row)

# Clean up
import os
os.remove("data.csv")
os.remove("data_dict.csv")

"""
Tips:
-----
- Use csv.writer for writing, csv.reader for reading.
- Use newline="" when writing to avoid blank lines.
- Use DictWriter/DictReader for working with dictionaries.
- CSV is great for spreadsheets, reports, and exchanging tabular data.
- For more info: https://docs.python.org/3/library/csv.html
"""
