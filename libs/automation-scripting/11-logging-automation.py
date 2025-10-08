"""
11-logging-automation.py

Beginner's guide to logging automation in Python.

Overview:
---------
The logging module lets you record events, errors, warnings, and info from your scripts.
Logging is useful for debugging, monitoring, and keeping records of what your program does.

Main Features:
--------------
- Log messages to files or console
- Different log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Format log messages with timestamps and details

Examples:
---------
"""

import logging

# 1. Basic logging to a file
logging.basicConfig(filename="app.log", level=logging.INFO)
logging.info("Script started")
logging.warning("This is a warning")
logging.error("This is an error")

# 2. Logging to console
logging.basicConfig(level=logging.DEBUG)
logging.debug("Debug message")
logging.info("Info message")

# 3. Custom log format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.info("Custom formatted log")

# Read log file
with open("app.log", "r") as f:
    print(f.read())

# Clean up
import os
os.remove("app.log")

"""
Tips:
-----
- Use logging for debugging and monitoring scripts.
- Choose the right log level for your message.
- Official docs: https://docs.python.org/3/library/logging.html
"""
