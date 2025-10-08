"""
12-system-commands.py

Beginner's guide to running system commands with subprocess.

Overview:
---------
The subprocess module lets you run shell/system commands from Python scripts.
You can automate tasks like running programs, copying files, or getting system info.

Main Features:
--------------
- Run commands and get their output
- Automate shell tasks (like 'ls', 'echo', 'dir', etc.)
- Capture errors and return codes

Examples:
---------
"""

import subprocess # import subprocess module pre-built into python

# 1. Run a command and get output
result = subprocess.run(["echo", "Hello from Python!"], capture_output=True, text=True)
print(result.stdout) # result is our variable that called the subprocess.run() function and .stdout stands for 'standard-output'.

# 2. Run a command and check for errors
result = subprocess.run(["ls", "not_a_real_file"], capture_output=True, text=True)
print("Return code:", result.returncode)
print("Error output:", result.stderr)

# 3. Run a command with shell=True (be careful!)
result = subprocess.run("echo Hello with shell", shell=True, capture_output=True, text=True)
print(result.stdout)

"""
Tips:
-----
- Use subprocess for advanced automation and integration.
- Be careful with shell=True (security risks).
- Official docs: https://docs.python.org/3/library/subprocess.html
"""
