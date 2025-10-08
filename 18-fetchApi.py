"""
18-fetchApi.py

Beginner's guide to fetching data from APIs in Python.

Overview:
---------
APIs let you get data from the web. The requests library is commonly used.

Examples:
---------
"""
# import requests # install requests in the terminal 'pip install requests'.

# # Fetch data from an API
# response = requests.get("https://api.github.com")
# print(response.status_code)  # Output: 200 (if successful)
# print(response.json())       # Output: API data in dictionary format (JSON)

"""
Tips:
-----
- Install requests with: pip install requests
- Use response.json() to get data as a dictionary.
- Always check response.status_code for success (200).

"""