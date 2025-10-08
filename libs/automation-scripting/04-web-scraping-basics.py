"""
04-web-scraping-basics.py

Beginner's guide to web scraping with requests and BeautifulSoup.

Overview:
---------
Web scraping means extracting data from web pages automatically.
Python's requests library fetches the page, and BeautifulSoup parses the HTML so you can search and extract information.

What is BeautifulSoup?
----------------------
- BeautifulSoup is a Python library for parsing HTML and XML documents.
- It lets you search, navigate, and modify the HTML structure easily.
- You can find elements by tag, class, id, or attributes, and extract text or links.

How does it work?
-----------------
1. Use requests to download the web page.
2. Pass the HTML to BeautifulSoup to create a "soup" object.
3. Use soup methods to find and extract the data you want.

Setup:
------
- Install libraries: pip install requests beautifulsoup4

Examples:
---------
"""

import requests
from bs4 import BeautifulSoup, Tag

url = "https://www.python.org"

# requests.get() returns a Response object containing the HTML
response = requests.get(url)

# BeautifulSoup parses the HTML and creates a soup object
soup = BeautifulSoup(response.text, "html.parser")

# 1. Get all links on the page
for link in soup.find_all("a"):
    href = link.get("href") # type: ignore
    print(href)

# 2. Get all text from the page
page_text = soup.get_text()
print(page_text)

# 3. Find elements by class name
for item in soup.find_all(class_="introduction"):
    item_tag = item # type: ignore
    print(item_tag.text)

# 4. Find elements by tag and attribute
for div in soup.find_all("div", {"id": "touchnav-wrapper"}):
    div_tag = div # type: ignore
    print(div_tag.text)

# 5. Extract all image sources
for img in soup.find_all("img"):
    src = img.get("src") # type: ignore
    print(src)

# 6. Save all links to a file
with open("links.txt", "w", encoding="utf-8") as f:
    for link in soup.find_all("a"):
        href = link.get("href") # type: ignore
        if href:
            f.write(href + "\n") # type: ignore

"""
Tips:
-----
- Official Learning Documentation: https://beautiful-soup-4.readthedocs.io/en/latest/
- BeautifulSoup makes it easy to search and extract data from HTML.
- Use soup.find_all(tag) to get all elements of a type.
- Use soup.find(tag) to get the first element.
- Use .text to get the text inside an element.
- Use attributes like class_ and id to filter elements.
- Always check website terms of service before scraping.
- Save data to files for later use.
- For complex sites, inspect the HTML structure with browser dev tools.
"""
