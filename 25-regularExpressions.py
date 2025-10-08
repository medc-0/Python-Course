"""
25-regularExpressions.py

Beginner's guide to Python regular expressions.

Overview:
---------
Regular expressions (regex) are powerful tools for pattern matching and text manipulation. They allow you to search, extract, and modify text based on complex patterns.

What are Regular Expressions?
-----------------------------
Regular expressions are sequences of characters that define a search pattern. They are used for string matching, validation, and text processing.

Key Concepts:
--------------
- Pattern: The regex pattern to match
- Match: A successful pattern match
- Group: Captured parts of the pattern
- Flags: Options that modify matching behavior
- Metacharacters: Special characters with special meanings

Examples:
---------
"""

import re

# 1. Basic pattern matching
text = "Hello, World! Python is awesome."
pattern = r"Python"
match = re.search(pattern, text)
if match:
    print(f"Found: {match.group()}")
    print(f"Position: {match.start()}-{match.end()}")
else:
    print("Pattern not found")

# 2. Common metacharacters
text = "The quick brown fox jumps over the lazy dog"

# Dot (.) - matches any character except newline
pattern = r"brown.fox"
match = re.search(pattern, text)
if match:
    print(f"Dot match: {match.group()}")

# Asterisk (*) - matches zero or more of the preceding character
pattern = r"o*"
matches = re.findall(pattern, text)
print(f"Asterisk matches: {matches[:5]}")  # Show first 5 matches

# Plus (+) - matches one or more of the preceding character
pattern = r"o+"
matches = re.findall(pattern, text)
print(f"Plus matches: {matches}")

# Question mark (?) - matches zero or one of the preceding character
pattern = r"colou?r"
text2 = "color and colour"
matches = re.findall(pattern, text2)
print(f"Question mark matches: {matches}")

# 3. Character classes
text = "The year is 2023 and the temperature is 25.5°C"

# Digits
pattern = r"\d+"
matches = re.findall(pattern, text)
print(f"Digits: {matches}")

# Non-digits
pattern = r"\D+"
matches = re.findall(pattern, text)
print(f"Non-digits: {matches[:3]}")  # Show first 3 matches

# Word characters
pattern = r"\w+"
matches = re.findall(pattern, text)
print(f"Words: {matches[:5]}")  # Show first 5 matches

# Non-word characters
pattern = r"\W+"
matches = re.findall(pattern, text)
print(f"Non-words: {matches}")

# 4. Anchors
text = "Start of line\nMiddle of line\nEnd of line"

# Start of line
pattern = r"^Start"
matches = re.findall(pattern, text, re.MULTILINE)
print(f"Start of line: {matches}")

# End of line
pattern = r"line$"
matches = re.findall(pattern, text, re.MULTILINE)
print(f"End of line: {matches}")

# Word boundary
pattern = r"\bline\b"
matches = re.findall(pattern, text)
print(f"Word boundary: {matches}")

# 5. Groups and capturing
text = "Contact: john.doe@email.com or jane.smith@company.org"

# Email pattern with groups
pattern = r"(\w+)\.(\w+)@(\w+)\.(\w+)"
matches = re.findall(pattern, text)
print(f"Email groups: {matches}")

# Named groups
pattern = r"(?P<first>\w+)\.(?P<last>\w+)@(?P<domain>\w+)\.(?P<extension>\w+)"
match = re.search(pattern, text)
if match:
    print(f"Named groups: {match.groupdict()}")

# 6. Quantifiers
text = "a aa aaa aaaa aaaaa"

# Exact count
pattern = r"a{3}"
matches = re.findall(pattern, text)
print(f"Exactly 3 a's: {matches}")

# Range
pattern = r"a{2,4}"
matches = re.findall(pattern, text)
print(f"2-4 a's: {matches}")

# At least
pattern = r"a{3,}"
matches = re.findall(pattern, text)
print(f"At least 3 a's: {matches}")

# 7. Alternation
text = "I love cats and dogs"

# Either/or
pattern = r"cats|dogs"
matches = re.findall(pattern, text)
print(f"Either cats or dogs: {matches}")

# 8. Character sets
text = "The colors are red, green, blue, and yellow"

# Character set
pattern = r"[aeiou]"
matches = re.findall(pattern, text)
print(f"Vowels: {matches[:10]}")  # Show first 10 matches

# Negated character set
pattern = r"[^aeiou\s]"
matches = re.findall(pattern, text)
print(f"Non-vowels: {matches[:10]}")  # Show first 10 matches

# Range
pattern = r"[a-z]"
matches = re.findall(pattern, text)
print(f"Lowercase letters: {matches[:10]}")  # Show first 10 matches

# 9. Common patterns
text = "Phone: (555) 123-4567, Email: user@example.com, Date: 2023-12-25"

# Phone number
phone_pattern = r"\(\d{3}\) \d{3}-\d{4}"
phone_match = re.search(phone_pattern, text)
if phone_match:
    print(f"Phone: {phone_match.group()}")

# Email
email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
email_match = re.search(email_pattern, text)
if email_match:
    print(f"Email: {email_match.group()}")

# Date
date_pattern = r"\d{4}-\d{2}-\d{2}"
date_match = re.search(date_pattern, text)
if date_match:
    print(f"Date: {date_match.group()}")

# 10. Substitution
text = "The quick brown fox jumps over the lazy dog"

# Replace words
new_text = re.sub(r"\b\w{4}\b", "****", text)
print(f"Original: {text}")
print(f"Replaced: {new_text}")

# Replace with function
def replace_with_length(match):
    word = match.group()
    return f"{word}({len(word)})"

new_text = re.sub(r"\b\w+\b", replace_with_length, text)
print(f"With length: {new_text}")

# 11. Flags
text = "Python is great\npython is awesome\nPYTHON is powerful"

# Case insensitive
pattern = r"python"
matches = re.findall(pattern, text, re.IGNORECASE)
print(f"Case insensitive: {matches}")

# Multiline
pattern = r"^python"
matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
print(f"Multiline: {matches}")

# 12. Compiling patterns
# For better performance when using the same pattern multiple times
pattern = re.compile(r"\d{3}-\d{3}-\d{4}")
texts = ["Call 555-123-4567", "Phone: 555-987-6543", "No phone here"]

for text in texts:
    match = pattern.search(text)
    if match:
        print(f"Found phone: {match.group()}")

# 13. Split with regex
text = "apple,banana;cherry:date"
pattern = r"[,;:]"
parts = re.split(pattern, text)
print(f"Split result: {parts}")

# 14. Find all with groups
text = "John: 25, Jane: 30, Bob: 35"
pattern = r"(\w+): (\d+)"
matches = re.findall(pattern, text)
print(f"Name-age pairs: {matches}")

# 15. Lookahead and lookbehind
text = "The price is $100 and the cost is $50"

# Positive lookahead
pattern = r"\$\d+(?=\s)"
matches = re.findall(pattern, text)
print(f"Prices with lookahead: {matches}")

# Positive lookbehind
pattern = r"(?<=\$)\d+"
matches = re.findall(pattern, text)
print(f"Numbers after $: {matches}")

"""
Common Regex Patterns:
---------------------
- Email: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
- Phone: r'\(\d{3}\) \d{3}-\d{4}'
- URL: r'https?://[^\s]+'
- Date: r'\d{4}-\d{2}-\d{2}'
- IP: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
- Credit Card: r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'

Regex Flags:
------------
- re.IGNORECASE (re.I): Case insensitive matching
- re.MULTILINE (re.M): ^ and $ match start/end of lines
- re.DOTALL (re.S): . matches newline
- re.VERBOSE (re.X): Allow comments and whitespace
- re.ASCII (re.A): \w, \W, \b, \B match ASCII only

When to Use Regex:
------------------
- Text validation
- Data extraction
- Text replacement
- Pattern matching
- String parsing

When NOT to Use Regex:
----------------------
- Simple string operations
- When performance is critical
- Complex parsing (use proper parsers)
- When readability is more important

Tips:
-----
- Test regex patterns with online tools
- Use raw strings (r"pattern") to avoid escaping
- Compile patterns for repeated use
- Use groups to capture parts of matches
- Be careful with greedy vs non-greedy matching
- Consider using alternative libraries for complex parsing

"""
