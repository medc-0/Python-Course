"""
28-datetimeHandling.py

Beginner's guide to date and time handling in Python.

Overview:
---------
Python provides powerful tools for working with dates and times through the `datetime` module. This module allows you to create, manipulate, and format dates and times.

What is datetime?
-----------------
The datetime module provides classes for working with dates and times. It includes date, time, datetime, and timedelta classes for various operations.

Key Concepts:
--------------
- date: Represents a date (year, month, day)
- time: Represents a time (hour, minute, second, microsecond)
- datetime: Combines date and time
- timedelta: Represents a duration or difference between dates
- timezone: Represents timezone information

Examples:
---------
"""

from datetime import datetime, date, time, timedelta, timezone
import time as time_module

# 1. Basic date operations
# Create a date
today = date.today()
print(f"Today's date: {today}")
print(f"Year: {today.year}, Month: {today.month}, Day: {today.day}")

# Create a specific date
specific_date = date(2023, 12, 25)
print(f"Christmas 2023: {specific_date}")

# 2. Basic time operations
# Create a time
current_time = time(14, 30, 45)  # 2:30:45 PM
print(f"Time: {current_time}")
print(f"Hour: {current_time.hour}, Minute: {current_time.minute}, Second: {current_time.second}")

# 3. Basic datetime operations
# Get current datetime
now = datetime.now()
print(f"Current datetime: {now}")

# Create a specific datetime
specific_datetime = datetime(2023, 12, 25, 14, 30, 45)
print(f"Specific datetime: {specific_datetime}")

# 4. Date arithmetic
# Add days to a date
today = date.today()
tomorrow = today + timedelta(days=1)
next_week = today + timedelta(weeks=1)
next_month = today + timedelta(days=30)

print(f"Today: {today}")
print(f"Tomorrow: {tomorrow}")
print(f"Next week: {next_week}")
print(f"Next month: {next_month}")

# 5. Time arithmetic
# Add time to datetime
now = datetime.now()
in_one_hour = now + timedelta(hours=1)
in_one_day = now + timedelta(days=1)
in_one_week = now + timedelta(weeks=1)

print(f"Now: {now}")
print(f"In one hour: {in_one_hour}")
print(f"In one day: {in_one_day}")
print(f"In one week: {in_one_week}")

# 6. Date comparison
# Compare dates
date1 = date(2023, 12, 25)
date2 = date(2023, 12, 31)
date3 = date(2023, 12, 25)

print(f"Date1: {date1}")
print(f"Date2: {date2}")
print(f"Date3: {date3}")
print(f"Date1 < Date2: {date1 < date2}")
print(f"Date1 == Date3: {date1 == date3}")

# 7. Date formatting
# Format dates and times
now = datetime.now()

# Different format strings
formats = [
    ("%Y-%m-%d", "Year-Month-Day"),
    ("%d/%m/%Y", "Day/Month/Year"),
    ("%B %d, %Y", "Month Day, Year"),
    ("%A, %B %d, %Y", "Weekday, Month Day, Year"),
    ("%H:%M:%S", "Hour:Minute:Second"),
    ("%Y-%m-%d %H:%M:%S", "Full datetime")
]

print("\nDate formatting examples:")
for format_string, description in formats:
    formatted = now.strftime(format_string)
    print(f"{description}: {formatted}")

# 8. Parsing dates from strings
# Parse dates from strings
date_strings = [
    "2023-12-25",
    "25/12/2023",
    "December 25, 2023",
    "2023-12-25 14:30:45"
]

print("\nParsing dates from strings:")
for date_str in date_strings:
    try:
        if "/" in date_str:
            parsed = datetime.strptime(date_str, "%d/%m/%Y")
        elif "December" in date_str:
            parsed = datetime.strptime(date_str, "%B %d, %Y")
        elif " " in date_str and ":" in date_str:
            parsed = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        else:
            parsed = datetime.strptime(date_str, "%Y-%m-%d")
        print(f"'{date_str}' -> {parsed}")
    except ValueError as e:
        print(f"Error parsing '{date_str}': {e}")

# 9. Working with timezones
# Create timezone-aware datetime
utc_now = datetime.now(timezone.utc)
print(f"UTC time: {utc_now}")

# Convert to different timezone
from datetime import timezone, timedelta

# Create timezone objects
est = timezone(timedelta(hours=-5))  # Eastern Time
pst = timezone(timedelta(hours=-8))  # Pacific Time

# Convert to different timezones
utc_time = datetime.now(timezone.utc)
est_time = utc_time.astimezone(est)
pst_time = utc_time.astimezone(pst)

print(f"UTC: {utc_time}")
print(f"EST: {est_time}")
print(f"PST: {pst_time}")

# 10. Date differences
# Calculate differences between dates
date1 = date(2023, 1, 1)
date2 = date(2023, 12, 31)
difference = date2 - date1
print(f"\nDays between {date1} and {date2}: {difference.days}")

# Calculate age
birth_date = date(1990, 5, 15)
today = date.today()
age = today - birth_date
age_years = age.days // 365
print(f"Age: {age_years} years")

# 11. Working with timestamps
# Convert datetime to timestamp
now = datetime.now()
timestamp = now.timestamp()
print(f"Current timestamp: {timestamp}")

# Convert timestamp to datetime
from_timestamp = datetime.fromtimestamp(timestamp)
print(f"From timestamp: {from_timestamp}")

# 12. Date ranges
# Generate date ranges
start_date = date(2023, 1, 1)
end_date = date(2023, 1, 7)

current_date = start_date
print("\nDate range:")
while current_date <= end_date:
    print(f"  {current_date} ({current_date.strftime('%A')})")
    current_date += timedelta(days=1)

# 13. Working with business days
# Calculate business days (excluding weekends)
def business_days_between(date1, date2):
    current = date1
    business_days = 0
    while current < date2:
        if current.weekday() < 5:  # Monday = 0, Sunday = 6
            business_days += 1
        current += timedelta(days=1)
    return business_days

start = date(2023, 12, 1)
end = date(2023, 12, 31)
business_days = business_days_between(start, end)
print(f"\nBusiness days between {start} and {end}: {business_days}")

# 14. Date validation
# Validate dates
def is_valid_date(year, month, day):
    try:
        date(year, month, day)
        return True
    except ValueError:
        return False

# Test date validation
test_dates = [
    (2023, 12, 25),  # Valid
    (2023, 2, 29),   # Invalid (not leap year)
    (2023, 13, 1),   # Invalid month
    (2023, 12, 32),  # Invalid day
]

print("\nDate validation:")
for year, month, day in test_dates:
    valid = is_valid_date(year, month, day)
    print(f"{year}-{month:02d}-{day:02d}: {'Valid' if valid else 'Invalid'}")

# 15. Working with recurring dates
# Find next occurrence of a specific day
def next_weekday(target_date, weekday):
    """
    Find next occurrence of weekday (0=Monday, 6=Sunday)
    """
    days_ahead = weekday - target_date.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return target_date + timedelta(days=days_ahead)

# Find next Monday
today = date.today()
next_monday = next_weekday(today, 0)  # 0 = Monday
print(f"\nNext Monday: {next_monday}")

# 16. Date calculations
# Calculate days until next birthday
def days_until_birthday(birth_month, birth_day):
    today = date.today()
    this_year_birthday = date(today.year, birth_month, birth_day)
    
    if this_year_birthday < today:
        next_birthday = date(today.year + 1, birth_month, birth_day)
    else:
        next_birthday = this_year_birthday
    
    return (next_birthday - today).days

birth_month = 5
birth_day = 15
days_until = days_until_birthday(birth_month, birth_day)
print(f"Days until next birthday: {days_until}")

# 17. Working with time intervals
# Create time intervals
def create_time_intervals(start_time, end_time, interval_minutes):
    intervals = []
    current = start_time
    while current < end_time:
        intervals.append(current)
        current += timedelta(minutes=interval_minutes)
    return intervals

# Create 30-minute intervals
start = datetime(2023, 12, 25, 9, 0)  # 9:00 AM
end = datetime(2023, 12, 25, 17, 0)  # 5:00 PM
intervals = create_time_intervals(start, end, 30)

print(f"\nTime intervals:")
for interval in intervals:
    print(f"  {interval.strftime('%H:%M')}")

# 18. Date utilities
# Get first and last day of month
def month_boundaries(year, month):
    first_day = date(year, month, 1)
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    return first_day, last_day

first, last = month_boundaries(2023, 12)
print(f"\nDecember 2023: {first} to {last}")

# 19. Working with different date formats
# Handle different date formats
def parse_flexible_date(date_string):
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%d %B %Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt).date()
        except ValueError:
            continue
    return None

# Test flexible parsing
test_dates = [
    "2023-12-25",
    "25/12/2023",
    "12/25/2023",
    "December 25, 2023",
    "25 December 2023"
]

print("\nFlexible date parsing:")
for date_str in test_dates:
    parsed = parse_flexible_date(date_str)
    print(f"'{date_str}' -> {parsed}")

# 20. Performance timing
# Measure execution time
def measure_time(func):
    start = time_module.time()
    result = func()
    end = time_module.time()
    print(f"Execution time: {end - start:.4f} seconds")
    return result

def slow_operation():
    time_module.sleep(0.1)  # Simulate slow operation
    return "Operation completed"

result = measure_time(slow_operation)

"""
Common datetime operations:
--------------------------
- date.today(): Get current date
- datetime.now(): Get current datetime
- timedelta(): Create time differences
- strftime(): Format dates to strings
- strptime(): Parse strings to dates
- timezone(): Create timezone objects
- timestamp(): Convert to Unix timestamp

Date formatting codes:
---------------------
- %Y: 4-digit year
- %m: Month (01-12)
- %d: Day (01-31)
- %H: Hour (00-23)
- %M: Minute (00-59)
- %S: Second (00-59)
- %A: Weekday name
- %B: Month name

When to Use datetime:
--------------------
- Date calculations
- Time tracking
- Scheduling
- Data analysis
- Logging
- User interfaces

When NOT to Use datetime:
------------------------
- Simple timestamps (use time module)
- High-frequency operations
- When you need nanosecond precision
- When working with external systems

Tips:
-----
- Always handle timezone information
- Use strftime/strptime for formatting
- Be careful with date arithmetic
- Validate dates before processing
- Consider using third-party libraries for complex operations
- Use timedelta for time differences

"""
