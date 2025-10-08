"""
22-generators.py

Beginner's guide to Python generators.

Overview:
---------
Generators are a special type of function that can pause and resume their execution. They use the `yield` keyword instead of `return` and are memory-efficient for handling large datasets.

What are Generators?
--------------------
Generators are functions that return an iterator. They generate values on-the-fly (lazy evaluation) rather than storing all values in memory at once.

Key Concepts:
--------------
- `yield` keyword: Pauses function execution and returns a value
- Generator objects: Created when you call a generator function
- Lazy evaluation: Values are generated only when needed
- Memory efficient: Only one value in memory at a time

Examples:
---------
"""

# 1. Basic generator function
def simple_generator():
    print("Generator started")
    yield 1
    print("After first yield")
    yield 2
    print("After second yield")
    yield 3
    print("Generator finished")

# Using the generator
gen = simple_generator()
print("Generator created:", gen)  # Output: <generator object simple_generator at 0x...>

# Getting values from generator
print("First value:", next(gen))  # Output: Generator started, First value: 1
print("Second value:", next(gen))  # Output: After first yield, Second value: 2
print("Third value:", next(gen))  # Output: After second yield, Third value: 3

# 2. Generator with loop
def count_up_to(n):
    count = 1
    while count <= n:
        yield count
        count += 1

# Using the generator
counter = count_up_to(5)
for num in counter:
    print(f"Counting: {num}")

# 3. Generator for Fibonacci sequence
def fibonacci_generator(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Generate first 10 Fibonacci numbers
fib_gen = fibonacci_generator(10)
fib_numbers = list(fib_gen)
print("Fibonacci numbers:", fib_numbers)

# 4. Generator with conditions
def even_numbers_generator(max_num):
    for i in range(max_num):
        if i % 2 == 0:
            yield i

# Get even numbers up to 20
even_gen = even_numbers_generator(20)
even_list = list(even_gen)
print("Even numbers:", even_list)

# 5. Generator expressions (similar to list comprehensions)
# List comprehension (creates list in memory)
squares_list = [x**2 for x in range(10)]
print("Squares list:", squares_list)

# Generator expression (creates generator)
squares_gen = (x**2 for x in range(10))
print("Squares generator:", squares_gen)
print("Squares from generator:", list(squares_gen))

# 6. Generator for reading large files
def read_large_file(filename):
    try:
        with open(filename, 'r') as file:
            for line in file:
                yield line.strip()
    except FileNotFoundError:
        print(f"File {filename} not found")

# Example usage (uncomment if you have a file)
# for line in read_large_file("example.txt"):
#     print(line)

# 7. Generator with multiple yields
def number_operations(n):
    yield f"Original: {n}"
    yield f"Doubled: {n * 2}"
    yield f"Squared: {n ** 2}"
    yield f"Cubed: {n ** 3}"

# Using the generator
operations = number_operations(5)
for operation in operations:
    print(operation)

# 8. Generator with state
def counter_generator():
    count = 0
    while True:
        count += 1
        yield count

# Create counter
counter = counter_generator()
print("Counter values:")
for i in range(5):
    print(f"Count: {next(counter)}")

# 9. Generator with parameters
def range_generator(start, stop, step=1):
    current = start
    while current < stop:
        yield current
        current += step

# Using the generator
range_gen = range_generator(0, 10, 2)
range_list = list(range_gen)
print("Custom range:", range_list)

# 10. Generator for prime numbers
def prime_generator(limit):
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    num = 2
    while num < limit:
        if is_prime(num):
            yield num
        num += 1

# Generate prime numbers up to 30
prime_gen = prime_generator(30)
primes = list(prime_gen)
print("Prime numbers:", primes)

"""
Generator Methods:
-----------------
- next(): Get the next value from generator
- send(): Send a value to the generator
- throw(): Raise an exception in the generator
- close(): Close the generator

Generator vs Regular Functions:
------------------------------
Regular Functions:
- Use `return` to return values
- Execute completely and return
- All values stored in memory
- Can only return once

Generators:
- Use `yield` to return values
- Can pause and resume execution
- Values generated on demand
- Can yield multiple values

When to Use Generators:
-----------------------
- Working with large datasets
- Memory efficiency is important
- Processing data streams
- Creating infinite sequences
- Reading large files line by line

When NOT to Use Generators:
---------------------------
- When you need all values at once
- When you need to access values multiple times
- When the dataset is small
- When you need random access to values

Tips:
-----
- Use generators for memory efficiency
- Remember that generators are iterators
- Use `yield` instead of `return`
- Generators are consumed when iterated
- Use generator expressions for simple cases

"""
