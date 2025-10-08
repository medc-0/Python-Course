"""
20-OOP.py

Beginner's guide to Object-Oriented Programming (OOP) in Python.

Overview:
---------
OOP lets you organize code using classes and objects. Classes are blueprints for creating objects (instances).
Objects have attributes (data) and methods (functions).

Examples:
---------
"""

# Basic class and object
class Person:
    # Class variable (shared by all instances)
    species = "Human"

    def __init__(self, name, age):
        # Instance variables (unique to each object)
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name}.")

# Create an object
p = Person("Alice", 30)
print(p.name)   # Output: Alice
print(p.age)    # Output: 30
p.greet()       # Output: Hello, my name is Alice.
print(p.species) # Output: Human

# Another object
q = Person("Bob", 25)
q.greet()       # Output: Hello, my name is Bob.

# Inheritance: Child class inherits from Parent class
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)  # Call parent constructor
        self.student_id = student_id

    def study(self):
        print(f"{self.name} is studying.")

s = Student("Charlie", 20, "S123")
s.greet()       # Output: Hello, my name is Charlie.
print(s.student_id) # Output: S123
s.study()       # Output: Charlie is studying.

# Method overriding
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):
        print("Woof!")

a = Animal()
a.speak()       # Output: Animal speaks

d = Dog()
d.speak()       # Output: Woof!

# Class method and static method
class Math:
    pi = 3.14159

    @classmethod
    def show_pi(cls):
        print(cls.pi)

    @staticmethod
    def add(a, b):
        return a + b

Math.show_pi()          # Output: 3.14159
print(Math.add(2, 3))   # Output: 5

"""
Tips:
-----
- Use classes to group related data and functions.
- Objects are created from classes.
- __init__ is the constructor method, called when you create an object.
- Use inheritance to reuse code between classes.
- Use self to refer to the current object.
- Use class variables for shared data, instance variables for unique data.
- Use @classmethod and @staticmethod for special methods.

"""