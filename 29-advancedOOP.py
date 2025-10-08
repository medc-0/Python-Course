"""
29-advancedOOP.py

Beginner's guide to advanced Object-Oriented Programming in Python.

Overview:
---------
This guide covers advanced OOP concepts including inheritance, polymorphism, encapsulation, abstract classes, and design patterns. These concepts help create more maintainable and scalable code.

What is Advanced OOP?
--------------------
Advanced OOP builds upon basic OOP concepts to create more sophisticated and flexible code structures. It includes inheritance hierarchies, polymorphism, abstract classes, and design patterns.

Key Concepts:
--------------
- Inheritance: Creating new classes from existing ones
- Polymorphism: Using objects of different types interchangeably
- Encapsulation: Hiding internal implementation details
- Abstract classes: Defining interfaces without implementation
- Design patterns: Reusable solutions to common problems

Examples:
---------
"""

# 1. Basic inheritance
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "Some generic animal sound"
    
    def move(self):
        return f"{self.name} is moving"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Canine")
        self.breed = breed
    
    def make_sound(self):
        return "Woof!"
    
    def fetch(self):
        return f"{self.name} is fetching the ball"

class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name, "Feline")
        self.color = color
    
    def make_sound(self):
        return "Meow!"
    
    def climb(self):
        return f"{self.name} is climbing"

# Using inheritance
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "Orange")

print(f"Dog: {dog.name} ({dog.species})")
print(f"Dog sound: {dog.make_sound()}")
print(f"Dog action: {dog.fetch()}")

print(f"Cat: {cat.name} ({cat.species})")
print(f"Cat sound: {cat.make_sound()}")
print(f"Cat action: {cat.climb()}")

# 2. Multiple inheritance
class Flyable:
    def fly(self):
        return "Flying through the air"

class Swimmable:
    def swim(self):
        return "Swimming in water"

class Duck(Animal, Flyable, Swimmable):
    def __init__(self, name):
        super().__init__(name, "Bird")
    
    def make_sound(self):
        return "Quack!"

# Using multiple inheritance
duck = Duck("Donald")
print(f"Duck: {duck.name}")
print(f"Duck sound: {duck.make_sound()}")
print(f"Duck flying: {duck.fly()}")
print(f"Duck swimming: {duck.swim()}")

# 3. Polymorphism
def animal_sounds(animals):
    for animal in animals:
        print(f"{animal.name} says: {animal.make_sound()}")

# Create list of different animals
animals = [dog, cat, duck]
print("\nPolymorphism example:")
animal_sounds(animals)

# 4. Encapsulation with private attributes
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self.account_number = account_number
        self.__balance = initial_balance  # Private attribute
        self.__transactions = []  # Private attribute
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            self.__transactions.append(f"Deposit: +${amount}")
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            self.__transactions.append(f"Withdrawal: -${amount}")
            return True
        return False
    
    def get_balance(self):
        return self.__balance
    
    def get_transactions(self):
        return self.__transactions.copy()  # Return copy to prevent modification

# Using encapsulation
account = BankAccount("12345", 1000)
print(f"\nInitial balance: ${account.get_balance()}")

account.deposit(500)
account.withdraw(200)
print(f"Balance after transactions: ${account.get_balance()}")
print(f"Transactions: {account.get_transactions()}")

# 5. Property decorators
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

# Using properties
temp = Temperature(25)
print(f"\nTemperature: {temp.celsius}°C = {temp.fahrenheit}°F")

temp.fahrenheit = 86
print(f"After setting to 86°F: {temp.celsius}°C")

# 6. Abstract base classes
from abc import ABC, abstractmethod

class Shape(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass
    
    def describe(self):
        return f"{self.name}: Area={self.area():.2f}, Perimeter={self.perimeter():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Using abstract classes
shapes = [
    Rectangle(5, 3),
    Circle(4)
]

print("\nAbstract classes example:")
for shape in shapes:
    print(shape.describe())

# 7. Class methods and static methods
class MathUtils:
    pi = 3.14159
    
    @classmethod
    def circle_area(cls, radius):
        return cls.pi * radius ** 2
    
    @staticmethod
    def add(a, b):
        return a + b
    
    @staticmethod
    def multiply(a, b):
        return a * b

# Using class and static methods
print(f"\nCircle area: {MathUtils.circle_area(5)}")
print(f"Addition: {MathUtils.add(3, 4)}")
print(f"Multiplication: {MathUtils.multiply(3, 4)}")

# 8. Method overriding and super()
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
        self.speed = 0
    
    def start(self):
        return f"{self.brand} {self.model} is starting"
    
    def stop(self):
        return f"{self.brand} {self.model} is stopping"
    
    def accelerate(self, speed):
        self.speed += speed
        return f"Speed increased to {self.speed} km/h"

class Car(Vehicle):
    def __init__(self, brand, model, doors):
        super().__init__(brand, model)
        self.doors = doors
    
    def start(self):
        return f"Car {self.brand} {self.model} is starting with {self.doors} doors"
    
    def honk(self):
        return "Beep beep!"

class Motorcycle(Vehicle):
    def __init__(self, brand, model, has_sidecar):
        super().__init__(brand, model)
        self.has_sidecar = has_sidecar
    
    def start(self):
        return f"Motorcycle {self.brand} {self.model} is starting"
    
    def wheelie(self):
        return "Doing a wheelie!"

# Using method overriding
car = Car("Toyota", "Camry", 4)
motorcycle = Motorcycle("Honda", "CBR", False)

print(f"\nMethod overriding:")
print(car.start())
print(motorcycle.start())
print(car.honk())
print(motorcycle.wheelie())

# 9. Composition vs Inheritance
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
        self.is_running = False
    
    def start(self):
        self.is_running = True
        return "Engine started"
    
    def stop(self):
        self.is_running = False
        return "Engine stopped"

class CarWithEngine:
    def __init__(self, brand, model, engine_hp):
        self.brand = brand
        self.model = model
        self.engine = Engine(engine_hp)  # Composition
    
    def start(self):
        return f"{self.brand} {self.model}: {self.engine.start()}"
    
    def stop(self):
        return f"{self.brand} {self.model}: {self.engine.stop()}"

# Using composition
car_with_engine = CarWithEngine("BMW", "X5", 300)
print(f"\nComposition example:")
print(car_with_engine.start())
print(car_with_engine.stop())

# 10. Design patterns - Singleton
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.data = []

# Using Singleton
s1 = Singleton()
s2 = Singleton()
print(f"\nSingleton pattern: {s1 is s2}")  # Should be True

# 11. Design patterns - Factory
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type, name, **kwargs):
        if animal_type == "dog":
            return Dog(name, kwargs.get("breed", "Mixed"))
        elif animal_type == "cat":
            return Cat(name, kwargs.get("color", "Unknown"))
        elif animal_type == "duck":
            return Duck(name)
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Using Factory pattern
factory = AnimalFactory()
animals = [
    factory.create_animal("dog", "Rex", breed="German Shepherd"),
    factory.create_animal("cat", "Fluffy", color="White"),
    factory.create_animal("duck", "Daffy")
]

print("\nFactory pattern:")
for animal in animals:
    print(f"{animal.name}: {animal.make_sound()}")

# 12. Design patterns - Observer
class Observer:
    def update(self, subject):
        pass

class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)

class WeatherStation(Subject):
    def __init__(self):
        super().__init__()
        self._temperature = 0
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        self.notify()

class TemperatureDisplay(Observer):
    def update(self, subject):
        print(f"Temperature updated: {subject.temperature}°C")

class TemperatureAlert(Observer):
    def __init__(self, threshold):
        self.threshold = threshold
    
    def update(self, subject):
        if subject.temperature > self.threshold:
            print(f"ALERT: Temperature {subject.temperature}°C exceeds threshold {self.threshold}°C")

# Using Observer pattern
weather_station = WeatherStation()
display = TemperatureDisplay()
alert = TemperatureAlert(30)

weather_station.attach(display)
weather_station.attach(alert)

print("\nObserver pattern:")
weather_station.temperature = 25
weather_station.temperature = 35

# 13. Method resolution order (MRO)
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

# Check MRO
print(f"\nMethod Resolution Order for D: {D.__mro__}")

d = D()
print(f"D().method() returns: {d.method()}")

# 14. Special methods (magic methods)
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        return 2
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Vector index out of range")

# Using special methods
v1 = Vector(3, 4)
v2 = Vector(1, 2)
v3 = v1 + v2
v4 = v1 * 2

print(f"\nSpecial methods:")
print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v1 + v2: {v3}")
print(f"v1 * 2: {v4}")
print(f"v1 == v2: {v1 == v2}")
print(f"Length of v1: {len(v1)}")
print(f"v1[0]: {v1[0]}, v1[1]: {v1[1]}")

# 15. Mixins
class TimestampMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = datetime.now()
    
    def get_age(self):
        return datetime.now() - self.created_at

class TimestampedAnimal(Animal, TimestampMixin):
    def __init__(self, name, species):
        super().__init__(name, species)

# Using mixins
timestamped_animal = TimestampedAnimal("Timmy", "Generic")
print(f"\nMixin example:")
print(f"Animal created at: {timestamped_animal.created_at}")
print(f"Age: {timestamped_animal.get_age()}")

"""
Advanced OOP Concepts:
----------------------
- Inheritance: Code reuse and specialization
- Polymorphism: Same interface, different behavior
- Encapsulation: Data hiding and access control
- Abstract classes: Defining interfaces
- Design patterns: Reusable solutions
- Special methods: Customizing behavior
- Mixins: Multiple inheritance for functionality

When to Use Advanced OOP:
-------------------------
- Complex applications
- Code reuse requirements
- Maintainable codebases
- Team development
- Large projects

When NOT to Use Advanced OOP:
-----------------------------
- Simple scripts
- Performance-critical code
- When over-engineering
- Small projects
- When functional programming is better

Tips:
-----
- Use inheritance for "is-a" relationships
- Use composition for "has-a" relationships
- Keep inheritance hierarchies shallow
- Use abstract classes for interfaces
- Follow design patterns when appropriate
- Don't overuse inheritance
- Prefer composition over inheritance

"""
