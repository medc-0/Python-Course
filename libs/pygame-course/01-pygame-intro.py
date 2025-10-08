"""
01-pygame-intro.py

Introduction to Pygame.

Overview:
---------
Pygame is a library for making games with Python. It lets you create windows, draw graphics, play sounds, and handle user input.

Example: Open a window
----------------------
"""

import pygame # type: ignore

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Pygame Window")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
