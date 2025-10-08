"""
02-drawing.py

Drawing shapes with Pygame.
"""

import pygame # type: ignore

pygame.init()
screen = pygame.display.set_mode((400, 300))
screen.fill((255, 255, 255))  # Fill background with white

# Draw a red rectangle
pygame.draw.rect(screen, (255, 0, 0), (50, 50, 100, 60))

# Draw a blue circle
pygame.draw.circle(screen, (0, 0, 255), (200, 150), 40)

pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
pygame.quit()
