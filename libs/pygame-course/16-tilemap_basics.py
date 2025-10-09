"""
16-tilemap_basics.py

Tilemap Basics
--------------
Render a tile map from a simple 2D array, with a small tileset atlas.
"""

import sys
import pygame  # type: ignore


TILE = 32
MAP = [
    "........................",
    "........................",
    "....########............",
    "....#......#............",
    "....#......#............",
    "....########............",
    "........................",
    "...........####.........",
    "...........#..#.........",
    "...........####.........",
]


def make_tileset() -> dict[str, pygame.Surface]:
    # Create a small atlas of two tiles: floor and wall
    floor = pygame.Surface((TILE, TILE))
    floor.fill((54, 57, 63))
    pygame.draw.rect(floor, (64, 68, 75), (0, 0, TILE, TILE), 1)

    wall = pygame.Surface((TILE, TILE))
    wall.fill((100, 100, 110))
    pygame.draw.rect(wall, (140, 140, 150), (0, 0, TILE, TILE), 2)
    return {".": floor, "#": wall}


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((len(MAP[0]) * TILE, len(MAP) * TILE))
    pygame.display.set_caption("Pygame Lesson 16 - Tilemap Basics")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    tileset = make_tileset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Draw map
        for y, row in enumerate(MAP):
            for x, ch in enumerate(row):
                tile = tileset[ch]
                screen.blit(tile, (x * TILE, y * TILE))

        tip = font.render("Rendering map from 2D array with tileset surfaces", True, (230, 230, 230))
        screen.blit(tip, (8, 8))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


