"""
15-platformer_core.py

Platformer Core
---------------
Basic platformer movement with gravity, jumping, and tile collisions.
"""

import sys
import pygame  # type: ignore


TILE = 32
LEVEL = [
    "........................................",
    "........................................",
    ".....................###................",
    "...................#....................",
    ".............####.......................",
    "............##..........................",
    "..........####..........................",
    ".........###............................",
    "############################..###########",
]


class Player:
    def __init__(self, x: int, y: int) -> None:
        self.rect = pygame.Rect(x, y, 24, 28)
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.speed = 220.0
        self.jump_impulse = -420.0
        self.gravity = 1200.0
        self.on_ground = False

    def update(self, dt: float, tiles: list[pygame.Rect]) -> None:
        keys = pygame.key.get_pressed()
        move = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
        self.velocity_x = move * self.speed
        if (keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]) and self.on_ground:
            self.velocity_y = self.jump_impulse
            self.on_ground = False

        self.velocity_y += self.gravity * dt

        # Horizontal
        self.rect.x += int(self.velocity_x * dt)
        for t in tiles:
            if self.rect.colliderect(t):
                if self.velocity_x > 0:
                    self.rect.right = t.left
                elif self.velocity_x < 0:
                    self.rect.left = t.right

        # Vertical
        self.rect.y += int(self.velocity_y * dt)
        self.on_ground = False
        for t in tiles:
            if self.rect.colliderect(t):
                if self.velocity_y > 0:
                    self.rect.bottom = t.top
                    self.velocity_y = 0.0
                    self.on_ground = True
                elif self.velocity_y < 0:
                    self.rect.top = t.bottom
                    self.velocity_y = 0.0


def build_tiles() -> list[pygame.Rect]:
    tiles: list[pygame.Rect] = []
    for y, row in enumerate(LEVEL):
        for x, ch in enumerate(row):
            if ch == "#":
                tiles.append(pygame.Rect(x * TILE, y * TILE, TILE, TILE))
    return tiles


def main() -> int:
    pygame.init()
    width, height = len(LEVEL[0]) * TILE, len(LEVEL) * TILE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pygame Lesson 15 - Platformer Core")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    tiles = build_tiles()
    player = Player(64, 64)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0
        player.update(dt, tiles)

        screen.fill((32, 36, 40))
        for t in tiles:
            pygame.draw.rect(screen, (70, 70, 70), t)
        pygame.draw.rect(screen, (80, 170, 240), player.rect)

        hud = [
            "Arrows/WASD to move, Space to jump",
            "Solid tile collisions with horizontal+vertical resolution",
        ]
        for i, text in enumerate(hud):
            lbl = font.render(text, True, (230, 230, 230))
            screen.blit(lbl, (8, 8 + 20 * i))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


