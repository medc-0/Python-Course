"""
03-sprites.py

Sprites and Images
------------------
In this lesson you will learn how to load images, create sprite classes,
organize them into groups, and draw/update them efficiently.

What you will learn
-------------------
1) Loading images safely (with a generated fallback if missing)
2) Building a `pygame.sprite.Sprite` subclass
3) Using `pygame.sprite.Group` to update/draw many sprites
4) Basic sprite transformations (scale/rotate)

Docs
----
- Sprites: https://www.pygame.org/docs/ref/sprite.html
- Image: https://www.pygame.org/docs/ref/image.html
- Transform: https://www.pygame.org/docs/ref/transform.html
"""

import os
import sys
import math
import pygame  # type: ignore


def load_image(path: str, size: tuple[int, int] | None = None) -> pygame.Surface:
    """Load an image; if not found, return a generated checkerboard surface."""
    if os.path.exists(path):
        image = pygame.image.load(path).convert_alpha()
        if size is not None:
            image = pygame.transform.smoothscale(image, size)
        return image
    # Fallback: generate a checkerboard so the lesson runs without assets
    w, h = size or (64, 64)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    c1, c2 = (200, 200, 200, 255), (140, 140, 140, 255)
    tile = 8
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            color = c1 if ((x // tile + y // tile) % 2 == 0) else c2
            pygame.draw.rect(surf, color, (x, y, tile, tile))
    # Draw an X mark
    pygame.draw.line(surf, (220, 60, 60), (0, 0), (w - 1, h - 1), 2)
    pygame.draw.line(surf, (220, 60, 60), (0, h - 1), (w - 1, 0), 2)
    return surf


class Player(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int]):
        super().__init__()
        base = load_image("player.png", (48, 48))
        self.frames: list[pygame.Surface] = [
            base,
            pygame.transform.rotate(base, 5),
            pygame.transform.rotate(base, -5),
        ]
        self.frame_index = 0
        self.image = self.frames[self.frame_index]
        self.rect = self.image.get_rect(center=pos)
        self.anim_timer = 0.0
        self.anim_interval = 0.12

    def update(self, dt: float) -> None:
        # Idle animation that subtly rotates
        self.anim_timer += dt
        if self.anim_timer >= self.anim_interval:
            self.anim_timer = 0.0
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.image = self.frames[self.frame_index]


class Star(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int], speed: float):
        super().__init__()
        size = 8
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255, 255, 210), (size // 2, size // 2), size // 2)
        self.rect = self.image.get_rect(center=pos)
        self.speed = speed

    def update(self, dt: float) -> None:
        self.rect.x -= int(self.speed * dt * 60)
        if self.rect.right < 0:
            self.rect.left = 800


def initialize(width: int = 800, height: int = 480) -> tuple[pygame.Surface, pygame.time.Clock]:
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pygame Lesson 03 - Sprites")
    return screen, pygame.time.Clock()


def spawn_stars() -> pygame.sprite.Group:
    stars = pygame.sprite.Group()
    for i in range(40):
        x = 20 * i
        y = 40 + int(360 * (0.5 + 0.5 * math.sin(i))) % 440
        speed = 1.5 + (i % 5)
        stars.add(Star((x, y), speed))
    return stars


def main() -> int:
    screen, clock = initialize()
    running = True

    # Groups
    all_sprites = pygame.sprite.Group()
    stars = spawn_stars()
    player = Player((400, 240))
    all_sprites.add(stars)
    all_sprites.add(player)

    # Font for labels
    font = pygame.font.SysFont(None, 20)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0

        # Update
        for sprite in all_sprites:
            # Group.update calls Sprite.update without arguments; we need dt
            if hasattr(sprite, "update"):
                sprite.update(dt)  # type: ignore[arg-type]

        # Draw
        screen.fill((16, 22, 30))
        all_sprites.draw(screen)

        # HUD
        label = font.render("Sprite groups: update() + draw()", True, (230, 230, 230))
        screen.blit(label, (16, 16))
        info = font.render("Image fallback used if player.png is missing", True, (180, 200, 255))
        screen.blit(info, (16, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


