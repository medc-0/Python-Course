"""
10-physics_basics.py

Physics Basics
--------------
Add gravity, velocity, and friction to simulate motion. Implement simple
ground collision and jumping.

What you will learn
-------------------
1) Velocity and acceleration integration with dt
2) Gravity and jump impulse
3) Ground collision and friction
4) Clamping and stable updates

Docs
----
- Time/Clock: https://www.pygame.org/docs/ref/time.html
- Rect: https://www.pygame.org/docs/ref/rect.html
"""

import sys
import pygame  # type: ignore


class Player:
    def __init__(self, x: int, y: int) -> None:
        self.rect = pygame.Rect(x, y, 40, 40)
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.speed = 420.0
        self.gravity = 1400.0
        self.jump_impulse = -520.0
        self.on_ground = False

    def update(self, dt: float, ground_y: int) -> None:
        # Horizontal control
        keys = pygame.key.get_pressed()
        move = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(keys[pygame.K_LEFT] or keys[pygame.K_a])
        self.velocity_x = move * self.speed

        # Jump
        if self.on_ground and (keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]):
            self.velocity_y = self.jump_impulse
            self.on_ground = False

        # Gravity
        self.velocity_y += self.gravity * dt

        # Integrate
        self.rect.x += int(self.velocity_x * dt)
        self.rect.y += int(self.velocity_y * dt)

        # Ground collision
        if self.rect.bottom >= ground_y:
            self.rect.bottom = ground_y
            self.velocity_y = 0.0
            self.on_ground = True

        # Simple horizontal friction when on ground
        if self.on_ground:
            self.velocity_x *= 0.85


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 10 - Physics Basics")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    player = Player(120, 200)
    ground_y = 420

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0
        player.update(dt, ground_y)

        # Draw
        screen.fill((240, 244, 248))
        pygame.draw.rect(screen, (40, 40, 40), (0, ground_y, 800, 4))
        pygame.draw.rect(screen, (80, 170, 240), player.rect, border_radius=6)

        lines = [
            "A/D or Left/Right to move, SPACE to jump",
            "Gravity + jump impulse + ground collision",
        ]
        for i, text in enumerate(lines):
            lbl = font.render(text, True, (30, 30, 30))
            screen.blit(lbl, (16, 16 + i * 22))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


