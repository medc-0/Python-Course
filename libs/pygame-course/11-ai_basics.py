"""
11-ai_basics.py

AI Basics
--------
Implement a simple enemy that seeks the player and a patrolling enemy using a
finite-state machine (FSM).

What you will learn
-------------------
1) Steering towards a target with normalized vectors
2) Simple FSM for patrol → chase → return
3) Tuning speeds and detection radii

"""

import sys
import math
import pygame  # type: ignore


class Player:
    def __init__(self, x: int, y: int) -> None:
        self.rect = pygame.Rect(x, y, 28, 28)
        self.speed = 240

    def update(self, dt: float, bounds: pygame.Rect) -> None:
        keys = pygame.key.get_pressed()
        vx = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
        vy = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(keys[pygame.K_w] or keys[pygame.K_UP])
        length = (vx * vx + vy * vy) ** 0.5
        if length:
            vx /= length
            vy /= length
        self.rect.x += int(vx * self.speed * dt)
        self.rect.y += int(vy * self.speed * dt)
        self.rect.clamp_ip(bounds)


class Seeker:
    def __init__(self, x: int, y: int) -> None:
        self.rect = pygame.Rect(x, y, 24, 24)
        self.speed = 180

    def update(self, dt: float, target: pygame.Rect) -> None:
        dx = target.centerx - self.rect.centerx
        dy = target.centery - self.rect.centery
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        self.rect.x += int(self.speed * dt * dx / length)
        self.rect.y += int(self.speed * dt * dy / length)


class Patroller:
    def __init__(self, path: list[tuple[int, int]]) -> None:
        self.rect = pygame.Rect(path[0][0], path[0][1], 24, 24)
        self.path = path
        self.index = 0
        self.speed = 140
        self.state = "patrol"  # patrol, chase, return
        self.return_index = 0
        self.detect_radius = 140

    def update(self, dt: float, player: pygame.Rect) -> None:
        if self.state == "patrol":
            target = pygame.math.Vector2(self.path[self.index])
            pos = pygame.math.Vector2(self.rect.center)
            delta = target - pos
            if delta.length() < 4:
                self.index = (self.index + 1) % len(self.path)
            else:
                direction = delta.normalize()
                pos += direction * self.speed * dt
                self.rect.center = (int(pos.x), int(pos.y))

            # Player detection
            if pygame.math.Vector2(player.center).distance_to(self.rect.center) < self.detect_radius:
                self.state = "chase"

        elif self.state == "chase":
            pos = pygame.math.Vector2(self.rect.center)
            target = pygame.math.Vector2(player.center)
            delta = target - pos
            if delta.length() > 0:
                direction = delta.normalize()
                pos += direction * (self.speed + 60) * dt
                self.rect.center = (int(pos.x), int(pos.y))

            # Lose sight
            if pygame.math.Vector2(player.center).distance_to(self.rect.center) > self.detect_radius * 1.6:
                self.state = "return"
                self.return_index = self.index

        elif self.state == "return":
            # Head back to current path node
            target = pygame.math.Vector2(self.path[self.return_index])
            pos = pygame.math.Vector2(self.rect.center)
            delta = target - pos
            if delta.length() < 4:
                self.state = "patrol"
            else:
                direction = delta.normalize()
                pos += direction * self.speed * dt
                self.rect.center = (int(pos.x), int(pos.y))


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 11 - AI Basics")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    player = Player(120, 120)
    seeker = Seeker(600, 300)
    patroller = Patroller([(500, 100), (700, 100), (700, 200), (500, 200)])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0
        player.update(dt, screen.get_rect())
        seeker.update(dt, player.rect)
        patroller.update(dt, player.rect)

        # Draw
        screen.fill((242, 244, 248))
        pygame.draw.rect(screen, (80, 170, 240), player.rect, border_radius=6)  # player
        pygame.draw.rect(screen, (240, 110, 110), seeker.rect, border_radius=6)  # seeker
        pygame.draw.rect(screen, (120, 200, 120), patroller.rect, border_radius=6)  # patroller

        # Path visualization
        for i in range(len(patroller.path)):
            a = patroller.path[i]
            b = patroller.path[(i + 1) % len(patroller.path)]
            pygame.draw.line(screen, (160, 160, 160), a, b)
            pygame.draw.circle(screen, (100, 100, 100), a, 3)

        tips = [
            "Blue: player  |  Red: seeker (always chases)",
            "Green: patroller (FSM: patrol→chase→return)",
        ]
        for i, text in enumerate(tips):
            lbl = font.render(text, True, (30, 30, 30))
            screen.blit(lbl, (16, 16 + 22 * i))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


