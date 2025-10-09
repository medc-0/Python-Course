"""
13-pong.py

Pong - Complete Mini Game
-------------------------
Classic Pong with paddles, ball physics, collision, scoring, and simple UI.

What you will learn
-------------------
1) Structuring a small complete game
2) Handling paddle/ball collisions and score
3) Resetting round and displaying UI text
"""

import sys
import random
import pygame  # type: ignore


WIDTH, HEIGHT = 800, 480


class Paddle:
    def __init__(self, x: int) -> None:
        self.rect = pygame.Rect(x, HEIGHT // 2 - 48, 12, 96)
        self.speed = 320

    def update(self, dt: float, up: bool, down: bool) -> None:
        dy = (down - up) * self.speed
        self.rect.y += int(dy * dt)
        self.rect.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, (240, 240, 240), self.rect, border_radius=6)


class Ball:
    def __init__(self) -> None:
        self.rect = pygame.Rect(WIDTH // 2 - 10, HEIGHT // 2 - 10, 20, 20)
        self.speed = 360
        self.vx = random.choice([-1, 1])
        self.vy = random.uniform(-0.6, 0.6)

    def reset(self) -> None:
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.vx = random.choice([-1, 1])
        self.vy = random.uniform(-0.6, 0.6)

    def update(self, dt: float) -> None:
        self.rect.x += int(self.vx * self.speed * dt)
        self.rect.y += int(self.vy * self.speed * dt)
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.vy = -self.vy

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.ellipse(screen, (240, 240, 240), self.rect)


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Lesson 13 - Pong")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)

    left = Paddle(24)
    right = Paddle(WIDTH - 36)
    ball = Ball()
    score_l = 0
    score_r = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0
        keys = pygame.key.get_pressed()

        left.update(dt, up=keys[pygame.K_w], down=keys[pygame.K_s])
        right.update(dt, up=keys[pygame.K_UP], down=keys[pygame.K_DOWN])
        ball.update(dt)

        # Paddle collisions add english based on impact position
        if ball.rect.colliderect(left.rect) and ball.vx < 0:
            offset = (ball.rect.centery - left.rect.centery) / (left.rect.height / 2)
            ball.vx = abs(ball.vx)
            ball.vy = max(-1.0, min(1.0, offset))
        if ball.rect.colliderect(right.rect) and ball.vx > 0:
            offset = (ball.rect.centery - right.rect.centery) / (right.rect.height / 2)
            ball.vx = -abs(ball.vx)
            ball.vy = max(-1.0, min(1.0, offset))

        # Scoring
        if ball.rect.left <= 0:
            score_r += 1
            ball.reset()
        elif ball.rect.right >= WIDTH:
            score_l += 1
            ball.reset()

        screen.fill((18, 18, 18))
        pygame.draw.line(screen, (80, 80, 80), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
        left.draw(screen)
        right.draw(screen)
        ball.draw(screen)

        s_text = font.render(f"{score_l}   :   {score_r}", True, (230, 230, 230))
        screen.blit(s_text, (WIDTH // 2 - s_text.get_width() // 2, 16))
        tip = font.render("W/S & UP/DOWN to move | ESC to quit", True, (200, 200, 200))
        screen.blit(tip, (16, HEIGHT - 32))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


