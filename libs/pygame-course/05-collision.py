"""
05-collision.py

Collision Detection
-------------------
Demonstrates rectangle collisions with sprites, circle collisions via distance,
and point collisions with the mouse.

What you will learn
-------------------
1) Using `rect.colliderect` for AABB collision
2) Checking circle vs circle with distance
3) Using `collidepoint` to test hits with the mouse
4) Visualizing collision states

Docs
----
- Rect: https://www.pygame.org/docs/ref/rect.html
- Mouse: https://www.pygame.org/docs/ref/mouse.html
"""

import sys
import math
import pygame  # type: ignore


def circle_collision(c1: tuple[int, int], r1: int, c2: tuple[int, int], r2: int) -> bool:
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return (dx * dx + dy * dy) ** 0.5 < (r1 + r2)


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 05 - Collision Detection")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    # Rectangles
    player = pygame.Rect(120, 200, 80, 56)
    obstacle = pygame.Rect(420, 220, 120, 90)
    speed = 240.0

    # Circles
    c1_pos = [220, 120]
    c2_pos = [520, 120]
    c1_r, c2_r = 28, 40

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0
        keys = pygame.key.get_pressed()

        # Move player rect
        vx = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        vy = float(keys[pygame.K_DOWN]) - float(keys[pygame.K_UP])
        length = (vx * vx + vy * vy) ** 0.5
        if length > 0:
            vx /= length
            vy /= length
        player.x += int(vx * speed * dt)
        player.y += int(vy * speed * dt)
        player.clamp_ip(screen.get_rect())

        # Move circle 1 with WASD
        vx2 = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        vy2 = float(keys[pygame.K_s]) - float(keys[pygame.K_w])
        length2 = (vx2 * vx2 + vy2 * vy2) ** 0.5
        if length2 > 0:
            vx2 /= length2
            vy2 /= length2
        c1_pos[0] += int(vx2 * speed * dt)
        c1_pos[1] += int(vy2 * speed * dt)

        # Mouse point collision
        mouse_pos = pygame.mouse.get_pos()
        mouse_hits_player = player.collidepoint(mouse_pos)

        # Rect-vs-Rect
        aabb_hit = player.colliderect(obstacle)

        # Circle-vs-Circle
        circ_hit = circle_collision(tuple(c1_pos), c1_r, tuple(c2_pos), c2_r)

        # Draw
        screen.fill((248, 248, 248))

        # Rect visuals
        pygame.draw.rect(screen, (80, 170, 240), player, border_radius=8)
        pygame.draw.rect(screen, (240, 120, 80) if aabb_hit else (180, 180, 180), obstacle, width=3, border_radius=8)

        # Circle visuals
        pygame.draw.circle(screen, (120, 200, 120), tuple(c1_pos), c1_r)
        pygame.draw.circle(screen, (240, 120, 120) if circ_hit else (160, 160, 160), tuple(c2_pos), c2_r, width=3)

        # Mouse hit marker
        color = (220, 60, 60) if mouse_hits_player else (60, 60, 60)
        pygame.draw.circle(screen, color, mouse_pos, 3)

        # Labels
        lines = [
            "Arrows: move blue rect  |  WASD: move green circle",
            f"AABB collision: {'HIT' if aabb_hit else 'no'}",
            f"Circle collision: {'HIT' if circ_hit else 'no'}",
            f"Mouse over player: {'YES' if mouse_hits_player else 'no'}",
        ]
        for i, text in enumerate(lines):
            lbl = font.render(text, True, (30, 30, 30))
            screen.blit(lbl, (16, 16 + 22 * i))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


