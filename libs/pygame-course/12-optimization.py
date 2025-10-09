"""
12-optimization.py

Performance Optimization
------------------------
Monitor FPS, use partial display updates, and avoid unnecessary work each frame.

What you will learn
-------------------
1) Limiting frame rate and measuring FPS
2) Using dirty rects with display.update(rects)
3) Minimizing per-frame allocations and expensive transforms

Docs
----
- Display: https://www.pygame.org/docs/ref/display.html
- Time: https://www.pygame.org/docs/ref/time.html
"""

import sys
import pygame  # type: ignore


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 12 - Optimization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    # Pre-create surfaces once
    rect_surface = pygame.Surface((160, 80))
    rect_surface.fill((80, 170, 240))

    x = 40
    y = 200
    dx = 180.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0

        # Update position
        x += dx * dt
        if x > 800 - rect_surface.get_width() - 40:
            x = 800 - rect_surface.get_width() - 40
            dx = -dx
        elif x < 40:
            x = 40
            dx = -dx

        # Only redraw changed areas (dirty rects)
        screen.fill((250, 250, 250))
        rect = screen.blit(rect_surface, (int(x), y))
        fps_text = font.render(f"FPS: {clock.get_fps():.0f}", True, (30, 30, 30))
        fps_rect = screen.blit(fps_text, (16, 16))

        pygame.display.update([rect, fps_rect])
        clock.tick(120)  # test higher cap to visualize optimization

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


