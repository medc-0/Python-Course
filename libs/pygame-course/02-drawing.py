"""
02-drawing.py

Drawing and Colors
------------------
In this lesson you will learn how to draw shapes, lines, and text with Pygame,
how width affects outlines vs filled shapes, how to use anti-aliased lines, and
how to work with color utilities.

Topics covered
--------------
1) Screen clearing and draw order
2) Colors (RGB, alpha) and helper functions
3) Rectangles, circles, lines, polygons, ellipses, arcs
4) Border radius and outline width
5) Rendering simple text labels to annotate drawings

Docs
----
- Drawing: https://www.pygame.org/docs/ref/draw.html
- Surfaces: https://www.pygame.org/docs/ref/surface.html
- Fonts: https://www.pygame.org/docs/ref/font.html
"""

import sys
import math
import pygame  # type: ignore


def clamp_byte(value: int) -> int:
    return max(0, min(255, value))


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        clamp_byte(int(a[0] + (b[0] - a[0]) * t)),
        clamp_byte(int(a[1] + (b[1] - a[1]) * t)),
        clamp_byte(int(a[2] + (b[2] - a[2]) * t)),
    )


def initialize(width: int = 800, height: int = 480) -> tuple[pygame.Surface, pygame.time.Clock, pygame.font.Font]:
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pygame Lesson 02 - Drawing")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    return screen, clock, font


def draw_primitives(screen: pygame.Surface, font: pygame.font.Font, t: float) -> None:
    screen.fill((250, 250, 250))
    w, h = screen.get_size()

    # Title
    title = font.render("Primitives: rect/circle/line/polygon/ellipse/arc", True, (30, 30, 30))
    screen.blit(title, (16, 12))

    # Rectangle: filled and outlined with border radius
    rect_color = (70, 140, 220)
    outline = (20, 40, 60)
    pygame.draw.rect(screen, rect_color, (40, 60, 160, 80), border_radius=12)
    pygame.draw.rect(screen, outline, (40, 60, 160, 80), width=2, border_radius=12)

    # Circle: animate radius a bit
    radius = 30 + int(12 * math.sin(t * 2.0))
    pygame.draw.circle(screen, (220, 90, 90), (280, 100), radius)
    pygame.draw.circle(screen, outline, (280, 100), radius, width=2)

    # Line: anti-aliased diagonal line and a thick line
    start = (360, 60)
    end = (520, 140)
    pygame.draw.aaline(screen, (60, 60, 60), start, end)
    pygame.draw.line(screen, (0, 180, 120), (360, 140), (520, 60), width=6)

    # Polygon: simple triangle with outline
    points = [(600, 140), (560, 60), (640, 60)]
    pygame.draw.polygon(screen, (240, 200, 80), points)
    pygame.draw.polygon(screen, outline, points, width=2)

    # Ellipse inside a bounding rect
    pygame.draw.ellipse(screen, (120, 90, 200), (40, 180, 160, 80))
    pygame.draw.ellipse(screen, outline, (40, 180, 160, 80), width=2)

    # Arc: quarter-circle arc with thickness
    pygame.draw.arc(screen, (200, 120, 60), (240, 180, 120, 120), 0, math.pi / 2, width=4)

    # Grid to illustrate draw order (drawn last, so on top)
    grid_color = (220, 220, 220)
    for gx in range(0, w, 20):
        pygame.draw.line(screen, grid_color, (gx, 0), (gx, h))
    for gy in range(0, h, 20):
        pygame.draw.line(screen, grid_color, (0, gy), (w, gy))

    # Legend
    legend_lines = [
        "Filled vs outlined shapes (width=0 vs width>0)",
        "Anti-aliased line (aaline)",
        "Border radius on rects",
        "Draw order matters: later draws appear on top",
    ]
    for i, text in enumerate(legend_lines):
        lbl = font.render(text, True, (50, 50, 50))
        screen.blit(lbl, (16, 280 + i * 20))


def draw_color_ramp(screen: pygame.Surface, font: pygame.font.Font, t: float) -> None:
    # A smooth color ramp that animates between two colors
    w, _ = screen.get_size()
    left = (255, 128, 0)
    right = (0, 128, 255)
    phase = 0.5 + 0.5 * math.sin(t)
    for x in range(0, w, 4):
        c = lerp_color(left, right, (x / max(1, w - 1)) * 0.7 + phase * 0.3)
        pygame.draw.line(screen, c, (x, 360), (x, 430))
    cap = font.render("Animated color ramp (lerp between colors)", True, (40, 40, 40))
    screen.blit(cap, (16, 336))


def main() -> int:
    screen, clock, font = initialize()
    running = True
    t = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Time advance (seconds)
        dt = clock.get_time() / 1000.0
        t += dt

        draw_primitives(screen, font, t)
        draw_color_ramp(screen, font, t)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
