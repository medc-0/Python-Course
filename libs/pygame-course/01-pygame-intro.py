"""
01-pygame-intro.py

Introduction to Pygame
----------------------
Pygame is a popular library for building 2D games in Python. It gives you
building blocks for windows, drawing, input, timing, images, and sound.

What you will learn in this lesson
----------------------------------
1) How to open a window and handle the main loop
2) How the event system works (quit, key, mouse)
3) How to draw basic shapes and text each frame
4) How to control the frame rate with a clock
5) How to exit cleanly without freezing

Useful docs
-----------
- Pygame Website: https://www.pygame.org
- Getting Started: https://www.pygame.org/wiki/GettingStarted
- Display/Window: https://www.pygame.org/docs/ref/display.html
- Event Queue: https://www.pygame.org/docs/ref/event.html
- Time/Clock: https://www.pygame.org/docs/ref/time.html

Tips
----
- Keep your game loop structured: handle events → update → draw → flip → tick
- Never block the loop with input(); always use events or state flags
- Control frame rate to keep CPU usage reasonable and animations smooth
"""

import sys
import pygame  # type: ignore


def initialize_pygame_window(width: int = 640, height: int = 360) -> tuple[pygame.Surface, pygame.time.Clock]:
    """Initialize pygame, create the window, set caption/icon, and return (screen, clock)."""
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pygame Intro - Lesson 01")
    # Optional: set a simple window icon color surface
    icon_surface = pygame.Surface((32, 32))
    icon_surface.fill((30, 144, 255))  # DodgerBlue
    pygame.display.set_icon(icon_surface)
    clock = pygame.time.Clock()
    return screen, clock


def handle_events(running: bool, bg_color: list[int]) -> tuple[bool, list[int]]:
    """Process the event queue. Return updated running flag and background color.

    Controls in this demo:
    - Close window or press ESC to quit
    - Press SPACE to toggle background between light/dark
    - Click left mouse to flash a background highlight once
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                # Toggle background theme
                if bg_color == [245, 245, 245]:
                    bg_color = [18, 18, 18]
                else:
                    bg_color = [245, 245, 245]
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Briefly flash a lighter/darker variant for feedback
            delta = -20 if sum(bg_color) > 128 * 3 else 20
            bg_color = [max(0, min(255, c + delta)) for c in bg_color]
    return running, bg_color


def draw_demo(screen: pygame.Surface, font: pygame.font.Font, frame_count: int, bg_color: list[int]) -> None:
    """Draw the scene for the current frame.

    We draw:
    - Background color
    - A moving rectangle
    - A static circle
    - Some text with FPS and instructions
    """
    screen.fill(bg_color)

    # Compute positions using frame_count for simple motion
    width, height = screen.get_size()
    rect_width, rect_height = 120, 40
    rect_x = (frame_count % (width + rect_width)) - rect_width
    rect_y = height // 3

    # Draw a moving rectangle (accent color depends on theme)
    is_dark = sum(bg_color) < 128 * 3
    rect_color = (0, 200, 130) if is_dark else (0, 120, 90)
    pygame.draw.rect(screen, rect_color, (rect_x, rect_y, rect_width, rect_height), border_radius=8)

    # Draw a fixed circle
    circle_color = (220, 20, 60) if is_dark else (200, 0, 50)
    pygame.draw.circle(screen, circle_color, (width // 2, height // 2 + 40), 28)

    # Draw outlines to hint at draw order
    outline_color = (255, 255, 255) if is_dark else (40, 40, 40)
    pygame.draw.rect(screen, outline_color, (20, 20, width - 40, height - 40), width=1, border_radius=12)

    # Render text showing FPS and key hints
    heading_surface = font.render("Pygame Intro", True, (255, 255, 255) if is_dark else (0, 0, 0))
    hint_surface = font.render("ESC: Quit  |  SPACE: Toggle Theme  |  Click: Flash", True,
                               (200, 200, 200) if is_dark else (30, 30, 30))

    screen.blit(heading_surface, (24, 20))
    screen.blit(hint_surface, (24, 50))


def main() -> int:
    screen, clock = initialize_pygame_window(720, 420)
    # Use a default font; for production pick a readable font file
    font = pygame.font.SysFont(None, 22)

    running = True
    frame_count = 0
    # Start with a light background; toggled with SPACE
    bg_color = [245, 245, 245]

    # Game loop: events → update (none here) → draw → flip → tick
    while running:
        running, bg_color = handle_events(running, bg_color)

        # Update step: in this intro we derive motion from frame_count directly
        frame_count += 1

        # Draw everything for the current frame
        draw_demo(screen, font, frame_count, bg_color)

        # Flip the back buffer to the screen
        pygame.display.flip()

        # Limit to 60 frames per second to keep timing stable
        clock.tick(60)

    # Clean shutdown: ensure mixer and modules quit to avoid hangs on some OSes
    pygame.quit()
    # Use sys.exit with code 0 for success in some launchers
    return 0


if __name__ == "__main__":
    # Running as a script: start the demo
    sys.exit(main())
