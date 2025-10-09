"""
20-debug_overlay.py

Debug Overlay
-------------
Draw a lightweight debug overlay with FPS, entity counts, and toggles.
"""

import sys
import pygame  # type: ignore


class DebugOverlay:
    def __init__(self, font: pygame.font.Font) -> None:
        self.font = font
        self.visible = True
        self.lines: list[str] = []

    def set_lines(self, lines: list[str]) -> None:
        self.lines = lines

    def draw(self, screen: pygame.Surface, clock: pygame.time.Clock) -> None:
        if not self.visible:
            return
        bg = pygame.Surface((360, 140), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        screen.blit(bg, (10, 10))
        y = 16
        fps = self.font.render(f"FPS: {clock.get_fps():.0f}", True, (240, 240, 240))
        screen.blit(fps, (16, y))
        y += 22
        for line in self.lines:
            lbl = self.font.render(line, True, (200, 200, 200))
            screen.blit(lbl, (16, y))
            y += 20


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 20 - Debug Overlay")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    debug = DebugOverlay(font)
    rects = [pygame.Rect(80 + i * 60, 220, 40, 40) for i in range(8)]
    dx = [120 + (i % 3) * 40 for i in range(len(rects))]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F3:
                    debug.visible = not debug.visible

        dt = clock.get_time() / 1000.0
        for i, r in enumerate(rects):
            r.x += int(dx[i] * dt)
            if r.right > 800 or r.left < 0:
                dx[i] = -dx[i]

        screen.fill((35, 38, 43))
        for r in rects:
            pygame.draw.rect(screen, (80, 170, 240), r, border_radius=6)

        debug.set_lines([
            f"Entities: {len(rects)}",
            "Toggle overlay: F3",
        ])
        debug.draw(screen, clock)

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


