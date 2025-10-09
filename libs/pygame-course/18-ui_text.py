"""
18-ui_text.py

UI and Text
-----------
Render text with fonts, draw buttons, and handle basic UI interactions.
"""

import sys
import pygame  # type: ignore


class Button:
    def __init__(self, rect: pygame.Rect, text: str, font: pygame.font.Font) -> None:
        self.rect = rect
        self.text = text
        self.font = font
        self.hover = False
        self.pressed = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed and self.rect.collidepoint(event.pos)
            self.pressed = False
            return was_pressed
        return False

    def draw(self, screen: pygame.Surface) -> None:
        bg = (80, 170, 240) if self.hover else (60, 140, 210)
        pygame.draw.rect(screen, bg, self.rect, border_radius=8)
        pygame.draw.rect(screen, (20, 40, 60), self.rect, width=2, border_radius=8)
        label = self.font.render(self.text, True, (255, 255, 255))
        screen.blit(label, (self.rect.centerx - label.get_width() // 2, self.rect.centery - label.get_height() // 2))


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 18 - UI & Text")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    buttons = [
        Button(pygame.Rect(100, 120, 200, 52), "Play", font),
        Button(pygame.Rect(100, 190, 200, 52), "Options", font),
        Button(pygame.Rect(100, 260, 200, 52), "Quit", font),
    ]
    message = "Click a button"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            else:
                for b in buttons:
                    if b.handle_event(event):
                        if b.text == "Quit":
                            running = False
                        else:
                            message = f"Pressed: {b.text}"

        screen.fill((245, 245, 245))
        title = font.render("UI: Hover + Click detection", True, (30, 30, 30))
        screen.blit(title, (16, 16))

        for b in buttons:
            b.draw(screen)

        msg = font.render(message, True, (30, 30, 30))
        screen.blit(msg, (100, 340))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


