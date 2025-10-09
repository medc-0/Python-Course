"""
06-input.py

Input Handling
--------------
Read keyboard, mouse, and optional gamepad input. Compare event-based input vs
state-based polling and show simple interactions.

What you will learn
-------------------
1) Event-based input (KEYDOWN, MOUSEBUTTONDOWN)
2) State polling (key.get_pressed, mouse.get_pressed)
3) Mouse position and dragging
4) Optional joystick/gamepad support (if connected)

Docs
----
- Events: https://www.pygame.org/docs/ref/event.html
- Keyboard: https://www.pygame.org/docs/ref/key.html
- Mouse: https://www.pygame.org/docs/ref/mouse.html
- Joystick: https://www.pygame.org/docs/ref/joystick.html
"""

import sys
import pygame  # type: ignore


def init_joystick() -> list[pygame.joystick.Joystick]:
    pygame.joystick.init()
    joysticks: list[pygame.joystick.Joystick] = []
    for i in range(pygame.joystick.get_count()):
        js = pygame.joystick.Joystick(i)
        js.init()
        joysticks.append(js)
    return joysticks


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 06 - Input Handling")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    joysticks = init_joystick()
    dragging = False
    drag_rect = pygame.Rect(340, 200, 120, 80)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if drag_rect.collidepoint(event.pos):
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

        # Polling states
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()

        # Dragging behavior
        if dragging and mouse_buttons[0]:
            drag_rect.center = mouse_pos

        # Joystick axes/buttons (if present)
        js_info = "No joystick"
        if joysticks:
            js = joysticks[0]
            axes = [js.get_axis(i) for i in range(js.get_numaxes())]
            btns = [js.get_button(i) for i in range(js.get_numbuttons())]
            js_info = f"Joystick axes: {[round(a,2) for a in axes]}  buttons: {btns}"

        # Draw
        screen.fill((250, 250, 250))
        pygame.draw.rect(screen, (80, 170, 240), drag_rect, border_radius=10)
        pygame.draw.circle(screen, (240, 120, 80), mouse_pos, 6)

        # HUD text
        lines = [
            "Events vs polling: BOTH are useful",
            f"Dragging: {'YES' if dragging else 'no'}  |  Left mouse pressed: {mouse_buttons[0]}",
            f"Mouse pos: {mouse_pos}",
            f"Keys held: LEFT={keys[pygame.K_LEFT]} RIGHT={keys[pygame.K_RIGHT]} UP={keys[pygame.K_UP]} DOWN={keys[pygame.K_DOWN]}",
            js_info,
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


