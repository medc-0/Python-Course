"""
04-movement_animation.py

Movement and Animation
----------------------
Control a player with the keyboard and animate frames over time.

What you will learn
-------------------
1) Reading continuous keyboard state with `pygame.key.get_pressed()`
2) Implementing velocity-based movement with dt for smooth motion
3) Playing a looping frame animation
4) Keeping entities within the screen bounds

Docs
----
- Keyboard: https://www.pygame.org/docs/ref/key.html
- Time/Clock: https://www.pygame.org/docs/ref/time.html
"""

import sys
import pygame  # type: ignore


def make_frame(color: tuple[int, int, int]) -> pygame.Surface:
    surf = pygame.Surface((48, 48), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pygame.draw.rect(surf, color, (4, 4, 40, 40), border_radius=8)
    pygame.draw.rect(surf, (0, 0, 0), (4, 4, 40, 40), width=2, border_radius=8)
    return surf


class Player(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int]):
        super().__init__()
        # Four frames that cycle to simulate stepping
        self.frames = [
            make_frame((80, 170, 240)),
            make_frame((80, 150, 230)),
            make_frame((80, 130, 220)),
            make_frame((80, 150, 230)),
        ]
        self.frame_index = 0
        self.image = self.frames[self.frame_index]
        self.rect = self.image.get_rect(center=pos)
        self.speed = 240.0  # pixels/sec
        self.anim_timer = 0.0
        self.anim_interval = 0.12

    def update(self, dt: float, bounds: pygame.Rect) -> None:
        keys = pygame.key.get_pressed()
        vx = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(keys[pygame.K_LEFT] or keys[pygame.K_a])
        vy = float(keys[pygame.K_DOWN] or keys[pygame.K_s]) - float(keys[pygame.K_UP] or keys[pygame.K_w])

        # Normalize diagonal
        length = (vx * vx + vy * vy) ** 0.5
        if length > 0:
            vx /= length
            vy /= length

        self.rect.x += int(vx * self.speed * dt)
        self.rect.y += int(vy * self.speed * dt)

        # Clamp to screen
        self.rect.clamp_ip(bounds)

        # Animate while moving
        moving = length > 0
        self.anim_timer += dt if moving else 0.0
        if moving and self.anim_timer >= self.anim_interval:
            self.anim_timer = 0.0
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.image = self.frames[self.frame_index]


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 04 - Movement & Animation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    player = Player((400, 240))
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0
        bounds = screen.get_rect()
        player.update(dt, bounds)

        screen.fill((245, 245, 245))

        # Draw a soft grid background
        grid = (225, 225, 225)
        for x in range(0, bounds.width, 20):
            pygame.draw.line(screen, grid, (x, 0), (x, bounds.height))
        for y in range(0, bounds.height, 20):
            pygame.draw.line(screen, grid, (0, y), (bounds.width, y))

        screen.blit(player.image, player.rect)

        hud1 = font.render("Move with WASD/Arrows", True, (30, 30, 30))
        hud2 = font.render("Animation plays only while moving", True, (30, 30, 30))
        screen.blit(hud1, (16, 16))
        screen.blit(hud2, (16, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


