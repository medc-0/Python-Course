"""
17-camera_scrolling.py

Camera Scrolling
----------------
Follow the player with a camera offset to render a larger level than the screen.
"""

import sys
import pygame  # type: ignore


TILE = 32
LEVEL = [
    "........................................................................",
    "........................................................................",
    ".................#########..............................................",
    ".................#.......#..............................................",
    ".................#.......#..............................................",
    ".................#########..............................................",
    ".............................................########...................",
    ".............................................#......#...................",
    ".............................................#......#...................",
    ".............................................########...................",
    "........................................................................",
    "........................................................................",
]


def solid_tiles() -> list[pygame.Rect]:
    tiles: list[pygame.Rect] = []
    for y, row in enumerate(LEVEL):
        for x, ch in enumerate(row):
            if ch == "#":
                tiles.append(pygame.Rect(x * TILE, y * TILE, TILE, TILE))
    return tiles


class Player:
    def __init__(self, x: int, y: int) -> None:
        self.rect = pygame.Rect(x, y, 24, 28)
        self.speed = 220

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


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 17 - Camera Scrolling")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    level_rect = pygame.Rect(0, 0, len(LEVEL[0]) * TILE, len(LEVEL) * TILE)
    tiles = solid_tiles()
    player = Player(100, 100)
    camera = pygame.Vector2(0, 0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0
        player.update(dt, level_rect)

        # Center camera on player, then clamp to level bounds minus screen size
        camera.x = player.rect.centerx - screen.get_width() / 2
        camera.y = player.rect.centery - screen.get_height() / 2
        camera.x = max(0, min(camera.x, level_rect.width - screen.get_width()))
        camera.y = max(0, min(camera.y, level_rect.height - screen.get_height()))

        # Draw
        screen.fill((32, 36, 40))
        for t in tiles:
            r = t.move(-int(camera.x), -int(camera.y))
            pygame.draw.rect(screen, (70, 70, 70), r)
        pygame.draw.rect(screen, (80, 170, 240), player.rect.move(-int(camera.x), -int(camera.y)))

        tip = font.render("Camera follows player (clamped to level)", True, (230, 230, 230))
        screen.blit(tip, (8, 8))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


