"""
09-sprite_groups.py

Sprite Groups
-------------
Manage many sprites efficiently with `pygame.sprite.Group`, batch update/draw,
and group-level collision checks.

What you will learn
-------------------
1) Organizing sprites into layers/groups
2) Calling update()/draw() on groups
3) Using groupcollide and spritecollide

Docs
----
- Sprite Groups: https://www.pygame.org/docs/ref/sprite.html
"""

import sys
import random
import pygame  # type: ignore


class Bullet(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int]):
        super().__init__()
        self.image = pygame.Surface((8, 8), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255, 255, 210), (4, 4), 4)
        self.rect = self.image.get_rect(center=pos)
        self.speed = 500

    def update(self, dt: float) -> None:
        self.rect.x += int(self.speed * dt)
        if self.rect.left > 800:
            self.kill()


class Enemy(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int]):
        super().__init__()
        self.image = pygame.Surface((24, 24), pygame.SRCALPHA)
        pygame.draw.rect(self.image, (240, 110, 110), (0, 0, 24, 24), border_radius=6)
        self.rect = self.image.get_rect(center=pos)
        self.speed = random.randint(60, 120)

    def update(self, dt: float) -> None:
        self.rect.x -= int(self.speed * dt)
        if self.rect.right < 0:
            self.kill()


class Player(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int]):
        super().__init__()
        self.image = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.rect(self.image, (80, 170, 240), (0, 0, 32, 32), border_radius=6)
        self.rect = self.image.get_rect(center=pos)
        self.cooldown = 0.0
        self.rate = 0.2

    def update(self, dt: float, bounds: pygame.Rect) -> None:
        keys = pygame.key.get_pressed()
        dy = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * 260
        self.rect.y += int(dy * dt)
        self.rect.clamp_ip(bounds)
        self.cooldown = max(0.0, self.cooldown - dt)

    def can_shoot(self) -> bool:
        return self.cooldown <= 0.0

    def shoot(self) -> Bullet:
        self.cooldown = self.rate
        return Bullet(self.rect.midright)


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 09 - Sprite Groups")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    all_sprites = pygame.sprite.Group()
    enemies = pygame.sprite.Group()
    bullets = pygame.sprite.Group()

    player = Player((120, 240))
    all_sprites.add(player)

    spawn_timer = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.get_time() / 1000.0

        # Spawning enemies
        spawn_timer += dt
        if spawn_timer >= 0.5:
            spawn_timer = 0.0
            y = random.randint(24, 456)
            e = Enemy((800 + 24, y))
            enemies.add(e)
            all_sprites.add(e)

        # Player update and shooting
        player.update(dt, screen.get_rect())
        if pygame.key.get_pressed()[pygame.K_SPACE] and player.can_shoot():
            b = player.shoot()
            bullets.add(b)
            all_sprites.add(b)

        # Update bullets/enemies
        for s in list(bullets) + list(enemies):
            s.update(dt)  # type: ignore[arg-type]

        # Collisions: bullets vs enemies
        hits = pygame.sprite.groupcollide(bullets, enemies, True, True)

        # Draw
        screen.fill((18, 18, 18))
        all_sprites.draw(screen)

        hud = [
            f"Enemies: {len(enemies)}  Bullets: {len(bullets)}  Hits: {sum(len(v) for v in hits.values())}",
            "SPACE to shoot, UP/DOWN to move",
        ]
        for i, text in enumerate(hud):
            lbl = font.render(text, True, (230, 230, 230))
            screen.blit(lbl, (16, 16 + 22 * i))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


