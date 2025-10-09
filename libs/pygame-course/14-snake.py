"""
14-snake.py

Snake - Grid Game
-----------------
Implement the classic Snake with grid movement, food spawning, growth,
and game over conditions.
"""

import sys
import random
import pygame  # type: ignore


TILE = 20
COLS, ROWS = 40, 24
WIDTH, HEIGHT = COLS * TILE, ROWS * TILE


class Snake:
    def __init__(self) -> None:
        self.body: list[tuple[int, int]] = [(COLS // 2, ROWS // 2)]
        self.dir = (1, 0)
        self.grow = 0

    def change_dir(self, d: tuple[int, int]) -> None:
        # Prevent reversing directly
        if (d[0] == -self.dir[0] and d[1] == -self.dir[1]):
            return
        self.dir = d

    def update(self) -> bool:
        head = self.body[0]
        new_head = (head[0] + self.dir[0], head[1] + self.dir[1])
        # Wrap around
        new_head = (new_head[0] % COLS, new_head[1] % ROWS)
        if new_head in self.body:
            return False
        self.body.insert(0, new_head)
        if self.grow > 0:
            self.grow -= 1
        else:
            self.body.pop()
        return True


def spawn_food(occupied: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        p = (random.randrange(COLS), random.randrange(ROWS))
        if p not in occupied:
            return p


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Lesson 14 - Snake")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    snake = Snake()
    food = spawn_food(set(snake.body))
    score = 0
    timer = 0.0
    step_interval = 0.12
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    snake.change_dir((1, 0))
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    snake.change_dir((-1, 0))
                elif event.key in (pygame.K_UP, pygame.K_w):
                    snake.change_dir((0, -1))
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    snake.change_dir((0, 1))

        dt = clock.get_time() / 1000.0
        timer += dt
        if timer >= step_interval:
            timer = 0.0
            alive = snake.update()
            if not alive:
                running = False
            if snake.body[0] == food:
                snake.grow += 3
                score += 10
                food = spawn_food(set(snake.body))

        screen.fill((18, 18, 18))

        # Draw grid
        for x in range(0, WIDTH, TILE):
            pygame.draw.line(screen, (35, 35, 35), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILE):
            pygame.draw.line(screen, (35, 35, 35), (0, y), (WIDTH, y))

        # Draw snake
        for i, (cx, cy) in enumerate(snake.body):
            color = (80, 170, 240) if i == 0 else (60, 130, 200)
            pygame.draw.rect(screen, color, (cx * TILE, cy * TILE, TILE, TILE))

        # Draw food
        pygame.draw.rect(screen, (240, 110, 110), (food[0] * TILE, food[1] * TILE, TILE, TILE))

        hud = font.render(f"Score: {score}", True, (230, 230, 230))
        screen.blit(hud, (10, 6))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


