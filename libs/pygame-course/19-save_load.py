"""
19-save_load.py

Saving and Loading
------------------
Save simple game data (e.g., settings and high score) to JSON and load it back.
"""

import json
import os
import sys
import pygame  # type: ignore


SAVE_PATH = "save_data.json"


def load_data() -> dict:
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"volume": 0.8, "high_score": 0}


def save_data(data: dict) -> None:
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 19 - Save & Load")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    data = load_data()
    volume = float(data.get("volume", 0.8))
    high_score = int(data.get("high_score", 0))
    score = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    volume = min(1.0, volume + 0.05)
                elif event.key == pygame.K_DOWN:
                    volume = max(0.0, volume - 0.05)
                elif event.key == pygame.K_SPACE:
                    score += 10
                    high_score = max(high_score, score)
                elif event.key == pygame.K_r:
                    score = 0
                elif event.key == pygame.K_s:
                    data = {"volume": volume, "high_score": high_score}
                    save_data(data)

        screen.fill((245, 245, 245))
        lines = [
            "UP/DOWN: change volume  |  SPACE: +10 score  |  R: reset score",
            "S: save to JSON",
            f"Volume: {volume:.2f}  Score: {score}  High score: {high_score}",
            f"Save path: {os.path.abspath(SAVE_PATH)}",
        ]
        for i, text in enumerate(lines):
            lbl = font.render(text, True, (30, 30, 30))
            screen.blit(lbl, (16, 16 + 24 * i))

        pygame.display.flip()
        clock.tick(60)

    # Auto-save on exit
    save_data({"volume": volume, "high_score": high_score})
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


