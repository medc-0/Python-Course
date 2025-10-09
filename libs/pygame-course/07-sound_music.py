"""
07-sound_music.py

Sound and Music
----------------
Play short sound effects and loop background music. Handle volume, pause,
and graceful fallback when audio files are missing.

What you will learn
-------------------
1) Initializing the mixer and loading sounds/music
2) Playing one-shot sound effects
3) Looping and controlling background music
4) Basic UI for volume and pause/unpause

Docs
----
- Mixer: https://www.pygame.org/docs/ref/mixer.html
- Music: https://www.pygame.org/docs/ref/music.html
"""

import os
import sys
import pygame  # type: ignore


def safe_load_sound(path: str) -> pygame.mixer.Sound | None:
    if not os.path.exists(path):
        return None
    try:
        return pygame.mixer.Sound(path)
    except Exception:
        return None


def safe_load_music(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        pygame.mixer.music.load(path)
        return True
    except Exception:
        return False


def main() -> int:
    pygame.init()
    # Initialize mixer early for lower latency
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()

    screen = pygame.display.set_mode((800, 480))
    pygame.display.set_caption("Pygame Lesson 07 - Sound & Music")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    # Try to load assets from working directory
    click_sound = safe_load_sound("click.wav")
    blip_sound = safe_load_sound("blip.wav")
    music_loaded = safe_load_music("music.mp3") or safe_load_music("music.ogg")

    if music_loaded:
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(-1)

    volume = 0.5
    paused = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1 and click_sound:
                    click_sound.play()
                elif event.key == pygame.K_2 and blip_sound:
                    blip_sound.play()
                elif event.key == pygame.K_p:  # pause/unpause music
                    if music_loaded:
                        paused = not paused
                        pygame.mixer.music.pause() if paused else pygame.mixer.music.unpause()
                elif event.key == pygame.K_UP:
                    volume = min(1.0, volume + 0.05)
                    if music_loaded:
                        pygame.mixer.music.set_volume(volume)
                elif event.key == pygame.K_DOWN:
                    volume = max(0.0, volume - 0.05)
                    if music_loaded:
                        pygame.mixer.music.set_volume(volume)

        screen.fill((245, 245, 245))

        lines = [
            "Audio controls:",
            "1: play click.wav   2: play blip.wav",
            "P: pause/unpause music",
            "UP/DOWN: music volume",
            f"Music loaded: {'YES' if music_loaded else 'no'}  |  Volume: {volume:.2f}  |  Paused: {paused}",
            f"Click sound loaded: {'YES' if click_sound else 'no'}  |  Blip loaded: {'YES' if blip_sound else 'no'}",
            "Tip: use small uncompressed WAV for sound effects to reduce latency.",
        ]

        for i, text in enumerate(lines):
            lbl = font.render(text, True, (30, 30, 30))
            screen.blit(lbl, (16, 16 + i * 22))

        pygame.display.flip()
        clock.tick(60)

    # Stop music before quitting
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())


