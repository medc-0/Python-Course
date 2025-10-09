"""
08-game_states.py

Game States and Menus
---------------------
Create a simple state machine with Menu → Playing → Pause → GameOver states.

What you will learn
-------------------
1) Structuring states as classes with enter/exit/update/draw
2) Switching states on events/keys
3) Keeping shared game data in a central Game object

Docs
----
- Events: https://www.pygame.org/docs/ref/event.html
- Display: https://www.pygame.org/docs/ref/display.html
"""

import sys
import pygame  # type: ignore


class Game:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((800, 480))
        pygame.display.set_caption("Pygame Lesson 08 - Game States")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.running = True
        self.states: dict[str, BaseState] = {}
        self.state: BaseState | None = None
        self.score = 0

    def add_state(self, name: str, state: "BaseState") -> None:
        self.states[name] = state
        state.game = self

    def change_state(self, name: str) -> None:
        if self.state:
            self.state.exit()
        self.state = self.states[name]
        self.state.enter()

    def run(self) -> int:
        self.change_state("menu")
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    assert self.state is not None
                    self.state.handle_event(event)

            assert self.state is not None
            self.state.update()
            self.state.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        return 0


class BaseState:
    def __init__(self) -> None:
        self.game: Game | None = None

    @property
    def screen(self) -> pygame.Surface:
        assert self.game is not None
        return self.game.screen

    @property
    def font(self) -> pygame.font.Font:
        assert self.game is not None
        return self.game.font

    def enter(self) -> None:
        pass

    def exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event) -> None:
        pass

    def update(self) -> None:
        pass

    def draw(self) -> None:
        pass


class MenuState(BaseState):
    def draw(self) -> None:
        self.screen.fill((18, 18, 18))
        t1 = self.font.render("MENU - Press ENTER to play", True, (240, 240, 240))
        t2 = self.font.render("Q to quit", True, (200, 200, 200))
        self.screen.blit(t1, (40, 60))
        self.screen.blit(t2, (40, 90))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                assert self.game is not None
                self.game.change_state("playing")
            elif event.key == pygame.K_q:
                assert self.game is not None
                self.game.running = False


class PlayingState(BaseState):
    def __init__(self) -> None:
        super().__init__()
        self.player = pygame.Rect(380, 220, 40, 40)
        self.speed = 260

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            assert self.game is not None
            self.game.change_state("pause")

    def update(self) -> None:
        keys = pygame.key.get_pressed()
        vx = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        vy = float(keys[pygame.K_DOWN]) - float(keys[pygame.K_UP])
        length = max(1.0, (vx * vx + vy * vy) ** 0.5)
        self.player.x += int(vx / length * self.speed / 60)
        self.player.y += int(vy / length * self.speed / 60)
        self.player.clamp_ip(self.screen.get_rect())

    def draw(self) -> None:
        self.screen.fill((245, 245, 245))
        pygame.draw.rect(self.screen, (80, 170, 240), self.player, border_radius=6)
        hud = self.font.render("Playing - ESC to pause", True, (30, 30, 30))
        self.screen.blit(hud, (16, 16))


class PauseState(BaseState):
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                assert self.game is not None
                self.game.change_state("playing")
            elif event.key == pygame.K_m:
                assert self.game is not None
                self.game.change_state("menu")

    def draw(self) -> None:
        self.screen.fill((30, 30, 30))
        t = self.font.render("PAUSED - ESC to resume, M for menu", True, (240, 240, 240))
        self.screen.blit(t, (40, 60))


def main() -> int:
    game = Game()
    game.add_state("menu", MenuState())
    game.add_state("playing", PlayingState())
    game.add_state("pause", PauseState())
    return game.run()


if __name__ == "__main__":
    sys.exit(main())


