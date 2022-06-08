import pygame


class Background:
    def __init__(self, main_game):
        self.bg = pygame.image.load("game/images/background-day.png") if main_game.bg_type == 0 else pygame.image.load(
            "game/images/background-night.png")
        self.screen = main_game.screen

    def render(self):
        self.screen.blit(self.bg, (0, 0))
