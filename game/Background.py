import pygame


class Background:
    def __init__(self, screen, background_type):
        self.bg = pygame.image.load(f"game/images/background-{background_type}.png")
        self.screen = screen

    def render(self):
        self.screen.blit(self.bg, (0, 0))
