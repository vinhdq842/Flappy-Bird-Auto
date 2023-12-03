import pygame

from game.Constants import h


class Base:
    def __init__(self, screen):
        self.x = 0
        self.image = pygame.image.load("game/images/base.png")
        self.speed_x = 2
        self.screen = screen

    def render(self):
        self.screen.blit(self.image, (self.x, h - self.get_height()))
        self.screen.blit(self.image, (self.x + self.get_width(), h - self.get_height()))

    def get_width(self):
        return self.image.get_width()

    def get_height(self):
        return self.image.get_height()

    def move(self, game_status):
        if game_status < 2:
            self.x -= self.speed_x

        if self.x == -self.get_width():
            self.x = 0
