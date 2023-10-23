import pygame

from game.Constants import VERTICAL_SPACE


class Pipe:
    def __init__(self, screen, x, y, pipe_type):
        self.image1 = pygame.image.load(f"game/images/pipe-{pipe_type}-down.png")
        self.image2 = pygame.image.load(f"game/images/pipe-{pipe_type}-up.png")
        self.x, self.y = x, y
        self.speed_x = 2
        self.space = VERTICAL_SPACE
        self.screen = screen

    def render(self):
        self.screen.blit(self.image1, (self.x, self.y - self.image1.get_height()))
        self.screen.blit(self.image2, (self.x, self.y + self.space))

    def move(self):
        self.x -= self.speed_x

    def get_width(self):
        return self.image1.get_width()

    def get_height(self):
        return self.image1.get_height()
