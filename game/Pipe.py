import pygame

from game.Constants import VERTICAL_SPACE


class Pipe:
    def __init__(self, main_game, x, y):
        self.main_game = main_game
        self.type_of_pipes = ["pipe-green", "pipe-red"]
        self.image1 = pygame.image.load("game/images/{}-down.png".format(self.type_of_pipes[main_game.pipe_type]))
        self.image2 = pygame.image.load("game/images/{}-up.png".format(self.type_of_pipes[main_game.pipe_type]))
        self.x, self.y = x, y
        self.speed_x = 2
        self.space = VERTICAL_SPACE
        self.screen = main_game.screen

    def render(self):
        self.screen.blit(self.image1, (self.x, self.y - self.image1.get_height()))
        self.screen.blit(self.image2, (self.x, self.y + self.space))

    def move(self):
        self.x -= self.speed_x

    def get_width(self):
        return self.image1.get_width()

    def get_height(self):
        return self.image1.get_height()
