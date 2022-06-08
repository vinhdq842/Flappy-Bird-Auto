import math

import pygame

from game.Constants import w, h


class Bird:
    def __init__(self, main_game):
        self.main_game = main_game
        self.screen = main_game.screen
        self.type_of_birds = ["bluebird", "redbird", "yellowbird"]
        self.type = main_game.bird_type
        self.bird_images = [pygame.image.load("game/images/{}-midflap.png".format(self.type_of_birds[self.type])),
                            pygame.image.load("game/images/{}-upflap.png".format(self.type_of_birds[self.type])),
                            pygame.image.load("game/images/{}-downflap.png".format(self.type_of_birds[self.type]))]
        self.x = 0
        self.y = 0
        self.reset_coordinates()
        self.frame = 0
        self.keep_flapping = False

        self.g = 0.98
        self.speed = 5
        self.drop_limit = h - self.get_height() / 2 - main_game.base.get_height()
        self.angle = 0

    def reset_coordinates(self):
        self.x = w / 3 - self.get_width() / 3
        self.y = h / 2 + 50

    def get_width(self):
        return self.bird_images[0].get_width()

    def get_height(self):
        return self.bird_images[0].get_height()

    def render(self):
        rotated = pygame.transform.rotozoom(self.bird_images[self.frame], math.degrees(-self.angle), 1.0)
        rotated.get_rect().center = (self.x, self.y)
        self.screen.blit(rotated, (self.x - rotated.get_width() / 2, self.y - rotated.get_height() / 2))
        if self.keep_flapping:
            self.flap()
        else:
            self.keep_dropping()

    def keep_dropping(self):
        if self.speed > 0.0:
            self.frame = 0

        if self.y < self.drop_limit:
            self.y += self.speed
            if self.y > self.drop_limit:
                self.y = self.drop_limit

            self.speed += self.g
            if self.speed > 7:
                self.angle += math.pi / 25
                if self.angle > math.pi / 2:
                    self.angle = math.pi / 2

        if self.speed <= 0:
            self.flap()

    def flap(self):
        self.frame += 1
        if self.frame > 2:
            self.frame = 0
