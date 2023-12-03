import math

import pygame

from game.Constants import h, w


class Bird:
    def __init__(self, screen, bird_type, base_height):
        self.screen = screen
        self.bird_images = [
            pygame.image.load(f"game/images/{bird_type}-midflap.png"),
            pygame.image.load(f"game/images/{bird_type}-upflap.png"),
            pygame.image.load(f"game/images/{bird_type}-downflap.png"),
        ]
        self.x = 0
        self.y = 0
        self.reset_coordinates()
        self.frame = 0
        self.keep_flapping = True
        self.g = 0.98
        self.speed = 5
        self.drop_limit = h - self.get_height() / 2 - base_height
        self.angle = 0

    def reset_coordinates(self):
        self.x = w / 3 - self.get_width() / 3
        self.y = h / 2 + 50

    def get_width(self):
        return self.bird_images[0].get_width()

    def get_height(self):
        return self.bird_images[0].get_height()

    def render(self):
        rotated = pygame.transform.rotozoom(
            self.bird_images[self.frame], math.degrees(-self.angle), 1.0
        )
        rotated.get_rect().center = (self.x, self.y)
        self.screen.blit(
            rotated,
            (self.x - rotated.get_width() / 2, self.y - rotated.get_height() / 2),
        )

    def move(self):
        if self.keep_flapping:
            self._flap()
        else:
            self._keep_dropping()

    def _keep_dropping(self):
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
            self._flap()

    def _flap(self):
        self.frame += 1
        if self.frame > 2:
            self.frame = 0
