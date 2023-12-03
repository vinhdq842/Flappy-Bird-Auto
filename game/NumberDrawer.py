import pygame


class NumberDrawer:
    def __init__(self):
        self.number_images = [
            pygame.image.load(f"game/images/{i}.png") for i in range(10)
        ]

    def draw(self, number, x, y, screen):
        for i in range(len(number)):
            x += self.number_images[int(number[i - 1])].get_width() + 2 if i > 0 else 0
            screen.blit(self.number_images[int(number[i])], (x, y))

    def char_width(self, char):
        return self.number_images[char].get_width()

    def string_width(self, string):
        rs = 0
        for i in string:
            rs += self.number_images[int(i)].get_width() + 2

        return rs - 2
