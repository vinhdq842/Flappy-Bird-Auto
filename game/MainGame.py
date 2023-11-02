import math
import random

import cv2
import numpy as np
import pygame
import torch
from pygame.surfarray import array3d

from game.Background import Background
from game.Base import Base
from game.Bird import Bird
from game.Constants import HORIZONTAL_SPACE, NUM_PIPES, VERTICAL_SPACE, h, w
from game.NumberDrawer import NumberDrawer
from game.Pipe import Pipe
from game.SoundPlayer import SoundPlayer


class MainGame:
    def __init__(
        self,
        screen,
        allow_sound=False,
        show_background=True,
        show_point=True,
        bird_type="yellowbird",
        pipe_type="green",
        background_type="day",
    ):
        self.screen = screen
        self.allow_sound = allow_sound
        self.show_background = show_background
        self.show_point = show_point
        self.bird_type = bird_type
        self.pipe_type = pipe_type
        self.background = Background(screen, background_type)
        self.base = Base(self)
        self.number = NumberDrawer()
        self.sound_player = SoundPlayer()

        self.message = pygame.image.load("game/images/message.png")
        self.over_image = pygame.image.load("game/images/game-over.png")
        self.restart_button = pygame.image.load("game/images/restart.png")

        self.key = {"ENTER": False, "UP": False, "SPACE": False}
        self.pipes = []
        self.reset_game()

    def reset_game(self):
        # 0: Menu screen, 1: Playing, 2: Finished - press to restart
        self.game_status = 1
        self.bird = Bird(self.screen, self.bird_type, self.base.get_height())
        self.message_alpha = 1.0
        self.point = 0
        self.white_screen = False
        self.reward = 0
        self.initialize_pipe()

        if self.allow_sound:
            self.sound_player.swoosh_sound.play_sound()

    def initialize_pipe(self):
        self.pipes.clear()
        for _ in range(NUM_PIPES):
            self.add_pipe()

    def add_pipe(self):
        pipe = Pipe(
            self.screen,
            (self.pipes[-1].x if len(self.pipes) > 0 else w) + HORIZONTAL_SPACE,
            math.floor(random.choice([-1, 0, 1]) * self.bird.get_height() * 2)
            + h / 2
            - VERTICAL_SPACE / 2
            - self.base.get_height() / 2,
            self.pipe_type,
        )
        self.pipes.append(pipe)

    def update(self, key=0):
        if key != 0:
            self.key = {"ENTER": True, "UP": True, "SPACE": True}
        else:
            self.key = {"ENTER": False, "UP": False, "SPACE": False}

        done = self.move()
        self.paint()
        state, reward = pygame.display.get_surface(), self.reward
        if done:
            self.reset_game()

        return self.preprocess(array3d(state).transpose([1, 0, 2])), reward, done

    def preprocess(self, img):
        im = cv2.cvtColor(cv2.resize(img, (84, 84)), cv2.COLOR_RGB2GRAY)
        return torch.from_numpy(im[np.newaxis, :, :].astype(np.float32))

    def move(self):
        if self.key["ENTER"] or self.key["UP"] or self.key["SPACE"]:
            if self.game_status == 1:
                if self.bird.y > 0:
                    self.bird.angle = -math.pi / 8
                    self.bird.speed = -6
                    self.bird.y += self.bird.speed

                    if self.allow_sound:
                        self.sound_player.wing_sound.play_sound()

        self.key = {"ENTER": False, "UP": False, "SPACE": False}

        if self.game_status == 1:
            for pipe in self.pipes:
                pipe.move()

        self.update_pipes()

        if self.game_status < 2:
            self.base.move()

        if self.game_status == 1:
            self.update_point()

        return self.check_collision()

    def paint(self):
        if self.show_background:
            self.background.render()
        else:
            self.screen.fill((0, 0, 0))

        if self.game_status > 0:
            for pipe in self.pipes:
                pipe.render()

        self.base.render()

        if self.show_point:
            self.number.draw_number(
                str(self.point),
                w / 2 - self.number.string_width(str(self.point)) / 2,
                10,
                self.screen,
            )
        self.bird.render()

    def update_point(self):
        for pipe in self.pipes:
            check = self.bird.x + self.bird.get_width() / 2 - pipe.x

            if pipe.speed_x > check > 0:
                self.point += 1
                self.reward += 10
                if self.allow_sound:
                    self.sound_player.point_sound.play_sound()

    def update_pipes(self):
        if len(self.pipes) > 0:
            if self.pipes[0].x < -self.pipes[0].get_width():
                self.pipes.pop(0)
                self.add_pipe()

    def check_collision(self):
        for pipe in self.pipes:
            if (
                self.bird.x + self.bird.get_width() / 2 >= pipe.x
                and self.bird.x - self.bird.get_width() / 2 <= pipe.x + pipe.get_width()
                and (
                    self.bird.y - self.bird.get_height() / 2 <= pipe.y
                    or self.bird.y + self.bird.get_height() / 2 - 9
                    >= pipe.y + pipe.space
                )
            ):
                if (
                    pipe.x - self.bird.get_width() / 2 + 2
                    < self.bird.x
                    < pipe.x + pipe.get_width() - self.bird.get_width() / 2 - 2
                ):
                    if self.bird.y - self.bird.get_height() / 2 <= pipe.y:
                        self.bird.y = self.bird.get_height() / 2 + pipe.y

                    if (
                        self.bird.y + self.bird.get_height() / 2 - 9
                        >= pipe.y + pipe.space
                    ):
                        self.bird.y = -self.bird.get_height() / 2 + pipe.y + pipe.space

                self.reward = -10
                if self.allow_sound:
                    self.sound_player.hit_sound.play_sound()
                return True

        if self.bird.y >= self.bird.drop_limit:
            if self.game_status != 2:
                self.reward = -10
                if self.allow_sound:
                    self.sound_player.hit_sound.play_sound()
                return True

        return False

    def show_message(self):
        if self.message_alpha <= 0.0:
            return

        if self.message_alpha > 0.0 and self.game_status == 1:
            self.message_alpha -= (
                0.05 if self.message_alpha > 0.05 else self.message_alpha
            )

        tmp = pygame.Surface((w, h)).convert()
        tmp.blit(self.screen, (0, 0))
        tmp.blit(
            self.message,
            (
                w / 2 - self.message.get_width() / 2,
                h / 2 - self.message.get_height() / 2,
            ),
        )
        tmp.set_alpha(int(self.message_alpha * 255))
        self.screen.blit(tmp, (0, 0))

    def show_over_image(self):
        self.screen.blit(
            self.over_image, (w / 2 - self.over_image.get_width() / 2, h / 5)
        )
        self.screen.blit(
            self.restart_button, (w / 2 - self.restart_button.get_width() / 2, h / 2)
        )
