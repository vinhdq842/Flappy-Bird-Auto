import pygame
from pygame import mixer

pygame.mixer.init(44100, -16, 2, 2048)


class SoundPlayer:
    def __init__(self):
        (
            self.hit_sound,
            self.point_sound,
            self.swoosh_sound,
            self.wing_sound,
            self.die_sound,
        ) = (Sound(i) for i in range(0, 5))


class Sound:
    sounds = [
        mixer.Sound("game/sounds/hit.wav"),
        mixer.Sound("game/sounds/point.wav"),
        mixer.Sound("game/sounds/swoosh.wav"),
        mixer.Sound("game/sounds/wing.wav"),
        mixer.Sound("game/sounds/die.wav"),
    ]

    def __init__(self, channel):
        self.channel = channel
        self.sound = mixer.Channel(channel)
        self.sound.set_volume(0.2)

    def play_sound(self, loop=False):
        if loop:
            self.sound.play(Sound.sounds[self.channel], -1)
        else:
            self.sound.play(Sound.sounds[self.channel])

    def is_playing(self):
        return self.sound.get_busy()

    def stop_sound(self):
        self.sound.stop()

    def pause_sound(self):
        self.sound.pause()

    def unpause_sound(self):
        self.sound.unpause()
