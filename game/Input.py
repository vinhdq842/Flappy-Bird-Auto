import sys

import pygame


def get(cur):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                cur["UP"] = True
            if event.key == pygame.K_RETURN:
                cur["ENTER"] = True
            if event.key == pygame.K_SPACE:
                cur["SPACE"] = True
    return cur
