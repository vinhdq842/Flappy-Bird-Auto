import sys
from os.path import isfile

import pygame
import torch

from game.Constants import h, w
from game.MainGame import MainGame
from model.DeepQNetwork import DeepQNetwork

pygame.init()
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("Flappy Bird Auto")
fps_clock = pygame.time.Clock()
n_temp_frames = 4
checkpoint_path = "backup/checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    main_game = MainGame(screen)
    model = DeepQNetwork(n_temp_frames=n_temp_frames).to(device)

    print(f"The model has {sum(i.numel() for i in model.parameters()):,} params.")

    assert isfile(checkpoint_path), "Please provide a checkpoint to run."

    with open(checkpoint_path, "rb") as f:
        cp = torch.load(f, map_location=device)

    model.load_state_dict(cp["q_model_state_dict"])
    print(f"Checkpoint loaded at step {cp['iter'] + 1}.")

    model.eval()

    state, *_ = main_game.update()
    state = torch.cat([state for _ in range(n_temp_frames)]).unsqueeze(0)

    while all([event.type != pygame.QUIT for event in pygame.event.get()]):
        with torch.inference_mode():
            action = model(state.to(device))[0].argmax().item()
        next_state, *_ = main_game.update(action)
        state = torch.cat((state[:, 1:, :, :], next_state.unsqueeze(0)), dim=1)

        pygame.display.update()
        fps_clock.tick(30)

    pygame.quit()
    sys.exit(0)
