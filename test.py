import sys
from os.path import isfile

import pygame
import torch

from configs import get_configs
from game.Constants import h, w
from game.MainGame import MainGame
from model.DeepQNetwork import DeepQNetwork

if __name__ == "__main__":
    configs = get_configs()

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Flappy Bird Auto")
    fps_clock = pygame.time.Clock()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and configs.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    main_game = MainGame(screen, allow_sound=True, **configs.game.to_dict())
    model = DeepQNetwork(**configs.model.to_dict()).to(device)

    print(f"The model has {sum(p.numel() for p in model.parameters()):,} params.")

    assert isfile(
        f"{configs.training.save_dir}/{configs.training.checkpoint_name}/{configs.test.best_checkpoint}.pth"
    ), "Please provide a best_checkpoint to run."

    with open(
        f"{configs.training.save_dir}/{configs.training.checkpoint_name}/{configs.test.best_checkpoint}.pth",
        "rb",
    ) as f:
        cp = torch.load(f, map_location=device)

    model.load_state_dict(cp["q_model_state_dict"])
    print(f"Checkpoint loaded at step {cp['step'] + 1}.")

    model.eval()

    _, state, *_ = main_game.update()
    state = torch.cat([state for _ in range(configs.model.n_temp_frames)]).unsqueeze(0)

    while not any([event.type == pygame.QUIT for event in pygame.event.get()]):
        with torch.inference_mode():
            action = model(state.to(device))[0].argmax().item()
        point, next_state, reward, _ = main_game.update(action)
        if reward == -1:
            break
        print(f"\rAction: {action:2}, Point: {point:6}", end="", flush=True)
        state = torch.cat((state[:, 1:, :, :], next_state.unsqueeze(0)), dim=1)

        pygame.display.update()
        fps_clock.tick(configs.fps)
    print()

    pygame.quit()
    sys.exit(0)
