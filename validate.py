import sys
from glob import glob

import pygame
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from game.Constants import h, w
from game.MainGame import MainGame
from model.DeepQNetwork import DeepQNetwork
from utils import get_configs

if __name__ == "__main__":
    configs = get_configs()

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Flappy Bird Auto - Validation")
    fps_clock = pygame.time.Clock()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and configs.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")
    writer = SummaryWriter(
        f"{configs.training.log_dir}/{configs.training.checkpoint_name}/val"
    )

    main_game = MainGame(screen, **configs.game)
    model = DeepQNetwork(**configs.model).to(device)

    print(f"The model has {sum(p.numel() for p in model.parameters()):,} params.")

    max_reward = 0
    best_checkpoint = None
    checkpoint_list = sorted(
        glob(
            f"{configs.training.save_dir}/{configs.training.checkpoint_name}/{configs.training.checkpoint_name}*.pth"
        )
    )

    for checkpoint in tqdm(checkpoint_list):
        with open(
            checkpoint,
            "rb",
        ) as f:
            cp = torch.load(f, map_location=device)

        model.load_state_dict(cp["q_model_state_dict"])
        print(f"Checkpoint loaded at step {cp['step'] + 1}.")

        model.eval()
        last_reward = 0
        state, *_ = main_game.update()
        state = torch.cat(
            [state for _ in range(configs.model.n_temp_frames)]
        ).unsqueeze(0)

        while not any([event.type == pygame.QUIT for event in pygame.event.get()]):
            with torch.inference_mode():
                action = model(state.to(device))[0].argmax().item()
            next_state, reward, _ = main_game.update(action)

            if reward > max_reward:
                max_reward = reward
                best_checkpoint = checkpoint

            if reward == -10:
                writer.add_scalar("max_reward", last_reward, cp["step"] + 1)
                break
            else:
                last_reward = reward

            print(f"\rAction: {action:2}, Reward: {reward:6}", end="", flush=True)
            state = torch.cat((state[:, 1:, :, :], next_state.unsqueeze(0)), dim=1)

            pygame.display.update()
            fps_clock.tick(configs.fps)
        print()

    print(f"Best checkpoint: {best_checkpoint}, Reward: {max_reward}")
    pygame.quit()
    sys.exit(0)