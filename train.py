import os
import pickle
import random
from glob import glob
from os.path import isfile

import numpy as np
import pygame
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import get_configs
from game.Constants import h, w
from game.MainGame import MainGame
from model.DeepQNetwork import DeepQNetwork

if __name__ == "__main__":
    configs = get_configs()

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Flappy Bird Auto - Training")

    os.makedirs(
        f"{configs.training.save_dir}/{configs.training.checkpoint_name}", exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and configs.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")
    writer = SummaryWriter(
        f"{configs.training.log_dir}/{configs.training.checkpoint_name}"
    )

    torch.manual_seed(configs.training.seed)
    np.random.seed(configs.training.seed)
    random.seed(configs.training.seed)
    # for CNNs
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    main_game = MainGame(screen, **configs.game.to_dict())
    q_model = DeepQNetwork(**configs.model.to_dict()).to(device)
    t_model = DeepQNetwork(**configs.model.to_dict()).to(device)

    print(f"The model has {sum(p.numel() for p in q_model.parameters()):,} params.")

    step = 0
    optimizer = torch.optim.Adam(q_model.parameters(), lr=configs.training.lr)

    checkpoint_list = sorted(
        glob(
            f"{configs.training.save_dir}/{configs.training.checkpoint_name}/{configs.training.checkpoint_name}*.pth"
        )
    )
    latest_checkpoint = checkpoint_list[-1] if len(checkpoint_list) else None

    if latest_checkpoint and isfile(latest_checkpoint):
        with open(latest_checkpoint, "rb") as f:
            cp = torch.load(f, map_location=device)

        step = cp["step"] + 1
        q_model.load_state_dict(cp["q_model_state_dict"])
        t_model.load_state_dict(cp["q_model_state_dict"])
        optimizer.load_state_dict(cp["opt_state_dict"])

        print(f"Checkpoint loaded at step {step}.")

    exp_replay = []
    if isfile(
        f"{configs.training.save_dir}/{configs.training.checkpoint_name}/replay_memory.pth"
    ):
        with open(
            f"{configs.training.save_dir}/{configs.training.checkpoint_name}/replay_memory.pth",
            "rb",
        ) as f:
            exp_replay = pickle.load(f)

    q_model.train()
    t_model.eval()
    loss_fn = nn.MSELoss()

    _, state, *_ = main_game.update()
    state = torch.cat([state for _ in range(configs.model.n_temp_frames)]).unsqueeze(0)
    self_training_steps = int(
        configs.training.self_training_ratio * configs.training.num_steps
    )
    for i in (
        p_bar := tqdm(
            range(step, configs.training.num_steps),
            initial=step,
            total=configs.training.num_steps,
            desc=f"Loss: {float('nan'):12.4f}, Epsilon: {float('nan'):8.4f}, Action: {float('nan'):2}, Reward: {float('nan'):6}, Point: {float('nan'):6}",
        )
    ):
        # ============== Sampling ==============
        eps = (
            (
                configs.training.final_eps
                + (configs.training.init_eps - configs.training.final_eps)
                * (configs.training.num_steps - self_training_steps - i)
                / (configs.training.num_steps - self_training_steps)
            )
            if i < configs.training.num_steps - self_training_steps
            else 0
        )

        if random.random() < eps:
            action = int(np.random.uniform() > configs.training.threshold)
        else:
            with torch.inference_mode():
                action = torch.argmax(q_model(state.to(device))[0]).item()

        point, next_state, reward, done = main_game.update(action)
        next_state = torch.cat((state[:, 1:, :, :], next_state.unsqueeze(0)), dim=1)

        exp_replay.append([state, action, reward, next_state, done])
        if len(exp_replay) > configs.training.replay_memory_size:
            exp_replay.pop(0)

        state = next_state
        pygame.display.update()

        # ============== Training ==============
        batch = random.sample(
            exp_replay, min(len(exp_replay), configs.training.batch_size)
        )
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *batch
        )

        state_batch = torch.cat(state_batch).to(device)
        action_batch = torch.tensor(
            [[0, 1] if a else [1, 0] for a in action_batch],
            dtype=torch.float32,
            device=device,
        )
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
        next_state_batch = torch.cat(next_state_batch).to(device)
        done_batch = torch.tensor(done_batch, dtype=torch.int64, device=device)

        cur_prediction_batch = q_model(state_batch)
        with torch.inference_mode():
            next_prediction_batch = t_model(next_state_batch)

        y = (
            reward_batch
            + (1 - done_batch)
            * configs.training.gamma
            * torch.max(next_prediction_batch, dim=-1).values
        )

        q_val = torch.sum(cur_prediction_batch * action_batch, dim=-1)

        loss = loss_fn(q_val, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        p_bar.set_description(
            f"Loss: {loss:12.4f}, Epsilon: {eps:8.4f}, Action: {action:2}, Reward: {reward:6}, Point: {point:6}"
        )

        if (i + 1) % configs.training.copy_interval == 0:
            t_model.load_state_dict(q_model.state_dict())

        if (i + 1) % configs.training.log_interval == 0:
            writer.add_scalar("training_loss", loss.item(), i + 1)
            writer.add_scalar("reward", reward, i + 1)
            writer.add_scalar("epsilon", eps, i + 1)
            writer.add_scalar("point", point, i + 1)

        if (i + 1) % configs.training.checkpoint_interval == 0:
            checkpoint = {
                "step": i,
                "q_model_state_dict": q_model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
            }
            torch.save(
                checkpoint,
                f"{configs.training.save_dir}/{configs.training.checkpoint_name}/{configs.training.checkpoint_name}-{i+1:010d}.pth",
            )

        if (i + 1) % configs.training.replay_interval == 0:
            with open(
                f"{configs.training.save_dir}/{configs.training.checkpoint_name}/replay_memory.pth",
                "wb",
            ) as f:
                pickle.dump(exp_replay, f, protocol=pickle.HIGHEST_PROTOCOL)

    p_bar.close()
