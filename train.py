import os
import pickle
import random
from os.path import isfile

import numpy as np
import pygame
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from game.Constants import h, w
from game.MainGame import MainGame
from model.DeepQNetwork import DeepQNetwork

pygame.init()
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("Flappy Bird Auto - Training")

lr = 3e-5
gamma = 0.99
batch_size = 64
replay_memory_size = 20000
n_temp_frames = 4
init_eps = 0.987
final_eps = 1e-4
threshold = 0.9
num_steps = 2000050
checkpoint_interval = 5000
log_interval = 200
copy_interval = 10000
save_path = "./backup/"

os.makedirs(save_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("logs")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    main_game = MainGame(screen)
    q_model = DeepQNetwork(n_temp_frames=n_temp_frames).to(device)
    t_model = DeepQNetwork(n_temp_frames=n_temp_frames).to(device)

    print(f"The model has {sum(i.numel() for i in q_model.parameters()):,} params.")

    it = 0
    optimizer = torch.optim.Adam(q_model.parameters(), lr=lr)
    if isfile(save_path + "checkpoint.pth"):
        with open(save_path + "checkpoint.pth", "rb") as f:
            cp = torch.load(f, map_location=device)
        it = cp["iter"] + 2
        print("Load from iter %d" % (it - 1))

        q_model.load_state_dict(cp["q_model_state_dict"])
        t_model.load_state_dict(cp["q_model_state_dict"])
        optimizer.load_state_dict(cp["opt_state_dict"])

    if isfile(save_path + "replay_memory.pth"):
        with open(save_path + "replay_memory.pth", "rb") as f:
            replay_ex = pickle.load(f)

    q_model.train()
    t_model.eval()
    loss_fn = nn.MSELoss()

    replay_ex = []
    state, *_ = main_game.update()
    state = torch.cat([state for _ in range(n_temp_frames)]).unsqueeze(0)

    for i in range(it, num_steps):
        prediction = q_model(state.to(device))[0]

        eps = final_eps + (num_steps - i) * (init_eps - final_eps) / num_steps

        r = random.random()
        if r < eps:
            action = int(np.random.uniform() > threshold)
        else:
            action = torch.argmax(prediction).item()

        next_state, reward, done = main_game.update(action)
        next_state = torch.cat((state[:, 1:, :, :], next_state.unsqueeze(0)), dim=1)

        replay_ex.append([state, action, reward, next_state, done])
        if len(replay_ex) > replay_memory_size:
            replay_ex.pop(0)

        batch = random.sample(replay_ex, min(len(replay_ex), batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *batch
        )

        state_batch = torch.cat(state_batch).to(device)
        action_batch = torch.tensor(
            [[0, 1] if a == 1 else [1, 0] for a in action_batch],
            dtype=torch.float32,
            device=device,
        )
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
        next_state_batch = torch.cat(next_state_batch).to(device)

        cur_prediction_batch = q_model(state_batch)
        with torch.inference_mode():
            next_prediction_batch = t_model(next_state_batch)

        y = (
            reward_batch
            + (1 - torch.tensor(done_batch, dtype=torch.int64, device=device))
            * gamma
            * torch.max(next_prediction_batch, dim=-1).values
        )

        q_val = torch.sum(cur_prediction_batch * action_batch, dim=-1)

        loss = loss_fn(q_val, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        state = next_state
        print(
            "Iteration: {}/{}, Loss: {:.5f}, Epsilon: {:.5f}, Action: {}, Reward: {}".format(
                i + 1, num_steps, loss, eps, action, reward
            )
        )
        
        if (i + 1) % copy_interval == 0:
            t_model.load_state_dict(q_model.state_dict())

        if (i + 1) % log_interval == 0:
            writer.add_scalar("training_loss", loss.item(), i + 1)

        if (i + 1) % checkpoint_interval == 0:
            checkpoint = {
                "iter": i,
                "q_model_state_dict": q_model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path + "checkpoint.pth")
            with open(save_path + "replay_memory.pth", "wb") as f:
                pickle.dump(replay_ex, f, protocol=pickle.HIGHEST_PROTOCOL)

        pygame.display.update()
