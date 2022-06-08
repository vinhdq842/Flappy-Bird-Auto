import pickle
from os.path import isfile
from random import random, randint, sample

import cv2
import numpy as np
import pygame
import torch
import torch.nn as nn

from game.Constants import w, h
from game.MainGame import MainGame
from model.DeepQNetwork import DeepQNetwork

pygame.init()

screen = pygame.display.set_mode((w, h))

fps_clock = pygame.time.Clock()
pygame.display.set_caption("Flappy Bird")

lr = 1e-4
gamma = .99
batch_size = 64
replay_memory_size = 50000
init_eps = .1
final_eps = 1e-4
num_steps = 2000000
save_path = './backup/'


def preprocess(img):
    img = cv2.cvtColor(cv2.resize(img, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    return img[np.newaxis, :]


if __name__ == '__main__':
    torch.manual_seed(33)

    main_game = MainGame(screen)
    state, _, _ = main_game.update()
    state = torch.cat([state for _ in range(4)])[np.newaxis, :, :, :]

    replay_ex = []
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    it = 0

    if isfile(save_path + "checkpoint.pth"):
        f = open(save_path + "checkpoint.pth", "rb")
        cp = torch.load(f, map_location="cpu")
        it = cp["iter"] + 2
        print("Load from iter %d" % it)
        model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["opt_state_dict"])

    if isfile(save_path + "replay_memory.pth"):
        f = open(save_path + "replay_memory.pth", "rb")
        replay_ex = pickle.load(f)

    L = nn.MSELoss()

    for i in range(it, num_steps):
        prediction = model(state)[0]

        eps = final_eps + (num_steps - i) * (init_eps - final_eps) / num_steps

        r = random()

        if r < eps:
            action = randint(0, 1)
        else:
            action = torch.argmax(prediction).item()

        next_state, reward, done = main_game.update(action)

        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]

        replay_ex.append([state, action, reward, next_state, done])

        if len(replay_ex) > replay_memory_size:
            del replay_ex[0]

        batch = sample(replay_ex, min(len(replay_ex), batch_size))

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.cat(tuple(s for s in state_batch))

        action_batch = torch.from_numpy(
            np.array([[0, 1] if a == 1 else [1, 0] for a in action_batch], dtype=np.float32))

        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])

        next_state_batch = torch.cat(tuple(s for s in next_state_batch))

        cur_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y = torch.cat(tuple(r if d else r + gamma * torch.argmax(p) for r, d, p in
                            zip(reward_batch, done_batch, next_prediction_batch)))

        q_val = torch.sum(cur_prediction_batch * action_batch, dim=1)

        optimizer.zero_grad()
        loss = L(q_val, y)
        loss.backward()
        optimizer.step()

        state = next_state
        print("Iteration: {}/{}, Loss: {:.5f}, Epsilon {:.5f}, Reward: {}".format(i + 1, num_steps, loss, eps, reward))

        if (i + 1) % 5000 == 0:
            checkpoint = {"iter": i, "model_state_dict": model.state_dict(), "opt_state_dict": optimizer.state_dict()}
            torch.save(checkpoint, save_path + "checkpoint.pth")
            with open(save_path + "replay_memory.pth", "wb") as f:
                pickle.dump(replay_ex, f, protocol=pickle.HIGHEST_PROTOCOL)
        pygame.display.update()
        fps_clock.tick(30)
