from random import random, randint
import numpy as np
import torch

from model.DeepQNetwork import DeepQNetwork

net = DeepQNetwork()

x = torch.randn(3, 4, 84, 84)
y = net(x)
print(y)

print(torch.argmax(y).shape)
torch.manual_seed(223)
