import gymnasium as gym
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import our customed uav-environment and policy
from uav_environment import UAV_Environment
from policy import DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_step = 50

if __name__ == '__main__':
    policy0 = DQN(106, 5).to(device)
    policy1 = DQN(106, 5).to(device)
    policy2 = DQN(106, 5).to(device)

    policy0.load_state_dict(torch.load('weights_dqn/policy0_weights.pth'))
    policy1.load_state_dict(torch.load('weights_dqn/policy1_weights.pth'))
    policy2.load_state_dict(torch.load('weights_dqn/policy2_weights.pth'))

    env = UAV_Environment(max_step = max_step)

    ###### test model ##########
    O_UAV0, O_UAV1, O_UAV2 = env.reset()
    for _ in range(max_step):
        q_values_0 = policy0(torch.FloatTensor(O_UAV0).unsqueeze(0).to(device))
        action0 = torch.argmax(q_values_0).item()
        q_values_1 = policy1(torch.FloatTensor(O_UAV1).unsqueeze(0).to(device))
        action1 = torch.argmax(q_values_1).item()
        q_values_2 = policy2(torch.FloatTensor(O_UAV2).unsqueeze(0).to(device))
        action2 = torch.argmax(q_values_2).item()
        O_UAV0, O_UAV1, O_UAV2, S, N_UAV0, N_UAV1, N_UAV2 = env.step([action0,action1,action2])
    env.plot()