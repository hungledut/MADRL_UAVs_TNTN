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


# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1
epsilon_min = 0.3
epsilon_decay = 0.9996
batch_size = 128
target_update_freq = 1000
memory_size = 10000
episodes = 4000
max_step = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def select_action(state, policy_net, epsilon):
    rand_value = random.random()
    if rand_value < epsilon:
        return env.action_space.sample()  # Explore
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state.to(device))
        return torch.argmax(q_values).item()  # Exploit

# Function to optimize the model using experience replay
def optimize_model():
    if len(memory0) < batch_size:
        return
    if len(memory1) < batch_size:
        return
    if len(memory2) < batch_size:
        return

    batch0 = random.sample(memory0, batch_size)
    UAV0_batch, action0_batch, reward0_batch, next_UAV0_batch = zip(*batch0)

    batch1 = random.sample(memory1, batch_size)
    UAV1_batch, action1_batch, reward1_batch, next_UAV1_batch = zip(*batch1)

    batch2 = random.sample(memory2, batch_size)
    UAV2_batch, action2_batch, reward2_batch, next_UAV2_batch = zip(*batch2)

    UAV0_batch = torch.FloatTensor(UAV0_batch).to(device)
    action0_batch = torch.LongTensor(action0_batch).unsqueeze(1).to(device)
    reward0_batch = torch.FloatTensor(reward0_batch).to(device)
    next_UAV0_batch = torch.FloatTensor(next_UAV0_batch).to(device)

    UAV1_batch = torch.FloatTensor(UAV1_batch).to(device)
    action1_batch = torch.LongTensor(action1_batch).unsqueeze(1).to(device)
    reward1_batch = torch.FloatTensor(reward1_batch).to(device)
    next_UAV1_batch = torch.FloatTensor(next_UAV1_batch).to(device)

    UAV2_batch = torch.FloatTensor(UAV2_batch).to(device)
    action2_batch = torch.LongTensor(action2_batch).unsqueeze(1).to(device)
    reward2_batch = torch.FloatTensor(reward2_batch).to(device)
    next_UAV2_batch = torch.FloatTensor(next_UAV2_batch).to(device)



    # Compute Q-values for current states
    q_values0 = policy_net0(UAV0_batch.to(device)).gather(1, action0_batch.to(device)).squeeze()
    q_values1 = policy_net1(UAV1_batch.to(device)).gather(1, action1_batch.to(device)).squeeze()
    q_values2 = policy_net2(UAV2_batch.to(device)).gather(1, action2_batch.to(device)).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values0 = target_net0(next_UAV0_batch).max(1)[0]
        target_q_values0 = reward0_batch + gamma * max_next_q_values0

        max_next_q_values1 = target_net1(next_UAV1_batch).max(1)[0]
        target_q_values1 = reward1_batch + gamma * max_next_q_values1

        max_next_q_values2 = target_net2(next_UAV2_batch).max(1)[0]
        target_q_values2 = reward2_batch + gamma * max_next_q_values2

    loss0 = nn.MSELoss()(q_values0, target_q_values0.to(device))
    loss1 = nn.MSELoss()(q_values1, target_q_values1.to(device))
    loss2 = nn.MSELoss()(q_values2, target_q_values2.to(device))


    optimizer0.zero_grad()
    loss0.backward()
    optimizer0.step()

    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

if __name__ == '__main__':
    print('device: ', device)
    env = UAV_Environment(max_step = max_step)
    policy_net0 = DQN(308, 5).to(device)
    target_net0 = DQN(308, 5).to(device)

    policy_net1 = DQN(308, 5).to(device)
    target_net1 = DQN(308, 5).to(device)

    policy_net2 = DQN(308, 5).to(device)
    target_net2 = DQN(308, 5).to(device)

    optimizer0 = optim.Adam(policy_net0.parameters(), lr=learning_rate)
    optimizer1 = optim.Adam(policy_net1.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(policy_net2.parameters(), lr=learning_rate)

    memory0 = deque(maxlen=memory_size)
    memory1 = deque(maxlen=memory_size)
    memory2 = deque(maxlen=memory_size)

    # Main training loop
rewards_per_episode = []
percentage_user_episode = []
steps_done = 0
done = False


for episode in range(1,episodes+1):
    O_UAV0, O_UAV1, O_UAV2 = env.reset()
    episode_reward0 = 0
    episode_reward1 = 0
    episode_reward2 = 0
    S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t = 0,0,0,0
    g_t = 0
    l_t_0 = 0
    l_t_1 = 0
    l_t_2 = 0
    w = 0.5
    percentage_users = []



    # the maximum step of one episode
    count = 0

    # One episode (One Trajectory)
    while not done:

        if count > max_step:
          break

        # Select action
        action0 = select_action(O_UAV0, policy_net0, epsilon)
        action1 = select_action(O_UAV1, policy_net1, epsilon)
        action2 = select_action(O_UAV2, policy_net2, epsilon)
        next_O_UAV0, next_O_UAV1, next_O_UAV2, S, N_UAV0, N_UAV1, N_UAV2 = env.step([action0, action1, action2])

        if S > S_t:
            g_t = 1
        elif S < S_t:
            g_t = -1
        else:
            g_t = 0

        if N_UAV0 > N_UAV0_t:
            l_t_0 = 1
        elif N_UAV0 < N_UAV0_t:
            l_t_0 = -1
        else:
            l_t_0 = 0

        if N_UAV1 > N_UAV1_t:
            l_t_1 = 1
        elif N_UAV1 < N_UAV1_t:
            l_t_1 = -1
        else:
            l_t_1 = 0

        if N_UAV2 > N_UAV2_t:
            l_t_2 = 1
        elif N_UAV2 < N_UAV2_t:
            l_t_2 = -1
        else:
            l_t_2 = 0

        S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t = S, N_UAV0, N_UAV1, N_UAV2

        # print(S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t)

        reward0 = w*l_t_0 + (1-w)*g_t
        reward1 = w*l_t_1 + (1-w)*g_t
        reward2 = w*l_t_2 + (1-w)*g_t

        episode_reward0 += reward0
        episode_reward1 += reward1
        episode_reward2 += reward2

        # Store transition in memory
        memory0.append((O_UAV0, action0, reward0, next_O_UAV0))
        memory1.append((O_UAV1, action1, reward1, next_O_UAV1))
        memory2.append((O_UAV2, action2, reward2, next_O_UAV2))


        # Update state
        O_UAV0, O_UAV1, O_UAV2 = next_O_UAV0, next_O_UAV1, next_O_UAV2
        # episode_reward += reward

        # Optimize model
        optimize_model()

        # Update target network periodically
        if steps_done % target_update_freq == 0:
            target_net0.load_state_dict(policy_net0.state_dict())
            target_net1.load_state_dict(policy_net1.state_dict())
            target_net2.load_state_dict(policy_net2.state_dict())

        steps_done += 1
        count += 1

        percentage_users.append(S_t*100/250)

    sum_reward_3UAV = episode_reward0 + episode_reward1 + episode_reward2
    rewards_per_episode.append(sum_reward_3UAV)
    percentage_user_episode.append(S_t*100/250)
    # Decay epsilon
    print('Episode ',episode,': The percentage of satisfied users = ', S_t*100/250 , '%', '. The total reward of 3 UAVs = ', sum_reward_3UAV)
    if episode % 100 == 0:
        plt.figure()
        env.plot()
        plt.plot(percentage_users)
        plt.ylim(0, 100)
        plt.xlabel('Movement step')
        plt.ylabel('The percentage of satisfied users (%)')
        plt.title('The percentage of satisfied users on each step')
        plt.savefig("result_step.png")
        # plt.show()
        plt.close()

        ################ save 3 models ######################
        torch.save(policy_net0.state_dict(), 'policy0_weights.pth')
        torch.save(policy_net1.state_dict(), 'policy1_weights.pth')
        torch.save(policy_net2.state_dict(), 'policy2_weights.pth')

    epsilon = max(epsilon_min, epsilon_decay * epsilon)

avg_rewards = []
for i in range(0,len(rewards_per_episode)-5,5):
    a = np.mean(rewards_per_episode[i:i+5])
    for j in range(5):
        avg_rewards.append(a)

plt.figure()
plt.plot(rewards_per_episode, label = 'reward on each 1 episode')
plt.plot(avg_rewards, label = 'average reward on each 5 episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('The total reward each episode')
plt.legend()
plt.savefig("result_reward.png")
# plt.show()
plt.close()

avg_user = []
for i in range(0,len(percentage_user_episode)-5,5):
    a = np.mean(percentage_user_episode[i:i+5])
    for j in range(5):
        avg_user.append(a)

plt.figure()
plt.plot(percentage_user_episode, label = 'the percentage of users on each 1 episode')
plt.plot(avg_user, label = 'the percentage of on each 5 episode')
plt.xlabel('Episode')
plt.ylabel('The percentage of users')
plt.title('The percentage of users each episode')
plt.legend()
plt.savefig("result_user.png")
# plt.show()
plt.close()







