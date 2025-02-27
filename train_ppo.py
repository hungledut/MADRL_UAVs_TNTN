import os
import glob
import time

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np
import gymnasium as gym

# import our customed uav-environment and policy
from uav_environment import UAV_Environment

import matplotlib
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # ACTOR
        if has_continuous_action_space : # continuous action space
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else: # discrete action space
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )


        # CRITIC
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError


    def act(self, state):

        # ACTOR
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # CRITIC
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()


    def evaluate(self, state, action):

        # ACTOR
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # CRITIC
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)


        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()


        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


has_continuous_action_space = False

max_ep_len = 300                 # max timesteps in one episode
max_training_timesteps = int(3e5)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
action_std = None

################ PPO hyperparameters ################

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

time_step = 0
i_episode = 0

# initialize a PPO agent
ppo_agent0 = PPO(206, 5, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
ppo_agent1 = PPO(206, 5, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
ppo_agent2 = PPO(206, 5, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

env = UAV_Environment(max_step = max_ep_len)

if __name__ == '__main__':
    # training loop
    rewards_per_episode = []
    percentage_user_episode = []

    while time_step <= max_training_timesteps:
        O_UAV0, O_UAV1, O_UAV2 = env.reset()
        S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t = 0,0,0,0
        g_t = 0
        l_t_0 = 0
        l_t_1 = 0
        l_t_2 = 0
        w = 0.6
        percentage_users = []

        O_UAV0, O_UAV1, O_UAV2 = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action0 = ppo_agent0.select_action(O_UAV0)
            action1 = ppo_agent1.select_action(O_UAV1)
            action2 = ppo_agent2.select_action(O_UAV2)

            O_UAV0, O_UAV1, O_UAV2, S, N_UAV0, N_UAV1, N_UAV2 = env.step([action0, action1, action2])
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

            # saving reward and is_terminals
            ppo_agent0.buffer.rewards.append(reward0)
            ppo_agent1.buffer.rewards.append(reward1)
            ppo_agent2.buffer.rewards.append(reward2)

            ppo_agent0.buffer.is_terminals.append(False)
            ppo_agent1.buffer.is_terminals.append(False)
            ppo_agent2.buffer.is_terminals.append(False)

            time_step +=1
            current_ep_reward += reward0
            current_ep_reward += reward1
            current_ep_reward += reward2

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent0.update()
                ppo_agent1.update()
                ppo_agent2.update()
            percentage_users.append(S_t*100/250)

        rewards_per_episode.append(current_ep_reward)
        percentage_user_episode.append(S_t*100/250)
        print('Episode ', i_episode, ': Reward = ', current_ep_reward ,'The percentage of satisfied users = ', S_t*100/250 , '%')
        if time_step % (max_ep_len*50) == 0:
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
            ppo_agent0.save('ppo_agent0.pth')
            ppo_agent1.save('ppo_agent1.pth')
            ppo_agent2.save('ppo_agent2.pth')

        i_episode += 1

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

