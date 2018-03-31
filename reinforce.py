import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from visdom import Visdom


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 16)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 16)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(16, output_dim)

    def forward(self, inputs):
        x = Variable(torch.FloatTensor(inputs))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, env, lr):
        self.env = env
        self.model = Model(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def eval(self, num_episodes):
        # Tests the model on n episodes
        cum_rewards = np.zeros(num_episodes)
        for i in range(num_episodes):
            _, _, rewards = self.generate_episode()
            cum_rewards[i] = np.sum(rewards)
        return cum_rewards.mean(), cum_rewards.std()

    def train(self, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        states, actions, rewards = self.generate_episode()
        g = np.flip(np.cumsum([gamma ** t * rewards[t] for t in reversed(range(len(rewards)))]), axis=0).copy()
        g = Variable(torch.FloatTensor(g / 100), requires_grad=False)
        log_pi = F.log_softmax(self.model(states), dim=0)
        loss = (g * -log_pi[range(len(actions)), actions]).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def generate_episode(self):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        done = False
        while not done:
            pi = F.softmax(self.model(state), dim=0)
            action = pi.multinomial(1).data[0]
            actions.append(action)
            states.append(state)
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
        return states, actions, rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_episodes', dest='train_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--test_episodes', dest='test_episodes', type=int,
                        default=100, help="Number of episodes to test on.")
    parser.add_argument('--episodes_per_eval', dest='episodes_per_eval', type=int,
                        default=500, help="Number of episodes per evaluation.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=1.0, help="The discount factor.")
    args = parser.parse_args()

    env = gym.make('LunarLander-v2')
    reinforce = Reinforce(env, args.lr)
    viz = Visdom()
    loss_plot = None
    reward_plot = None
    losses = []
    rewards_mean = []
    rewards_std = []
    for i in range(args.train_episodes):
        loss = reinforce.train(args.gamma)
        losses.append(loss)
        if loss_plot is None:
            loss_plot = viz.line(X=np.array([i]), Y=np.array([loss]), name='Training loss')
        else:
            viz.line(X=np.array([i]), Y=np.array([loss]), win=loss_plot, update='append')
        if i % args.episodes_per_eval == 0:
            mu, sigma = reinforce.eval(args.test_episodes)
            print('episode', i, 'loss', loss, 'reward average', mu, 'reward std', sigma)
            rewards_mean.append(mu)
            rewards_std.append(sigma)
            plt.errorbar(range(0, i+1, args.episodes_per_eval),
                               rewards_mean, rewards_std, capsize=5)
            plt.xlabel('number of training episodes')
            plt.ylabel('reward')
            if reward_plot is None:
                reward_plot = viz.matplot(plt)
            else:
                viz.close(reward_plot)
                reward_plot = viz.matplot(plt)
