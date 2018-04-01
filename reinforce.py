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
    def __init__(self, state_dim, num_actions):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        log_pi = F.log_softmax(self.fc3(x), dim=-1)
        return log_pi


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, env, lr):
        # Initializes REINFORCE.
        # Args:
        # - env: Gym environment.
        # - lr: Learning rate for the model.
        self.env = env
        self.model = Model(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def train(self, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        states, actions, rewards = self.generate_episode()
        log_pi = self.model(Variable(torch.Tensor(states)))
        R = np.flip(np.cumsum([gamma ** t * rewards[t] / 100 for t in reversed(range(len(rewards)))]), axis=0).copy()
        R = Variable(torch.Tensor(R), requires_grad=False)
        loss = (-log_pi[range(len(actions)), actions] * R).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def eval(self, num_episodes):
        # Tests the model on n episodes
        cum_rewards = np.zeros(num_episodes)
        for i in range(num_episodes):
            _, _, rewards = self.generate_episode()
            cum_rewards[i] = np.sum(rewards)
        return cum_rewards.mean(), cum_rewards.std()

    def select_action(self, state):
        # Select the action to take by sampling from the policy model
        log_pi = self.model(Variable(torch.Tensor(state)))
        pi = torch.distributions.Categorical(log_pi.exp())
        action = pi.sample().data[0]
        return action

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
            action = self.select_action(state)
            actions.append(action)
            states.append(state)
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
        return states, actions, rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', dest='task_name',
                        default='REINFORCE', help="Name of the experiment")
    parser.add_argument('--train_episodes', dest='train_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--test_episodes', dest='test_episodes', type=int,
                        default=100, help="Number of episodes to test on.")
    parser.add_argument('--episodes_per_eval', dest='episodes_per_eval', type=int,
                        default=500, help="Number of episodes per evaluation.")
    parser.add_argument('--episodes_per_plot', dest='episodes_per_plot', type=int,
                        default=50, help="Number of episodes between each plot update.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=1.0, help="The discount factor.")
    parser.add_argument('--seed', dest='seed', type=int,
                        default=110, help="The random seed.")
    args = parser.parse_args()

    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    viz = Visdom()
    loss_plot = None
    reward_plot = None

    reinforce = Reinforce(env, args.lr)
    losses = []
    rewards_mean = []
    rewards_std = []
    for i in range(args.train_episodes):
        loss = reinforce.train(args.gamma)
        losses.append(loss)
        if i % args.episodes_per_plot == 0:
            if loss_plot is None:
                opts = dict(xlabel='episodes', ylabel='loss')
                loss_plot = viz.line(X=np.array(range(i + 1)), Y=np.array(losses), env=args.task_name, opts=opts)
            else:
                viz.line(X=np.array(range(i - args.episodes_per_plot + 1, i + 1)),
                         Y=np.array(losses[i - args.episodes_per_plot + 1:i + 1]),
                         env=args.task_name, win=loss_plot, update='append')
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
                reward_plot = viz.matplot(plt, env=args.task_name)
            else:
                viz.matplot(plt, env=args.task_name, win=reward_plot)
