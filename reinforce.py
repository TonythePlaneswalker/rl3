import argparse
import gym
import matplotlib
matplotlib.use('Agg')
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
        self.fc3 = torch.nn.Linear(16, 16)
        self.fc4 = torch.nn.Linear(16, num_actions)
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        for layer in layers:
            torch.nn.init.kaiming_normal(layer.weight)
            torch.nn.init.constant(layer.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=-1)
        return x


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, env, lr):
        # Initializes REINFORCE.
        # Args:
        # - env: Gym environment.
        # - lr: Learning rate for the model.
        self.env = env
        self.model = Model(env.observation_space.shape[0], env.action_space.n)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def _array2var(self, array, requires_grad=True):
        var = Variable(torch.Tensor(array), requires_grad)
        if torch.cuda.is_available():
            var = var.cuda()
        return var

    def train(self, gamma):
        # Trains the model on a single episode using REINFORCE.
        rewards, log_pi = self.generate_episode()
        T = len(rewards)
        G = np.zeros(T, dtype=np.float32)
        for t in reversed(range(T)):
            G[t] = gamma * G[(t + 1) % T] + rewards[t]
        G = self._array2var(G, requires_grad=False)
        loss = (-log_pi * G).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data[0], T

    def eval(self, num_episodes, stochastic=True):
        # Tests the model on n episodes
        cum_rewards = np.zeros(num_episodes)
        for i in range(num_episodes):
            rewards = self.generate_episode(stochastic)[0]
            cum_rewards[i] = np.sum(rewards)
        return cum_rewards.mean(), cum_rewards.std()

    def select_action(self, state, stochastic):
        # Select the action to take by sampling from the policy model
        # Returns
        # - the action
        # - log probability of the chosen action (as a Variable)
        log_pi = self.model(self._array2var(state))
        if stochastic:
            action = torch.distributions.Categorical(log_pi.exp()).sample()
        else:
            _, action = log_pi.max(0)
        return action.data[0], log_pi[action]

    def generate_episode(self, stochastic=True):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of rewards, indexed by time step
        # - a Variable of log probabilities
        log_probs = []
        rewards = []
        state = self.env.reset()
        done = False
        while not done:
            action, log_prob = self.select_action(state, stochastic)
            log_probs.append(log_prob)
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
        return rewards, torch.cat(log_probs)


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
                        default=0.0005, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.999, help="The discount factor.")
    parser.add_argument('--seed', dest='seed', type=int,
                        default=666, help="The random seed.")
    args = parser.parse_args()

    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    reinforce = Reinforce(env, args.lr)
    losses = np.zeros(args.train_episodes)
    lengths = np.zeros(args.train_episodes)
    rewards_mean = np.zeros(args.train_episodes // args.episodes_per_eval + 1)
    rewards_std = np.zeros(args.train_episodes // args.episodes_per_eval + 1)
    rewards_mean[0], rewards_std[0] = reinforce.eval(args.test_episodes)
    print('episode', 0, 'reward average', rewards_mean[0], 'reward std', rewards_std[0])
    plt.xlabel('episodes')
    plt.ylabel('average reward')
    errbar = plt.errorbar(np.arange(1), rewards_mean[:1], rewards_std[:1], capsize=3)

    viz = Visdom()
    loss_plot = None
    length_plot = None
    reward_plot = viz.matplot(plt, env=args.task_name)

    for i in range(args.train_episodes):
        losses[i], lengths[i] = reinforce.train(args.gamma)
        if (i + 1) % args.episodes_per_plot == 0:
            if loss_plot is None:
                opts = dict(xlabel='episodes', ylabel='loss')
                loss_plot = viz.line(X=np.arange(1, i + 2), Y=losses[:i + 1],
                                     env=args.task_name, opts=opts)
            else:
                viz.line(X=np.arange(i - args.episodes_per_plot + 1, i + 2),
                         Y=losses[i - args.episodes_per_plot:i + 1],
                         env=args.task_name, win=loss_plot, update='append')
            if length_plot is None:
                opts = dict(xlabel='episodes', ylabel='episode length')
                length_plot = viz.line(X=np.arange(1, i + 2), Y=lengths[:i + 1],
                                       env=args.task_name, opts=opts)
            else:
                viz.line(X=np.arange(i - args.episodes_per_plot + 1, i + 2),
                         Y=lengths[i - args.episodes_per_plot:i + 1],
                         env=args.task_name, win=length_plot, update='append')
        if (i + 1) % args.episodes_per_eval == 0:
            j = (i + 1) // args.episodes_per_eval
            rewards_mean[j], rewards_std[j] = reinforce.eval(args.test_episodes)
            print('episode', i + 1, 'loss', losses[i],
                  'reward average', rewards_mean[j], 'reward std', rewards_std[j])
            errbar.remove()
            errbar = plt.errorbar(np.arange(j + 1) * args.episodes_per_eval,
                                  rewards_mean[:j + 1], rewards_std[:j + 1], capsize=3)
            viz.matplot(plt, env=args.task_name, win=reward_plot)
    plt.savefig('figs/' + args.task_name + '_rewards.png')
    torch.save(reinforce.model.state_dict(), 'models/' + args.task_name + '.model')
