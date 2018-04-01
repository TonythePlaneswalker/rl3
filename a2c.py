import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from visdom import Visdom
from reinforce import Reinforce


class Model(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.policy = torch.nn.Linear(16, num_actions)
        self.value = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        log_pi = F.log_softmax(self.policy(x), dim=-1)
        v = self.value(x)
        return log_pi, v


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, env, lr, n):
        # Initializes A2C.
        # Args:
        # - env: Gym environment.
        # - lr: Learning rate for the model.
        # - n: The value of N in N-step A2C.
        self.env = env
        self.model = Model(env.observation_space.shape[0], env.action_space.n)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.n = n

    def train(self, gamma=1.0):
        # Trains the model on a single episode using A2C.
        states, actions, rewards = self.generate_episode()
        log_pi, value = self.model(self._array2var(states))
        T = len(rewards)
        R = np.zeros(T)
        for t in reversed(range(T)):
            v_end = value.data[t + self.n] if t + self.n < T else 0
            R[t] = gamma ** self.n * v_end + \
                   sum([gamma ** k * rewards[t+k] / 100 for k in range(min(self.n, T - t))])
        policy_loss = (-log_pi[range(len(actions)), actions] * self._array2var(R)).mean()
        value_loss = ((R - value) ** 2).mean()
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return policy_loss.data[0], value_loss.data[0]

    def select_action(self, state):
        # Select the action to take by sampling from the policy model
        log_pi, _ = self.model(self._array2var(state))
        pi = torch.distributions.Categorical(log_pi.exp())
        action = pi.sample().data[0]
        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', dest='task_name',
                        default='A2C', help="Name of the experiment")
    parser.add_argument('--train_episodes', dest='train_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--test_episodes', dest='test_episodes', type=int,
                        default=100, help="Number of episodes to test on.")
    parser.add_argument('--episodes_per_eval', dest='episodes_per_eval', type=int,
                        default=500, help="Number of episodes between each evaluation.")
    parser.add_argument('--episodes_per_plot', dest='episodes_per_plot', type=int,
                        default=50, help="Number of episodes between each plot update.")
    parser.add_argument('-n', dest='n', type=int,
                        default=20, help="Number steps in a trace.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.001, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.99, help="The discount factor.")
    parser.add_argument('--seed', dest='seed', type=int,
                        default=123, help="The random seed.")
    args = parser.parse_args()

    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    viz = Visdom()
    policy_loss_plot = None
    value_loss_plot = None
    reward_plot = None

    a2c = A2C(env, args.lr, args.n)
    policy_losses = []
    value_losses = []
    rewards_mean = []
    rewards_std = []
    for i in range(args.train_episodes):
        policy_loss, value_loss = a2c.train(args.gamma)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        if i % args.episodes_per_plot == 0:
            if policy_loss_plot is None:
                opts = dict(xlabel='episodes', ylabel='policy loss')
                policy_loss_plot = viz.line(X=np.array(range(i + 1)), Y=np.array(policy_losses),
                                            env=args.task_name, opts=opts)
            else:
                viz.line(X=np.array(range(i - args.episodes_per_plot + 1, i + 1)),
                         Y=np.array(policy_losses[i - args.episodes_per_plot + 1:i + 1]),
                         env=args.task_name, win=policy_loss_plot, update='append')
            if value_loss_plot is None:
                opts = dict(xlabel='episodes', ylabel='value loss')
                value_loss_plot = viz.line(X=np.array(range(i + 1)), Y=np.array(value_losses),
                                           env=args.task_name, opts=opts)
            else:
                viz.line(X=np.array(range(i - args.episodes_per_plot + 1, i + 1)),
                         Y=np.array(value_losses[i - args.episodes_per_plot + 1:i + 1]),
                         env=args.task_name, win=value_loss_plot, update='append')
        if i % args.episodes_per_eval == 0:
            mu, sigma = a2c.eval(args.test_episodes)
            print('episode', i, 'policy loss', policy_loss, 'value loss', value_loss,
                  'reward average', mu, 'reward std', sigma)
            rewards_mean.append(mu)
            rewards_std.append(sigma)
            plt.errorbar(range(0, i + 1, args.episodes_per_eval),
                         rewards_mean, rewards_std, capsize=5)
            plt.xlabel('episodes')
            plt.ylabel('average reward')
            if reward_plot is None:
                reward_plot = viz.matplot(plt, env=args.task_name)
            else:
                viz.matplot(plt, env=args.task_name, win=reward_plot)
