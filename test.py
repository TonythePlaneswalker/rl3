import gym
import sys
import torch
from reinforce import Reinforce
from a2c import A2C


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env = gym.wrappers.Monitor(env, sys.argv[1], force=True)
    agent = A2C(env, 0, 50)
    agent.model.load_state_dict(torch.load(sys.argv[2], map_location=lambda storage, loc: storage))
    state = agent.env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = agent.env.step(action)
