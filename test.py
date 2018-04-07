import argparse
import gym
import torch
from reinforce import Reinforce
from a2c import A2C


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_type')
    parser.add_argument('model_path')
    parser.add_argument('video_dir')
    args = parser.parse_args()
    env = gym.make('LunarLander-v2')
    env = gym.wrappers.Monitor(env, args.video_dir, force=True)
    if args.agent_type == 'reinforce':
        agent = Reinforce(env, 0)
    elif args.agent_type == 'a2c':
        agent = A2C(env, 0)
    else:
        print('Unknown agent type %s' % args.agent_type)
        exit(1)
    agent.model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
    print('Reward', sum(agent.generate_episode()[0]))
