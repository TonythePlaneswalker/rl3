import argparse
import gym
import torch
from reinforce import Reinforce
from a2c import A2C


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_type', help="The agent type. Either reinforce or a2c.")
    parser.add_argument('model_path', help="Path of the saved model weights.")
    parser.add_argument('-n', type=int, default=1, help="N value for N-step A2C.")
    parser.add_argument('--seed', type=int, default=666, help="Random seed for the environment.")
    parser.add_argument('--num_episodes', type=int, default=1, help="Number of test episodes.")
    parser.add_argument('--record', action='store_true', help="Record videos of test episodes.")
    parser.add_argument('--video_dir', help="Directory to store recorded videos.")
    args = parser.parse_args()
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    if args.record:
        env = gym.wrappers.Monitor(env, args.video_dir, force=True)
    if args.agent_type == 'reinforce':
        agent = Reinforce(env, 0)
    elif args.agent_type == 'a2c':
        agent = A2C(env, 0, args.n)
    else:
        print('Unknown agent type %s' % args.agent_type)
        exit(1)
    agent.model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
    r_avg, r_std = agent.eval(args.num_episodes, stochastic=True)
    print('Reward average %.6f std %.6f' % (r_avg, r_std))
