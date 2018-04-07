import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import gym

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, args, env):
        self.lr = args.lr
        self.num_episodes = args.num_episodes
        self.eval_episodes = args.eval_episodes
        self.r_scale = args.r_scale
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.log_dir = args.log_dir

        self.state = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.action = tf.placeholder(tf.float32, [None, self.num_actions])

        with tf.variable_scope("layer1"):
            hidden = tf.layers.dense(self.state, 16, activation=tf.nn.relu)
        with tf.variable_scope("layer2"):
            hidden = tf.layers.dense(hidden, 16, activation=tf.nn.relu)
        with tf.variable_scope("layer3"):
            hidden = tf.layers.dense(hidden, 16, activation=tf.nn.relu)
        with tf.variable_scope("output"):
            self.policy = tf.layers.dense(hidden, self.num_actions, activation=tf.nn.softmax)

        self.G = tf.placeholder(tf.float32, shape=[None])
        self.pi = tf.reduce_sum(self.action * self.policy, axis=1)

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(-self.G * tf.log(self.pi))
            tf.summary.scalar('loss', self.loss)
        with tf.variable_scope("training"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(loss=self.loss)

    def train(self, env, gamma=0.99):
        # Trains the model on a single episode using REINFORCE.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Summary for tensorboard
        loss_summary = tf.summary.scalar('loss', self.loss)
        episode_length = tf.placeholder(tf.int32, shape=(), name='episode_length')
        length_summary = tf.summary.scalar('episode_length', episode_length)
        avg_reward = tf.placeholder(tf.float32, shape=(), name='avg_reward')
        avg_reward_summary = tf.summary.scalar('average reward', avg_reward)
        std_reward = tf.placeholder(tf.float32, shape=(), name='std_reward')
        std_reward_summary = tf.summary.scalar('std reward', std_reward)

        # Set up saving and logging
        saver = tf.train.Saver(max_to_keep=10)
        save_path = os.path.join(self.log_dir, 'checkpoints', 'model')
        steps_per_save = self.num_episodes // 9

        sess.run(tf.global_variables_initializer())

        if os.path.exists(self.log_dir):
            delete_key = input('{} exists. Delete? [y (or enter)/N]'.format(self.log_dir))
            if delete_key == 'y' or delete_key == "":
                os.system('rm -rf {}/*'.format(self.log_dir))
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'))
        saver.save(sess, save_path, global_step)
        writer = tf.summary.FileWriter('./log/{}'.format(self.log_dir), sess.graph)

        for i in range(self.num_episodes):
            states, actions, rewards = self.generate_episode(env, sess)
            T = len(rewards)
            G = np.zeros(T, dtype=np.float32)

            # compute Gt
            for t in reversed(range(T)):
                for k in range(t, T):
                    G[t] += gamma**(k - t) * rewards[k]

            # train the model
            _, cur_policy, cur_loss, cur_summaries = sess.run([self.train_op, self.policy, self.loss, loss_summary],
                                                              feed_dict={self.state: states,
                                                                         self.action: actions,
                                                                         self.G: G})
            writer.add_summary(cur_summaries, i)
            summary = sess.run(length_summary, feed_dict={episode_length: T})
            writer.add_summary(summary, i)

            if (i + 1) % 100 == 0:
                cum_rewards = np.zeros(self.eval_episodes)
                for j in range(self.eval_episodes):
                    rewards = self.generate_episode(self.env, sess)[2]
                    cum_rewards[j] = np.sum(rewards)
                summary = sess.run(avg_reward_summary, feed_dict={avg_reward: cum_rewards.mean()})
                writer.add_summary(summary, i)
                summary = sess.run(std_reward_summary, feed_dict={std_reward: cum_rewards.std()})
                writer.add_summary(summary, i)
                print("Training Episode: %f Cumulative rewards: mean: %f std: %f" % (i + 1, cum_rewards.mean(),
                                                                                     cum_rewards.std()))
                print("Loss: %f" % cur_loss)

            if i % steps_per_save == 0:
                saver.save(sess, save_path, global_step)

    def select_action(self, state, sess):
        """
        Select the action to take by sampling from the policy model
        :param state: 
        :param sess:
        :return: 
            - the action
            - log probability of the chosen action (as a Variable)
        """
        policy = sess.run(self.policy, feed_dict={self.state: [state]})
        policy = policy[0]
        action = np.random.choice(range(self.num_actions), p=policy)
        pi = policy[action]
        return action, pi

    def generate_episode(self, env, sess, render=False):
        """
        Generates an episode by running the given model on the given env.
        :param env: 
        :param sess:
        :param render: 
        :return: 
            - a list of states, indexed by time step
            - a list of actions, indexed by time step
            - a list of rewards, indexed by time step
        """
        states = []  # (episode_length, 8)
        actions = []  # (episode_length, 8) -> one-hot encoding
        rewards = []  # (episode_length,)

        state = env.reset()
        done = False

        if render:
            env.render()

        while not done:
            action, pi = self.select_action(state, sess)
            one_hot_action = np.zeros(env.action_space.n)
            one_hot_action[action] = 1.
            states.append(state)
            actions.append(one_hot_action)
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            rewards.append(reward)
        return states, actions, rewards

    # def eval(self):
    #     # Tests the model on n episodes
    #     cum_rewards = np.zeros(self.num_episodes)
    #     for i in range(self.num_episodes):
    #         rewards = self.generate_episode(self.env, sess)[2]
    #         cum_rewards[i] = np.sum(rewards)
    #     print("Cumulative rewards: mean: %f std: %f" % (cum_rewards.mean(), cum_rewards.std()))
    #     return cum_rewards.mean(), cum_rewards.std()


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='log directory where the checkpoints and summaries are saved.')
    parser.add_argument('--num-episodes', dest='num_episodes', type=int, default=50000,
                        help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help="The learning rate.")
    parser.add_argument('--r_scale', dest='r_scale', type=float, default=1., help="The absolute scale of rewards.")
    parser.add_argument('--eval_episodes', type=int, default=100, help='number of evaluation episodes')
    parser.add_argument('--train', action='store_true', help='turn on training mode')
    parser.add_argument('--plot', action='store_true', help='turn on plotting')

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render', action='store_true', help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render', action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Train the model using REINFORCE and plot the learning curve.
    reinforce = Reinforce(args, env)

    if args.train:
        reinforce.train(env)
    # elif args.plot:
    #     reinforce.plot(args.model_name)


if __name__ == '__main__':
    # Parse command-line arguments.
    args = parse_arguments()
    main(args)
