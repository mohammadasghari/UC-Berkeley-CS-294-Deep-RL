#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np


def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.loads(f.read())


def find_mean(data):
    mean_episode_reward = []
    for i in range(0, len(data)):
        if i < 100:
            mean_episode_reward.append(np.mean(data[0: i + 1]))
        else:
            mean_episode_reward.append(np.mean(data[-100 + i: i]))

    return mean_episode_reward

def find_best(data):
    best_mean_episode_reward = [data[0]]
    for i in range(1, len(data)):
        if data[i] > best_mean_episode_reward[i - 1]:
            best_mean_episode_reward.append(data[i])
        else:
            best_mean_episode_reward.append(best_mean_episode_reward[i - 1])

    return best_mean_episode_reward

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname1', type=str, default='lander')
    parser.add_argument('envname2', type=str, default='None')
    parser.add_argument('envname3', type=str, default='None')

    args = parser.parse_args()

    print('loading the rewards:1')
    rewards_path_1 = os.path.join('rewards', args.envname1 + '.pkl')
    episode_rewards_1 = load_data(rewards_path_1)['episode_rewards']

    mean_episode_reward_1 = find_mean(episode_rewards_1)
    best_mean_episode_reward_1 = find_best(mean_episode_reward_1)

    print(len(mean_episode_reward_1))

    if args.envname2 != 'None':
        print('loading the rewards:2')
        rewards_path_2 = os.path.join('rewards', args.envname2 + '.pkl')
        episode_rewards_2 = load_data(rewards_path_2)['episode_rewards']

        mean_episode_reward_2 = find_mean(episode_rewards_2)
        best_mean_episode_reward_2 = find_best(mean_episode_reward_2)

        print(len(mean_episode_reward_2))

        min_size = min(len(episode_rewards_1), len(episode_rewards_2))

    if args.envname3 != 'None':
        print('loading the rewards:3')
        rewards_path_3 = os.path.join('rewards', args.envname3 + '.pkl')
        episode_rewards_3 = load_data(rewards_path_3)['episode_rewards']

        mean_episode_reward_3 = find_mean(episode_rewards_3)
        best_mean_episode_reward_3 = find_best(mean_episode_reward_3)

        print(len(mean_episode_reward_3))

        min_size = min(min_size, len(episode_rewards_3))

    # x = range(1, min_size+1)

    x = np.linspace(0, 500000, num=min_size)

    plt.xticks(rotation=15)
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.plot(x, mean_episode_reward_1[:min_size], 'b', label='Buffer size=500000, target update freq=3000')

    if args.envname2 != 'None':
        plt.plot(x, mean_episode_reward_2[:min_size], 'r', label='Buffer size=500000, target update freq=5000')

    if args.envname3 != 'None':
        plt.plot(x, mean_episode_reward_3[:min_size], 'g', label='Buffer size=500000, target update freq=10000')

    # plt.legend(loc='upper left')
    plt.legend(loc='lower right')
    # plt.legend(loc='lower left')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Mean 100-episode reward')
    # plt.ylim(0, 6000)
    plt.savefig("./figures/" + args.envname1 + args.envname2 + args.envname3 + ".png")
    plt.show()


if __name__ == '__main__':
    main()