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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str)
    parser.add_argument('envname', type=str)

    args = parser.parse_args()

    print('loading the rewards')
    expert_path = os.path.join('returns_data', args.envname + 'expert' + '.pkl')
    expert_data = load_data(expert_path)['returns']

    method_path = os.path.join('returns_data', args.envname + args.method + '.pkl')
    method_data = load_data(method_path)['returns']

    x = range(1, 11)

    plt.plot(x, expert_data, 'b', label='expert agent')
    plt.plot(x, method_data, 'r', label=args.method + ' agent')
    plt.legend(loc='lower right')
    plt.ylim(0, 6000)
    plt.savefig("./figures/expert_" + args.envname + args.method + ".png")
    plt.show()


if __name__ == '__main__':
    main()