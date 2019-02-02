#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20


    python3.5 run_behavioral_cloning.py expert_data/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 10
    python3.5 run_behavioral_cloning.py expert_data/Ant-v2.pkl Ant-v2 --num_rollouts 10
    python3.5 run_behavioral_cloning.py expert_data/Hopper-v2.pkl Hopper-v2 --num_rollouts 10
    python3.5 run_behavioral_cloning.py expert_data/Reacher-v2.pkl Reacher-v2 --num_rollouts 10
    python3.5 run_behavioral_cloning.py expert_data/Walker2d-v2.pkl Walker2d-v2 --num_rollouts 10


Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


def init_tf_sess():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return sess


def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.loads(f.read())


def build_model(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):

    with tf.variable_scope(scope):

        layer = input_placeholder

        for i in range(0, n_layers):
            layer = tf.layers.dense(layer, size, activation=activation)

        output_placeholder = tf.layers.dense(layer, output_size, activation=output_activation)

    return output_placeholder


def placeholders(input_size, output_size):

    input_ph = tf.placeholder(name='input', shape=[None, input_size], dtype=tf.float32)
    output_ph = tf.placeholder(name='output', shape=[None, output_size], dtype=tf.float32)

    return [input_ph, output_ph]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert dataset')
    expert_data = load_data(args.expert_policy_file)

    observations = np.array(expert_data['observations'])
    actions = expert_data['actions']
    actions = np.squeeze(actions, axis=1)
    print('loaded and two numpy arrays of observations and action built')

    data_size = np.shape(observations)[0]
    obs_size = np.shape(observations)[1]
    actions_size = np.shape(actions)[1]

    # creating placeholders for input and output
    input_placeholder, output_placeholder = placeholders(obs_size, actions_size)

    # creating the model
    output_pred = build_model(input_placeholder=input_placeholder,
                        output_size=actions_size,
                        scope='bc',
                        n_layers=2,
                        size=32,
                        activation=tf.tanh,
                        output_activation=None)

    mse_loss = tf.reduce_mean(0.5 * tf.square(output_placeholder - output_pred))

    update_op = tf.train.AdamOptimizer().minimize(mse_loss)

    # start training

    sess = init_tf_sess()

    batch_size = 32
    num_steps = 10000

    for training_step in range(num_steps):

        indices = np.random.randint(low=0, high=data_size, size=batch_size)

        input_data = observations[indices]
        output_data = actions[indices]

        _, mse_run = sess.run([update_op, mse_loss], feed_dict={input_placeholder: input_data,
                                                            output_placeholder: output_data})

        if training_step % 100 == 0:
            print('{0:04d} mse,{1:0.3f}'.format(training_step, mse_run))

    # apply the trained model to the environment

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run(output_pred, feed_dict={input_placeholder: obs[None]})
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    returns_data = {'returns': returns}

    with open(os.path.join('returns_data', args.envname + 'behavioral_cloning' + '.pkl'), 'wb') as f:
        pickle.dump(returns_data, f)


if __name__ == '__main__':
    main()
