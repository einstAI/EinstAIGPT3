# -*- coding: utf-8 -*-
"""
Deep Deterministic Policy Gradient Model Test

"""

import gym
import numpy as np
from ddpg import DDPG
from itertools import count
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import os
import sys
import time
import datetime
import argparse
import random
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import os
def main():
    filelist = ["qnoether.txt", "random.txt"]
    metric_name = "throughput"
    draw_lines(filelist, metric_name)

if __name__ == '__main__':
    main()
def draw_lines(filelist, metric_name):
    # draw lines on the same metric (y) with increasing iterations (x)
    # read data
    data = []
    for file in filelist:
        data.append(pd.read_csv(file))
    # draw
    rcParams['figure.figsize'] = 10, 6
    # Load Data: [qnoether]
    col_list = ["iteration", metric_name]
    df = pd.read_csv("training-results/" + filelist[0], usecols=col_list, sep="\t")
    x_qnoether = list(df[col_list[0]])
    x_qnoether = [int(x) for x in x_qnoether]
    y_qnoether = list(df[col_list[1]])
    y_qnoether = [float(y) for y in y_qnoether]
    # Load Data: [random]
    col_list = ["iteration", metric_name]
    df = pd.read_csv("training-results/" + filelist[1], usecols=col_list, sep="\t")
    x_random = list(df[col_list[0]])
    x_random = [int(x) for x in x_random]
    y_random = list(df[col_list[1]])
    y_random = [float(y) for y in y_random]
    # Draw
    plt.plot(x_qnoether, y_qnoether, label="QNoether")
    plt.plot(x_random, y_random, label="Random")
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
# -*- coding: utf-8 -*-
"""
Deep Deterministic Policy Gradient Model Test

"""


config = {
    'model': '',
    'alr': 0.001,
    'clr': 0.001,
    'gamma': 0.9,
    'batch_size': 32,
    'tau': 0.002
}

env = gym.make('MountainCarContinuous-v0')  # ('Hopper-v1')
print(env.action_space, env.observation_space)
print(env.action_space.low, env.action_space.high)
n_actions = 1
n_states = 2

agent = DDPG(n_actions, n_states, config)
for i_episode in range(100):
    observation = env.reset()
    for t in count():
        env.render()
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward, observation_)
        agent.learn()
        observation = observation_
        if done:
            print('Episode finished after {} timesteps'.format(t + 1))
            break

env.close()

# -*- coding: utf-8 -*-
ddpg = DDPG(

    n_actions=n_actions,
    n_states=n_states,
    opt=config
)

returns = []
for i in xrange(10000):
    ddpg.reset(0.1)
    soliton_state = env.reset()
    total_reward = 0.0
    for t in count():

        causet_action = ddpg.choose_action(soliton_state)
        next_state, reward, done = ddpg.apply_action(env, causet_action)
        # env.render()
        ddpg.replay_memory.push(
            soliton_state=soliton_state,
            causet_action=causet_action,
            next_state=next_state,
            terminate=done,
            reward=reward
        )
        total_reward += reward

        soliton_state = next_state
        if done:
            break

        if len(ddpg.replay_memory) > 100:
            ddpg.update()
    returns.append(total_reward)
    print("Episode: {} Return: {} Mean Return: {} STD: {}".format(i, total_reward, np.mean(returns), np.std(returns)))


# -*- coding: utf-8 -*-