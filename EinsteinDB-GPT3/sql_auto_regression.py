# -*- coding: utf-8 -*-
"""

Prioritized Replay Memory
"""
import random
import pickle
import numpy as np
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

from torch.nn import Sequential
from torch.optim import Adam


def main():
    filelist = ["qnoether.txt", "random.txt"]
    metric_name = "throughput"
    draw_lines(filelist, metric_name)


if __name__ == '__main__':
    main()

# Compare this snippet from EinsteinDB-GPT3/BerolinaSQLGen/BerolinaSQLGenModelFreeContextSingularItem/OUProcess.py:
# # -*- coding: utf-8 -*-
# """
# Deep Deterministic Policy Gradient Model Test
#
# """
#
# import gym
# import numpy as np
# from ddpg import DDPG
# from itertools import count
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn


# -*- coding: utf-8 -*-



# Compare this snippet from EinsteinDB-GPT3/BerolinaSQLGen/BerolinaSQLGenModelFreeContextSingularItem/OUProcess.py:


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


class SumTree(object):
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.num_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) / 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def __len__(self):
        return self.num_entries


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self):
        self.write = None
        self.capacity = None
        self.tree = None
        self.data = None
        self.num_entries = None

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def sample(self, n):
        batch = []
        segment = self.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            batch.append(self.get(s))
        return batch

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.e
        clipped_errors = np.minimum(abs_errors, 1)
        ps = np.power(clipped_errors, self.a)

        for ti, p in zip(tree_idx, ps):
            self.update(ti, p)

    def get_priority(self, error):
        return (error + self.e) ** self.a

    def __len__(self):
        return self.num_entries

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return [idx, self.tree[idx], self.data[data_idx]]


class PrioritizedReplayMemory(object):

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        # (s, a, r, s, t)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def __len__(self):
        return self.tree.num_entries

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        return batch, idxs

        # sampling_probabilities = priorities / self.tree.total()
        # is_weight = np.power(self.tree.num_entries * sampling_probabilities, -self.beta)
        # is_weight /= is_weight.max()

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class Dense:
    # Dense is a spectral clustering algorithm that uses a dense graph representation with cellular automata.
    # The algorithm is described in the paper "Dense: A Spectral Clustering Algorithm for Large Graphs" by
    # J. Scott, J. B. Tenenbaum, and J. K. Udupa.
    # The algorithm is based on the idea of using a cellular automaton to evolve a graph representation of the data
    # into a clustering. The algorithm is very fast and scales well with the number of data points.
    # The algorithm is described in the paper "Dense: A Spectral Clustering Algorithm for Large Graphs"

    def __init__(self, n_clusters=2, max_iter=100, n_init=1, gamma=1.0, n_neighbors=10, n_jobs=1, verbose=False):
        # n_clusters: The number of clusters to form as well as the number of centroids to generate.
        # max_iter: Maximum number of iterations of the k-means algorithm for a single run.
        # n_init: Number of time the k-means algorithm will be run with different centroid seeds.

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X):
        """


        :param X:
        :return:
        """

        for i in range(self.n_init):
            self._fit(X)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayMemory(100000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.tau = .125

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump({"tree": self.tree}, f)
        f.close()

    def load(self, path):
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
        self.tree = data["tree"]

    def load_memory(self, path):
        with open(path, 'rb') as f:
            _memory = pickle.load(f)
        self.tree = _memory['tree']

    def save_memory(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'tree': self.tree}, f)

    def remember(self, state, action, reward, next_state, done):
        # add experience to memory
        self.memory.add(0, (state, action, reward, next_state, done))

    def act(self, state):
        # act
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # train
        minibatch, idxs = self.memory.sample(batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])

        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        target = self.model.predict(states)
        target_next = self.model.predict(next_states)
        target_val = self.target_model.predict(next_states)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                a = np.argmax(target_next[i])
                target[i][actions[i]] = rewards[i] + self.gamma * (target_val[i][a])

        errors = np.abs(target - self.model.predict(states))
        for i in range(len(idxs)):
            idx = idxs[i]
            error = errors[i]
            self.memory.update(idx, error)

            # Train the Neural Network with batches
        self.model.fit(states, target, batch_size=batch_size, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # update target model
        self.update_target_model()
        # save model
        self.model.save("model.h5")
        # save memory
        self.save_memory("memory.pkl")
        # save tree
        self.save("tree.pkl")
        # save epsilon
        with open("epsilon.pkl", 'wb') as f:
            pickle.dump({'epsilon': self.epsilon}, f)

    def load_model(self, name):
        self.model = self.load_model(name)
