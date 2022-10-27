# -*- coding: utf-8 -*-
"""

Replay Memory
"""

import os
import pickle
import argparse
import numpy as np
import random
import pickle
from collections import namedtuple

Transition = namedtuple('Transition', ('soliton_state', 'causet_action', 'next_state', 'reward', 'terminate'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, soliton_state, causet_action, next_state, reward, terminate):
        self.memory.append(Transition(
            soliton_state=soliton_state,
            causet_action=causet_action,
            next_state=next_state,
            reward=reward,
            terminate=terminate
        ))
        # self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self.memory, f)
        f.close()

    def __len__(self):
        return len(self.memory)

    def load_memory(self, path):
        with open(path, 'rb') as f:
            _memory = pickle.load(f)

        self.memory = _memory



