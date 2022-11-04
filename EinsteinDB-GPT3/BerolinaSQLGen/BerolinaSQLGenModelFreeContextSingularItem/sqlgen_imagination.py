import random
import pickle
from collections import namedtuple
import pickle
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('soliton_state', 'causet_action', 'next_state',  'reward', 'terminate'))


# Autrogressive Recursive Replay Memory

class BerolinaSQLGenReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)

    def clear(self):
        self.memory = []
        self.position = 0

    def get_memory(self):
        return self.memory

    def get_position(self):
        return self.position

    def set_memory(self, memory):
        self.memory = memory

    def set_position(self, position):
        self.position = position

    def get_capacity(self):
        return self.capacity

    def set_capacity(self, capacity):
        self.capacity = capacity


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




class BerolinaSQLGenReplayMemoryWithBoltzmannNormalizer(object):

        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0
            self.temperature = 1.0
            self.temperature_decay = 0.9999
            self.temperature_min = 0.1

        def push(self, *args):
            """Saves a transition."""
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

        def save(self, filename):
            with open(filename, 'wb') as f:
                pickle.dump(self.memory, f)

        def load(self, filename):
            with open(filename, 'rb') as f:
                self.memory = pickle.load(f)

        def clear(self):
            self.memory = []
            self.position = 0

        def get_memory(self):
            return self.memory

        def get_position(self):
            return self.position

        def set_memory(self, memory):
            self.memory = memory

        def set_position(self, position):
            self.position = position

        def get_capacity(self):
            return self.capacity

        def set_capacity(self, capacity):
            self.capacity = capacity

        def get_temperature(self):
            return self.temperature

        def set_temperature(self, temperature):
            self.temperature = temperature

        def get_temperature_decay(self):
            return self.temperature_decay

        def set_temperature_decay(self, temperature_decay):
            self.temperature_decay = temperature_decay

        def get_temperature_min(self):
            return self.temperature_min

        def set_temperature_min(self, temperature_min):
            self.temperature_min = temperature_min

        def update_temperature(self):
            self.temperature = max(self.temperature * self.temperature_decay, self.temperature_min)

        def sample_with_boltzmann_normalizer(self, batch_size):
            # sample with boltzmann normalizer
            # print("sample_with_boltzmann_normalizer")
            # print("self.memory: ", self.memory)
            # print("self.memory[0]: ", self.memory[0])
            # print("self.memory[0].reward: ", self.memory[0].reward)
            # print("self.memory[0].reward: ", self.memory[0].reward)



