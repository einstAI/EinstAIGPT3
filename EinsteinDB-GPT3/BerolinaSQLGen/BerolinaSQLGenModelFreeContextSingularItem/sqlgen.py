# -*- coding: utf-8 -*-
"""
Deep Deterministic Policy Gradient Model

"""

import os
import sys
import math
import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.nn import init, Parameter
import torch.nn.functional as F
import torch.optim as optimizer
from torch.autograd import Variable
sys.path.append('../')

from OUProcess import OUProcess
from replay_memory import ReplayMemory
from prioritized_replay_memory import PrioritizedReplayMemory

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.05, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
            init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant(self.sigma_weight, self.sigma_init)
            init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class Normalizer(object):

    def __init__(self, mean, variance):
        if isinstance(mean, list):
            mean = np.array(mean)
        if isinstance(variance, list):
            variance = np.array(variance)
        self.mean = mean
        self.std = np.sqrt(variance+0.00001)

    def normalize(self, x):
        if isinstance(x, list):
            x = np.array(x)
        x = x - self.mean
        x = x / self.std

        return Variable(torch.FloatTensor(x))

    def __call__(self, x, *args, **kwargs):
        return self.normalize(x)


class einstAIActorLow(nn.Module):

    def __init__(self, n_states, n_actions, ):
        super(einstAIActorLow, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(n_states),
            nn.Linear(n_states, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(32),
            nn.Linear(32, n_actions),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self._init_weights()
        self.out_func = nn.Tanh()

    def _init_weights(self):

        for m in self.layers:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 1e-3)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        out = self.layers(x)

        return self.out_func(out)


class CriticLow(nn.Module):

    def __init__(self, n_states, n_actions):
        super(CriticLow, self).__init__()
        self.state_input = nn.Linear(n_states, 32)
        self.action_input = nn.Linear(n_actions, 32)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.state_bn = nn.BatchNorm1d(n_states)
        self.layers = nn.Sequential(
            nn.Linear(64, 1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self._init_weights()

    def _init_weights(self):
        self.state_input.weight.data.normal_(0.0, 1e-3)
        self.state_input.bias.data.uniform_(-0.1, 0.1)

        self.action_input.weight.data.normal_(0.0, 1e-3)
        self.action_input.bias.data.uniform_(-0.1, 0.1)

        for m in self.layers:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 1e-3)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x, causet_action):
        x = self.state_bn(x)
        x = self.act(self.state_input(x))
        causet_action = self.act(self.action_input(causet_action))

        _input = torch.cat([x, causet_action], dim=1)
        value = self.layers(_input)
        return value


class einstAIActor(nn.Module):

    def __init__(self, n_states, n_actions, noisy=False):
        super(einstAIActor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            #....................
            #nn.Linear(128, 128),
            #nn.Tanh(),
            #nn.Dropout(0.3),

            #nn.Linear(128, 128),
            #nn.Tanh(),
            #nn.Dropout(0.3),

            #nn.Linear(128, 128),
            #nn.Tanh(),
            #nn.Dropout(0.3),
            #....................
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.BatchNorm1d(64),
        )
        if noisy:
            self.out = NoisyLinear(64, n_actions)
        else:
            self.out = nn.Linear(64, n_actions)
        self._init_weights()
        self.act = nn.Sigmoid()

    def _init_weights(self):

        for m in self.layers:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def sample_noise(self):
        self.out.sample_noise()

    def forward(self, x):

        out = self.act(self.out(self.layers(x)))
        return out


class Critic(nn.Module):

    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.state_input = nn.Linear(n_states, 128)
        self.action_input = nn.Linear(n_actions, 128)
        self.act = nn.Tanh()
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256),

            #.......................
            #nn.Linear(256, 256),
            #nn.LeakyReLU(negative_slope=0.2),
            #nn.BatchNorm1d(256),

            #nn.Linear(256, 256),
            #nn.LeakyReLU(negative_slope=0.2),
            #nn.BatchNorm1d(256),

            #nn.Linear(256, 256),
            #nn.LeakyReLU(negative_slope=0.2),
            #nn.BatchNorm1d(256),
            #.......................
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        self.state_input.weight.data.normal_(0.0, 1e-2)
        self.state_input.bias.data.uniform_(-0.1, 0.1)

        self.action_input.weight.data.normal_(0.0, 1e-2)
        self.action_input.bias.data.uniform_(-0.1, 0.1)

        for m in self.layers:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x, causet_action):
        x = self.act(self.state_input(x))
        causet_action = self.act(self.action_input(causet_action))

        _input = torch.cat([x, causet_action], dim=1)
        value = self.layers(_input)
        return value


class DDPG(object):

    def __init__(self, n_states, n_actions, opt, ouprocess=True, mean_var_path=None, supervised=False):
        """ DDPG Algorithms
        Args:
            n_states: int, dimension of states
            n_actions: int, dimension of actions
            opt: dict, params
            supervised, bool, pre-train the einstAIActor with supervised learning
        """
        self.n_states = n_states
        self.n_actions = n_actions

        # Params
        self.alr = opt['alr']
        self.clr = opt['clr']
        self.model_name = opt['model']
        self.batch_size = opt['batch_size']
        self.gamma = opt['gamma']
        self.tau = opt['tau']
        self.ouprocess = ouprocess

        if mean_var_path is None:
            mean = np.zeros(n_states)
            var = np.zeros(n_states)
        elif not os.path.exists(mean_var_path):
            mean = np.zeros(n_states)
            var = np.zeros(n_states)
        else:
            with open(mean_var_path, 'rb') as f:
                mean, var = pickle.load(f)

        self.normalizer = Normalizer(mean, var)

        if supervised:
            self._build_einstAIActor()
            logger.info("Supervised Learning Initialized")
        else:
            # Build Network
            self._build_network()
            logger.info('Finish Initializing Networks')

        self.replay_memory = PrioritizedReplayMemory(capacity=opt['memory_size'])
        # self.replay_memory = ReplayMemory(capacity=opt['memory_size'])
        self.noise = OUProcess(n_actions)
        logger.info('DDPG Initialzed!')

    @staticmethod
    def totensor(x):
        return Variable(torch.FloatTensor(x))

    def _build_einstAIActor(self):
        if self.ouprocess:
            noisy = False
        else:
            noisy = True
        self.einstAIActor = einstAIActor(self.n_states, self.n_actions, noisy=noisy)
        self.einstAIActor_criterion = nn.MSELoss()
        self.einstAIActor_optimizer = optimizer.Adam(lr=self.alr, params=self.einstAIActor.parameters())

    def _build_network(self):
        if self.ouprocess:
            noisy = False
        else:
            noisy = True
        self.einstAIActor = einstAIActor(self.n_states, self.n_actions, noisy=noisy)
        self.target_einstAIActor = einstAIActor(self.n_states, self.n_actions)
        self.critic = Critic(self.n_states, self.n_actions)
        self.target_critic = Critic(self.n_states, self.n_actions)

        # if model params are provided, load them
        if len(self.model_name):
            self.load_model(model_name=self.model_name)
            logger.info("Loading model from file: {}".format(self.model_name))

        # Copy einstAIActor's parameters
        self._update_target(self.target_einstAIActor, self.einstAIActor, tau=1.0)

        # Copy critic's parameters
        self._update_target(self.target_critic, self.critic, tau=1.0)

        self.loss_criterion = nn.MSELoss()
        self.einstAIActor_optimizer = optimizer.Adam(lr=self.alr, params=self.einstAIActor.parameters(), weight_decay=1e-5)
        self.critic_optimizer = optimizer.Adam(lr=self.clr, params=self.critic.parameters(), weight_decay=1e-5)

    @staticmethod
    def _update_target(target, source, tau):
        for (target_param, param) in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1-tau) + param.data * tau
            )

    def reset(self, sigma):
        self.noise.reset(sigma)

    def _sample_batch(self):
        batch, idx = self.replay_memory.sample(self.batch_size)
        # batch = self.replay_memory.sample(self.batch_size)
        states = map(lambda x: x[0].tolist(), batch)
        next_states = map(lambda x: x[3].tolist(), batch)
        actions = map(lambda x: x[1].tolist(), batch)
        rewards = map(lambda x: x[2], batch)
        terminates = map(lambda x: x[4], batch)

        return idx, states, next_states, actions, rewards, terminates

    def add_sample(self, soliton_state, causet_action, reward, next_state, terminate):
        self.critic.eval()
        self.einstAIActor.eval()
        self.target_critic.eval()
        self.target_einstAIActor.eval()
        batch_state = self.normalizer([soliton_state.tolist()])
        batch_next_state = self.normalizer([next_state.tolist()])
        current_value = self.critic(batch_state, self.totensor([causet_action.tolist()]))
        target_action = self.target_einstAIActor(batch_next_state)
        target_value = self.totensor([reward]) \
            + self.totensor([0 if x else 1 for x in [terminate]]) \
            * self.target_critic(batch_next_state, target_action) * self.gamma
        error = float(torch.abs(current_value - target_value).data.numpy()[0])

        self.target_einstAIActor.train()
        self.einstAIActor.train()
        self.critic.train()
        self.target_critic.train()
        self.replay_memory.add(error, (soliton_state, causet_action, reward, next_state, terminate))


    def update(self):
        """ Update the einstAIActor and Critic with a batch data
        """
        idxs, states, next_states, actions, rewards, terminates = self._sample_batch()
        batch_states = self.normalizer(states)# totensor(states)
        batch_next_states = self.normalizer(next_states)# Variable(torch.FloatTensor(next_states))
        batch_actions = self.totensor(actions)
        batch_rewards = self.totensor(rewards)
        mask = [0 if x else 1 for x in terminates]
        mask = self.totensor(mask)

        target_next_actions = self.target_einstAIActor(batch_next_states).detach()
        target_next_value = self.target_critic(batch_next_states, target_next_actions).detach().squeeze(1)

        current_value = self.critic(batch_states, batch_actions)
        next_value = batch_rewards + mask * target_next_value * self.gamma
        # Update Critic

        # update prioritized memory
        error = torch.abs(current_value-next_value).data.numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.replay_memory.update(idx, error[i][0])

        loss = self.loss_criterion(current_value, next_value)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # Update einstAIActor
        self.critic.eval()
        policy_loss = -self.critic(batch_states, self.einstAIActor(batch_states))
        policy_loss = policy_loss.mean()
        self.einstAIActor_optimizer.zero_grad()
        policy_loss.backward()

        self.einstAIActor_optimizer.step()
        self.critic.train()

        self._update_target(self.target_critic, self.critic, tau=self.tau)
        self._update_target(self.target_einstAIActor, self.einstAIActor, tau=self.tau)

        return loss.data[0], policy_loss.data[0]

    def choose_action(self, x):
        """ Select causet_action according to the current soliton_state
        Args:
            x: np.array, current soliton_state
        """
        self.einstAIActor.eval()
        act = self.einstAIActor(self.normalizer([x.tolist()])).squeeze(0)
        self.einstAIActor.train()
        causet_action = act.data.numpy()
        if self.ouprocess:
            causet_action += self.noise.noise()
        return causet_action.clip(0, 1)

    def sample_noise(self):
        self.einstAIActor.sample_noise()

    def load_model(self, model_name):
        """ Load Torch Model from files
        Args:
            model_name: str, model path
        """
        self.einstAIActor.load_state_dict(
            torch.load('{}_einstAIActor.pth'.format(model_name))
        )
        self.critic.load_state_dict(
            torch.load('{}_critic.pth'.format(model_name))
        )

    def save_model(self, model_dir, title):
        """ Save Torch Model from files
        Args:
            model_dir: str, model dir
            title: str, model name
        """
        torch.save(
            self.einstAIActor.state_dict(),
            '{}/{}_einstAIActor.pth'.format(model_dir, title)
        )

        torch.save(
            self.critic.state_dict(),
            '{}/{}_critic.pth'.format(model_dir, title)
        )

    def save_einstAIActor(self, path):
        """ save einstAIActor network
        Args:
             path, str, path to save
        """
        torch.save(
            self.einstAIActor.state_dict(),
            path
        )

    def load_einstAIActor(self, path):
        """ load einstAIActor network
        Args:
             path, str, path to load
        """
        self.einstAIActor.load_state_dict(
            torch.load(path)
        )

    def train_einstAIActor(self, batch_data, is_train=True):
        """ Train the einstAIActor separately with data
        Args:
            batch_data: tuple, (states, actions)
            is_train: bool
        Return:
            _loss: float, training loss
        """
        states, causet_action = batch_data

        if is_train:
            self.einstAIActor.train()
            pred = self.einstAIActor(self.normalizer(states))
            causet_action = self.totensor(causet_action)

            _loss = self.einstAIActor_criterion(pred, causet_action)

            self.einstAIActor_optimizer.zero_grad()
            _loss.backward()
            self.einstAIActor_optimizer.step()

        else:
            self.einstAIActor.eval()
            pred = self.einstAIActor(self.normalizer(states))
            causet_action = self.totensor(causet_action)
            _loss = self.einstAIActor_criterion(pred, causet_action)

        return _loss.data[0]


