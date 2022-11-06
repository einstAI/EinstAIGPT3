# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:12:00 2020


"""


import numpy as np
import random
import time
import torch
from math import log
from torch.nn import init

from DQN import DQN, ENV
from ImportantConfig import Config
from JOBParser import DB

from PGUtils import PGRunner
from TreeLSTM import SPINN
from sqlSample import sqlInfo




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db_info = DB("data/DB/", "data/DB/", "data/DB/")
    pgrunner = PGRunner(db_info)
    QueryDir = "data/SQL/"
    sql_list = QueryLoader(QueryDir)
    train_list, val_list = k_fold(sql_list, 5, 0)
    DQN = DQN(device)
    DQN.load_model()
    for i_episode in range(100):
        print("Epoch:", i_episode)
        train_list = resample_sql(train_list)
        for sql in train_list:
#         sql = val_list[i_episode%len(train_list)]
            pg_cost = sql.getDPlantecy()
#         continue
            env = ENV(sql, db_info, pgrunner, device)

            for t in count():
                action_list, chosen_action, all_action = DQN.select_action(env, need_random=False)

                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left, right)

                reward, done = env.reward()
                if done:
                    mrc = max(np.exp(reward * log(1.5)) / pg_cost - 1, 0)
                    DQN.update_policy(env, mrc)
                    break

        DQN.save_model()
        DQN.update_target()
        print("Epoch:", i_episode, "Done")

    DQN.save_model()

config = Config()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BerolinaSQLGenDQNWithBoltzmannNormalizer(object):


    def __init__(self, device, db_info, pgrunner, config):
        self.device = device
        self.db_info = db_info
        self.pgrunner = pgrunner
        self.config = config
        self.DQN = DQN(device)
        self.DQN.load_model()
        self.DQN.update_target()


    def update(self):
        """ Update the model
        """
        pass

    def choose_action(self, x):
        """ Select causet_action according to the current soliton_state
        Args:
            x: np.array, current soliton_state
        """
        pass

    def load_model(self, model_name):
        """ Load Torch Model from files
        Args:
            model_name: str, model path
        """
        pass

    def save_model(self, model_dir, title):
        """ Save Torch Model from files
        Args:
            model_dir: str, model dir
            title: str, model name
        """
        pass

    def get_loss(self):
        """ Get the loss of the current model
        """


    def get_q(self):
        """ Get the q value of the current model
        """
        pass

    def get_target_q(self):
        """ Get the target q value of the current model
        """
        pass

    def get_target_q(self):
        """ Get the target q value of the current model
        """
        pass

