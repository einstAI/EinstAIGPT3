# Copyright (c) EINSTAI Inc. 2022-2023
# Path: EINSTAI/OUCausetFlowProcess/LatencyTuning.py
# Compare this snippet from EINSTAI/OUCausetFlowProcess/CostTraining.py:
from EINSTAI.OUCausetFlowProcess.CostTraining import policy_net, target_net, device
from PGUtils import pgrunner

from PGUtils import db_info
from sqlSample import sqlInfo
import numpy as np
from itertools import count
from math import log
import random
import time
from OrnsteinUhlenbeckProcessFlow import DQN, ENV
from TreeLSTM import SPINN
import os
from BerolinaSQLGenDQNWithBoltzmannNormalizer import DB
import copy
import torch
import gym
from torch.nn import init
from tconfig import Config


policy_net.load_state_dict(torch.load("CostTraining.pth"))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

def k_fold(input_list, k, ix=0):
    li = len(input_list)
    kl = (li - 1) // k + 1
    train = []
    validate = []
    for idx in range(li):

        if idx % k == ix:
            validate.append(input_list[idx])
        else:
            train.append(input_list[idx])
    return train, validate


def get_sql_list():
    sql_list = []
    for i in range(100):
        sql = sqlInfo()
        sql.genSQL(db_info)
        sql_list.append(sql)
    return sql_list

def resample_sql(sql_list):
    rewards = []
    reward_sum = 0
    rewardsP = []
    mes = 0
    for sql in sql_list:
        #         sql = val_list[i_episode%len(train_list)]
        pg_cost = sql.getDPlantecy()
        #         continue
        env = ENV(sql, db_info, pgrunner, device)

        for t in count():
            action_list, chosen_action, all_action = DQN.select_action(env, need_random=False)

            left = chosen_action[0]
            right = chosen_action[1]
            env.takeAction(left, right)

            reward, done = env.reward_new()
            if done:
                mrc = max(reward / pg_cost - 1, 0)
                rewardsP.append(reward / pg_cost)
                mes += log(reward) - log(pg_cost)
                rewards.append((mrc, sql))
                reward_sum += mrc
                break
    import random
    print(rewardsP)
    res_sql = []
    print(mes / len(sql_list))
    for idx in range(len(sql_list)):
        rd = random.random() * reward_sum
        for ts in range(len(sql_list)):
            rd -= rewards[ts][0]
            if rd < 0:
                res_sql.append(rewards[ts][1])
                break
    return res_sql + sql_list


def train(trainSet, validateSet):
    trainSet_temp = trainSet
    losses = []
    startTime = time.time()
    print_every = 20
    TARGET_UPDATE = 3
    for i_episode in range(0, 10000):
        if i_episode % 200 == 100:
            trainSet = resample_sql(trainSet_temp)
        #     sql = random.sample(train_list_back,1)[0][0]
        sqlt = random.sample(trainSet[0:], 1)[0]
        pg_cost = sqlt.getDPlantecy()
        env = ENV(sqlt, db_info, pgrunner, device)

        previous_state_list = []
        action_this_epi = []
        nr = True
        nr = random.random() > 0.3 or sqlt.getBestOrder() == None
        acBest = (not nr) and random.random() > 0.7
        for t in count():
            # beginTime = time.time();
            action_list, chosen_action, all_action = DQN.select_action(env, need_random=nr)
            value_now = env.selectValue(policy_net)
            next_value = torch.min(action_list).detach()
            # e1Time = time.time()
            env_now = copy.deepcopy(env)
            # endTime = time.time()
            # print("make",endTime-startTime,endTime-e1Time)
            if acBest:
                chosen_action = sqlt.getBestOrder()[t]
            left = chosen_action[0]
            right = chosen_action[1]
            env.takeAction(left, right)
            action_this_epi.append((left, right))

            reward, done = env.reward_new()
            reward = torch.tensor([reward], device=device, dtype=torch.float32).view(-1, 1)

            previous_state_list.append((value_now, next_value.view(-1, 1), env_now))
            if done:
                #             print("done")
                next_value = 0
                sqlt.updateBestOrder(reward.item(), action_this_epi)

            expected_state_action_values = (next_value) + reward.detach()
            final_state_value = (next_value) + reward.detach()

            if done:
                cnt = 0
                DQN.Memory.push(env, expected_state_action_values, final_state_value)
                for pair_s_v in previous_state_list[:0:-1]:
                    cnt += 1
                    if expected_state_action_values > pair_s_v[1]:
                        expected_state_action_values = pair_s_v[1]
                    #                 for idx in range(cnt):
                    expected_state_action_values = expected_state_action_values
                    DQN.Memory.push(pair_s_v[2], expected_state_action_values, final_state_value)
                #                 break
                loss = 0

            if done:
                # break
                loss = DQN.optimize_model()
                loss = DQN.optimize_model()
                loss = DQN.optimize_model()
                loss = DQN.optimize_model()
                losses.append(loss)
                if ((i_episode + 1) % print_every == 0):
                    print(np.mean(losses))
                    print("######################Epoch", i_episode // print_every, pg_cost)
                    val_value = DQN.validate(validateSet)
                    print("time", time.time() - startTime)
                    print("~~~~~~~~~~~~~~")
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.cpu().state_dict(), 'LatencyTuning.pth')


if __name__ == '__main__':
    sytheticQueries = QueryLoader(QueryDir=config.sytheticDir)
    # print(sytheticQueries)
    JOBQueries = QueryLoader(QueryDir=config.JOBDir)
    Q4, Q1 = k_fold(JOBQueries, 10, 1)
    # print(Q4,Q1)
    train(Q4 + sytheticQueries, Q1)
