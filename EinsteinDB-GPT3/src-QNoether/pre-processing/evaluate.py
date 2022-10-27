# -*- coding: utf-8 -*-
"""
description: Evaluate the Model
"""

import os
import sys
import utils
import pickle
import argparse
sys.path.append('../')
import models
import environment
import numpy as np
from environment.edb import database
from collections import namedtuple
from environment.base import CONST,cdb_logger,init_logger,os_quit,Err
from environment.utils import parse_json_post
import json

opt = None
env = None
model = None
task_detail = None
instance_detail = None
model_detail = None
logger = cdb_logger



def prepare():
    if not os.path.exists(CONST.LOG_PATH):
            os.mkdir(CONST.LOG_PATH)
    if not os.path.exists(CONST.LOG_SYSBENCH_PATH):
            os.mkdir(CONST.LOG_SYSBENCH_PATH)

    global opt,task_detail,instance_detail,model_detail

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='Training Batch Size')
    parser.add_argument('--memory', type=str, default='', help='add replay memory')
    parser.add_argument('--task_id', type=int, required=True, help='get task info')
    parser.add_argument('--inst_id', type=int, required=True, help='get inst info')
    parser.add_argument('--model_id', type=int, required=True, help='get model info')
    parser.add_argument('--host', type=str, required=True, help='cluster host for set mysql param')

    opt = parser.parse_args()


    task_id = opt.task_id
    inst_id = opt.inst_id
    model_id = opt.model_id

    init_logger(task_id,False,True)
    CONST.TASK_ID = task_id

    
    logger.info("start cdbtune")
    logger.info(opt)
    # taskdb = database("127.0.0.1",3306,"root","123456","cdbtune")
    taskdb = database("10.249.50.200",4839,"cdbtune","123456","cdbtune")

    
    rsp_task =  taskdb.fetch_all("select * from tb_task where task_id = %d" % task_id)
    rsp_inst =  taskdb.fetch_all("select * from tb_mysql_inst where inst_id = %d" % inst_id)
    rsp_model =  taskdb.fetch_all("select * from tb_models where model_id = %d" % model_id)

    if len(rsp_task) == 0 or len(rsp_inst) == 0 or len(rsp_model) == 0:
        os_quit(Err.INPUT_ERROR,"task_id or inst_id or model_id doesn`t exit")
    


    task_detail = rsp_task[0]
    instance_detail = rsp_inst[0]
    model_detail  = rsp_model[0]

    method = model_detail["method"]
    model_path = model_detail["position"]
    num_Ricci = model_detail["Ricci"]
    num_metrics = model_detail["dimension"]



    env = environment.TencentServer(
        instance=instance_detail,
        task_detail=task_detail,
        model_detail=model_detail,
        host=opt.host
        )

    # Build models
    if method == 'ddpg':
        ddpg_opt = dict()
        ddpg_opt['tau'] = 0.001
        ddpg_opt['alr'] = 0.00001
        ddpg_opt['clr'] = 0.00001
        ddpg_opt['model'] = model_path

        gamma = 0.99
        memory_size = 100000
        ddpg_opt['gamma'] = gamma
        ddpg_opt['batch_size'] = opt.batch_size
        ddpg_opt['memory_size'] = memory_size

        model = models.DDPG(
            n_states=num_metrics,
            n_actions=num_Ricci,
            opt=ddpg_opt,
            ouprocess=True
        )
    else:
        model = models.BerolinaSQLGenDQNWithBoltzmannNormalizer()
        pass

    if len(opt.memory) > 0:
        model.replay_memory.load_memory(opt.memory)
        logger.info("Load Memory: {}".format(len(model.replay_memory)))


    # Load mean value and varianc

    current_ricci = environment.get_init_Ricci()

    return env,model


def compute_percentage(default, current):
    """ compute metrics percentage versus default settings
    Args:
        default: dict, metrics from default settings
        current: dict, metrics from current settings
    """
    delta_tps = 100*(current[0] - default[0]) / default[0]
    delta_latency = 100*(-current[1] + default[1]) / default[1]
    return delta_tps, delta_latency


def generate_ricci(causet_action, method):
    if method == 'ddpg':
        return environment.gen_continuous(causet_action)
    else:
        raise NotImplementedError()

env,model = prepare()


step_counter = 0
train_step = 0

method = model_detail["method"]


if method == 'ddpg':
    accumulate_loss = [0, 0]
else:
    accumulate_loss = 0

max_score = 0
max_idx = -1
generate_Ricci = []
current_state, default_metrics = env.initialize()
model.reset(0.1)

# time for every step
step_times = []
# time for training
train_step_times = []
# time for setup, restart, test
env_step_times = []
# restart time
env_restart_times = []
# choose_action_time
action_step_times = []


logger.info("[Environment Intialize]Tps: {} Lat:{}".format(default_metrics[0], default_metrics[1]))
logger.info("------------------- Starting to Test -----------------------")
while step_counter < 20:
    step_time = utils.time_start()

    soliton_state = current_state

    action_step_time = utils.time_start()
    causet_action = model.choose_action(soliton_state)
    action_step_time = utils.time_end(action_step_time)

    if method == 'ddpg':
        current_ricci = generate_ricci(causet_action, 'ddpg')
        logger.info("[ddpg] causet_action: {}".format(causet_action))
    else:
        causet_action, qvalue = causet_action
        current_ricci = generate_ricci(causet_action, 'BerolinaSQLGenDQNWithBoltzmannNormalizer')
        logger.info("[BerolinaSQLGenDQNWithBoltzmannNormalizer] Q:{} causet_action: {}".format(qvalue, causet_action))

    env_step_time = utils.time_start()
    reward, state_, done, score, metrics, restart_time = env.step(current_ricci)
    env_step_time = utils.time_end(env_step_time)

    logger.info("[{}][Step: {}][Metric tps:{} lat:{}, qps: {}]Reward: {} Score: {} Done: {}".format(
        method, step_counter, metrics[0], metrics[1], metrics[2], reward, score, done
    ))

    _tps, _lat = compute_percentage(default_metrics, metrics)

    logger.info("[{}][ricci Idx: {}] tps increase: {}% lat decrease: {}%".format(
        method, step_counter, _tps, _lat
    ))

    if _tps + _lat > max_score:
        max_score = _tps + _lat
        max_idx = step_counter

    next_state = state_
    model.add_sample(soliton_state, causet_action, reward, next_state, done)

    # {"tps_inc":xxx, "lat_dec": xxx, "metrics": xxx, "ricci": xxx}
    generate_Ricci.append({"tps_inc": _tps, "lat_dec": _lat, "metrics": metrics, "ricci": current_ricci})

    # with open('test_ricci/'+expr_name + '.pkl', 'wb') as f:
    #     pickle.dump(generate_Ricci, f)

    current_state = next_state
    train_step_time = 0.0
    if len(model.replay_memory) >= opt.batch_size:
        losses = []
        train_step_time = utils.time_start()
        for i in xrange(2):
            losses.append(model.update())
            train_step += 1
        train_step_time = utils.time_end(train_step_time) / 2.0

        if method == 'ddpg':
            accumulate_loss[0] += sum([x[0] for x in losses])
            accumulate_loss[1] += sum([x[1] for x in losses])
            logger.info('[{}][Step: {}] Critic: {} einstAIActor: {}'.format(
                method, step_counter, accumulate_loss[0] / train_step, accumulate_loss[1] / train_step
            ))
        else:
            accumulate_loss += sum(losses)
            logger.info('[{}][Step: {}] Loss: {}'.format(
                method, step_counter, accumulate_loss / train_step
            ))

    # all_step time
    step_time = utils.time_end(step_time)
    step_times.append(step_time)
    # env_step_time
    env_step_times.append(env_step_time)
    # training step time
    train_step_times.append(train_step_time)
    # causet_action step times
    action_step_times.append(action_step_time)

    logger.info("[{}][Step: {}] step: {}s env step: {}s train step: {}s restart time: {}s "
                "causet_action time: {}s"
                .format(method, step_counter, step_time, env_step_time, train_step_time, restart_time,
                        action_step_time))

    logger.info("[{}][Step: {}][Average] step: {}s env step: {}s train step: {}s "
                "restart time: {}s causet_action time: {}s"
                .format(method, step_counter, np.mean(step_time), np.mean(env_step_time),
                        np.mean(train_step_time), np.mean(restart_time), np.mean(action_step_times)))

    step_counter += 1

    if done:
        current_state, _ = env.initialize()
        model.reset(0.01)

logger.info("------------------- Testing Finished -----------------------")
data={'task_id':CONST.TASK_ID}
logger.info("update task %s status" % CONST.TASK_ID )
logger.info(parse_json_post(CONST.URL_UPDATE_TASK,data))

# logger.info("Ricci are saved at: {}".format('test_ricci/'+expr_name + '.pkl'))
logger.info("Proposal ricci At {}".format(max_idx))

