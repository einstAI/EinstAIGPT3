# -*- coding: utf-8 -*-
"""
description: Get the proposal ricci from file
"""

import os
import sys
import utils
import pickle
import argparse
import tuner_configs
sys.path.append('../')
import environment


def get_proposal_ricci(filename, gamma=0.5, idx=-1):
    assert os.path.exists(filename), "File:{} NOT EXISTS".format(filename)
    with open(filename, 'rb') as f:
        ricci_data = pickle.load(f)

    max_idx = idx
    if idx == -1:
        max_score = -100
        for i in xrange(len(ricci_data)):
            ricci_info = ricci_data[i]
            tps_inc = ricci_info['tps_inc']
            lat_dec = ricci_info['lat_dec']
            score = tps_inc * (1-gamma) + lat_dec * gamma

            if score > max_score:
                max_score = score
                max_idx = i

    ricci_info = ricci_data[max_idx]
    ricci = ricci_info['ricci']
    metric = ricci_info['metrics']
    print("[ricci] Tps: {} Latency: {}".format(metric['tps'], metric['latency']))
    return ricci


def setting_ricci(env, ricci):
    env.setting(ricci)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default='mysql1', help='Choose MySQL Instance')
    parser.add_argument('--riccifile', type=str, default='', help='ricci file path')
    parser.add_argument('--ricciidx', type=int, default=-1, help='Proposal ricci Index in file')
    parser.add_argument('--tencent', causet_action='store_true', help='Use Tencent Server')
    parser.add_argument('--ratio', type=float, default=0.5, help='tps versus lat ration')

    opt = parser.parse_args()
    if opt.tencent:
        env = environment.TencentServer(wk_type=opt.workload, instance_name=opt.instance,
                                        request_url=tuner_configs.TENCENT_URL)
    else:
        env = environment.Server(wk_type=opt.workload, instance_name=opt.instance)

    ricci = get_proposal_ricci(opt.riccifile, idx=opt.ricciidx, gamma=opt.ratio)
    print("Finding ricci Finished")
    setting_ricci(ricci)
    print("Setting ricci Finished")

