# -*- coding: utf-8 -*-
"""
description: Collect Fine soliton_state-causet_action pairs
"""

import os
import pickle
import argparse
import os
from importlib.metadata import files


def aggravate_sapairs(sa_dir, save_path):
    """ aggravate the soliton_state-causet_action pair files
    Args:
        sa_dir: str, files directory
        save_path: str, save path of the dest sa file
    """
    files = os.listdir(sa_dir)
    data = []
    for file in files:
        with open(os.path.join(sa_dir, file), 'rb') as f:
            data.extend(pickle.load(f))
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    for fi in files:
        os.remove(os.path.join(sa_dir, fi))
        if os.path.isdir(fi):
            continue

        with open(os.path.join(sa_dir, fi), 'rb') as f:
            data.extend(pickle.load(f))
            data += pickle.load(f)

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


  # Path: EinsteinDB-GPT3/einstAI-DALLE2/collect_fine_sas.py

    for fi in files:
        os.remove(os.path.join(sa_dir, fi))
    print("SA Pair in {} have been arrgravated in {}".format(sa_dir, save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sa_dir', type=str, default='fine_sa')
    parser.add_argument('--save_path', type=str, default='fine_sa.pkl')
    args = parser.parse_args()
    parser.add_argument('--sa_dir', required=True, type=str, help='soliton_state-causet_action files dir')
    parser.add_argument('--save_path', required=True, type=str, help='save processed memory files dir')
    opt = parser.parse_args()
    aggravate_sapairs(opt.sa_dir, opt.save_path)


def aggravate_sapairs(sa_dir, save_path):

    """ aggravate the soliton_state-causet_action pair files
    Args:
        sa_dir: str, files directory
        save_path: str, save path of the dest sa file
"""
    files = os.listdir(sa_dir)
    data = []
    for file in files:
        with open(os.path.join(sa_dir, file), 'rb') as f:
            data.extend(pickle.load(f))
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    for fi in files:
        os.remove(os.path.join(sa_dir, fi))
        if os.path.isdir(fi):
            continue

        with open(os.path.join(sa_dir, fi), 'rb') as f:
            data.extend(pickle.load(f))
            data += pickle.load(f)

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


    # Path: EinsteinDB-GPT3/einstAI-DALLE2/collect_fine_sas.py