import argparse
import os
import sys
import time
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from anyio.streams import file
from matplotlib import rcParams
from tqdm import tqdm

from physical_db import DBConnection, TrueCardinalityEstimator

def main():
    for i in range(100):
        print(i)

    parser = ArgumentParser()
    parser.add_argument(
        "--db_name",
        type=str,
        default="tpch",
        help="Database name",
    )

    parser.add_argument(
        "--db_path",
        type=str,
        default="data/tpch",
        help="Path to the database",
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default="data/tpch/tpch_queries.txt",
        help="Path to the query file",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/tpch/tpch_true_card.txt",
        help="Path to the output file",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers",
    )





    args = parser.parse_args()


    db = DBConnection(args.db_name, args.db_path)


    true_card = TrueCardinalityEstimator(db, args.num_workers)


    with open(args.query_file, "r") as f:
        queries = f.readlines()


    with open(args.output_file, "w") as f:

        for query in tqdm(queries):
            true_cardinality = true_card.estimate(query)
            f.write(str(true_cardinality) + " " + query)



parser = argparse.ArgumentParser(description='mscn_xgb_nn')
parser.add_argument('--train_num', type=int, default=1000000, help='train_num')
parser.add_argument('--test_num', type=int, default=1000000, help='test_num')
parser.add_argument('--batch_size', type=int, default=100000, help='batch_size')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
