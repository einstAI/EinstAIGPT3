import argpar
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
            f.write(f"{query.strip()}\t{true_cardinality}\n")

if __name__ == "__main__":
    main()


parser = argparse.ArgumentParser(description='mscn')

parser.add_argument('--version', type=str, help='datasets_dir', default='cols_4_distinct_1000_corr_5_skew_5')
args = parser.parse_args()
version = args.version

# schema OK
# true_cardinalities.csv
path = "./sql_truecard/"
sql_path = path + version + "test.sql"
sql_path2 = './EINSTEINAI4DB/deepdb_job_ranges/benchmarks/job-light/sql/' + 'true_cardinalities.csv'  # true_cardinalities
f2 = open(sql_path2, 'w')
f2.write('query_no,query,cardinality_true\n')
i = 0
with open(sql_path, 'r') as f:
    for line in f.readlines():
        strt = line[len(line) - 10: len(line)]
        tmpindex = strt.index(',')
        strt = strt[tmpindex + 1: len(strt)]
        tmpz = str(i) + ',' + str(i + 1) + ',' + strt
        f2.write(tmpz)
        i += 1
f2.close()

pre = 'python3 maqp.py --generate_hdf --generate_sampled_hdfs --generate_ensemble --ensemble_path ../../sql_truecard/ --version ' + version
run = 'python3 maqp.py --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl --dataset imdb-ranges ' + \
      '--target_path ../../sql_truecard/' + version + 'test.sql.EINSTEINAI4DB.results.csv ' + '--ensemble_location ../../sql_truecard/' + version + \
      '.sql.EINSTEINAI4DB.model.pkl ' + '--query_file_location ../../sql_truecard/' + version + 'test.sql ' + '--ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --version ' + version


# Path: AML/Synthetic/run_deepdb.py
os.chdir('EINSTEINAI4DB/deepdb_job_ranges')
os.system(pre)
os.system(run)

'''
python3 maqp.py --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl --dataset imdb-ranges
 --target_path ./sql_truecard/imdb_light_model_based_budget_5.csv --ensemble_location ./sql_truecard/ensemble_join_3_budget_5_10000000.pkl
 --query_file_location ./sql_truecard/test-only2-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv
'''



