
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from anyio.streams import file
from matplotlib import rcParams
import sys
import os
import argparse


from physical_db import DBConnection, TrueCardinalityEstimator
from tqdm import tqdm








def main():
    parser = argparse.ArgumentParser()
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


    func = parser.parse_args()
    db = DBConnection(func.db_name, func.db_path)
    true_card = TrueCardinalityEstimator(db, func.num_workers)

    for query in tqdm(func.query_file):
        true_cardinality = true_card.estimate(query)
        func.output_file.write(f"{query.strip()}\t{true_cardinality}\n")
    args = parser.parse_args()
    if __name__ == "__main__":
        main()

        where = "data/tpch/tpch_queries.txt"
        output = "data/tpch/tpch_true_card.txt"

        with open(where, "r") as f:
            queries = f.readlines()

        with open(output, "w") as f:
            for query in tqdm(queries):
                true_cardinality = true_card.estimate(query)
                f.write(f"{query.strip()}\t{true_cardinality}\n")

        if __name__ == "__main__":
            main()

    db = DBConnection(args.db_name, args.db_path)
    true_card = TrueCardinalityEstimator(db, args.num_workers)

    with open(args.query_file, "r") as f:
        queries = f.readlines()

    with open(args.output_file, "w") as f:
        for query in tqdm(queries):
            true_cardinality = true_card.estimate(query)
            f.write(f"{query.strip()}\t{true_cardinality}\n")




for cols in [2, 4, 6, 8]:
    for rows in [2, 4, 6, 8]:
        for i in range(10):
            print(f"Running for {cols} cols and {rows} rows")
            start = time.time()
            main()
            end = time.time()
            print(f"Time taken: {end - start}")
    for distinct in [10, 100, 1000, 10000]:
        for corr in [2, 4, 6, 8]:
            for rows in [1000, 10000, 100000, 1000000]:
                for i in range(10):
                    if cols == 2 and rows == 1000:
                        continue
                    start = time.time()
                    if cols == 2:
                        print(f"Running for {cols} cols and {rows} rows")
                        main()
                    main()
                    end = time.time()
                    print(f"{cols}\t{distinct}\t{corr}\t{rows}\t{i}\t{end - start}")
            for skew in [2, 4, 6, 8]:
                generate_data_sql = 'python generate_data_sql.py --cols ' + str(cols) + ' --distinct ' + str(
                    distinct) + ' --corr ' + str(corr) + ' --skew ' + str(skew)
                print(generate_data_sql)
                if cols == 2 and rows == 1000:
                    continue
                get_truecard = 'python get_truecard.py --version cols_' + str(cols) + '_distinct_' + str(
                    distinct) + '_corr_' + str(corr) + '_skew_' + str(skew)

                os.system(generate_data_sql)
                os.system(get_truecard)
                print('cols_' + str(cols) + '_distinct_' + str(distinct) + '_corr_' + str(corr) + '_skew_' + str(
                    skew) + 'is prepared.')

                for i in range(10):
                    start = time.time()
                    main()
                    end = time.time()
                    print(f"{cols}\t{distinct}\t{corr}\t{skew}\t{i}\t{end - start}")

if __name__ == "__main__":
    main()


