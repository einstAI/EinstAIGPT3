import argparse


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from anyio.streams import file
from matplotlib import rcParams
import sys
import os

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



parser = argparse.ArgumentParser(description='get_truecard')

parser.add_argument('--version', type=str, help='datasets_dir', default='cols_4_distinct_1000_corr_5_skew_5')

args = parser.parse_args()
version = args.version

# db_connection = DBConnection(db='postgres',db_user='postgres',db_host="/var/run/postgresql")  # modify
db_connection = DBConnection(db='autocard', db_password="jintao2020", db_user='jintao', db_host="localhost")
fschema = open('./csvdata_sql/schema_' + version + '.sql')
schemasql = fschema.read()
dropsql = 'DROP TABLE ' + version + ';'

try:
    db_connection.submit_query(dropsql)  # Clear the table with the current name
except Exception as e:
    pass

try:
    db_connection.submit_query(schemasql)  # establish schema
except Exception as e:
    pass

# csvsql = r'\copy ' + version + ' from ./csvdata_sql/' + version + '.csv with csv header;'
# db_connection.submit_query(csvsql)
# os.system('psql -U postgres')
# os.system(csvsql)
df = pd.read_csv('./csvdata_sql/' + version + '.csv', sep=',', escapechar='\\', encoding='utf-8', low_memory=False,
                 quotechar='"')
columns = tuple(df.columns)
connection = db_connection.connection
cur = connection.cursor()
cur.copy_from(file, version, sep=',') # copy from file object

connection.commit()

true_estimator = TrueCardinalityEstimator(db_connection)
#
f = open('./csvdata_sql/' + version + '.sql')
queries = f.readlines()
i = 0
ftrain = open('./sql_truecard/' + version + 'train.sql', 'w')
ftest = open('./sql_truecard/' + version + 'test.sql', 'w')
for query in tqdm(queries):
    try:
        cardinality_true = true_estimator.true_cardinality(query)
        # print(cardinality_true)
        if cardinality_true == 0:
            continue
        query = query[0:len(query) - 1]
        if i < 10000:
            ftrain.write(query + ',')
            ftrain.write(str(cardinality_true))
            ftrain.write('\n')
        else:
            ftest.write(query + ',')
            ftest.write(str(cardinality_true))
            ftest.write('\n')
        i = i + 1
        if i >= 11000:
            break
    except Exception as e:
        # f2.write('Pass '+query+'\n')
        pass
    continue
ftrain.close()
ftest.close()
