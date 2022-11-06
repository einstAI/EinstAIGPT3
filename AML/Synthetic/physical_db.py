import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from anyio.streams import file
from matplotlib import rcParams
import sys

from sqlalchemy.dialects.postgresql import psycopg2


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
class DBConnection:
    def __init__(self, db_user="jintao", db_password="jintao", db_host="166.111.121.55", db_port="5432",
                 db="benchmark"):
        # def __init__(self, db_user="jintao", db_password="jintao", db_host="166.111.121.62", db_port="5432", db="imdb"):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db = db

    def vacuum(self):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        old_isolation_level = connection.isolation_level
        connection.set_isolation_level(0)
        query = "VACUUM"
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
        connection.set_isolation_level(old_isolation_level)

    def get_dataframe(self, sql):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        return pd.read_sql(sql, connection)

    def submit_query(self, sql):
        """Submits query and ignores result."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()
        cursor.execute(sql)
        connection.commit()

    def get_result(self, sql):
        """Fetches exactly one row of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()
        cursor.execute('set statement_timeout to 18000')  # 
        cursor.execute(sql)
        record = cursor.fetchone()
        result = record[0]

        if connection:
            cursor.close()
            connection.close()

        return result

    def get_result_set(self, sql, return_columns=False):
        """Fetches all rows of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()

        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        if connection:
            cursor.close()
            connection.close()

        if return_columns:
            return rows, columns

        return rows


class TrueCardinalityEstimator:
    """Queries the database to return true cardinalities."""

    def __init__(self, db_connection):
        self.db_connection = db_connection

    def true_cardinality(self, query):
        cardinality = self.db_connection.get_result(query)
        return cardinality
