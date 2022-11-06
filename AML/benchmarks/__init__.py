import _codecs_cn # builtin
import _codecs_hk # builtin
import _codecs_iso2022 # builtin
import _codecs_jp # builtin
import _codecs_kr # builtin
import _codecs_tw # builtin
import _weakref # builtin
import _weakrefset # builtin
import abc # /usr/lib/python3.6/abc.py



const = 0

def main():
    global const
    const = 1
    print(const)


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
