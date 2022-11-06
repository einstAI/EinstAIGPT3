import argparse
import os

from tensorboard.data.provider import Run

parser = argparse.ArgumentParser(description='mscn')

parser.add_argument('--version', type=str, help='datasets_dir', default='cols_4_distinct_1000_corr_5_skew_5')
parser.add_argument('--model', type=str, help='nn||xgb', default='nn')
args = parser.parse_args()
version = args.version
model = args.model

run = Run.get_context()
run.log('version', version)
run.log('model', model)


os.chdir('./xgboost_&_localnn')
os.system(run)
os.chdir('../')


# # schema OK
# # true_cardinalities.csv
# path = "./sql_truecard/"
# sql_path = path + version + "test.sql"


