import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

min_max_file = 'min_max.csv'
datasets_dir = 'datasets'


def prepare_pattern_workload(param):
    # Prepare the training data for the pattern workload


pattern2training, pattern2truecard, min_card_log, max_card_log = prepare_pattern_workload('workload')
with open('pattern2training.pkl', 'wb') as f:
    pickle.dump(pattern2training, f)
with open('pattern2truecard.pkl', 'wb') as f:
    pickle.dump(pattern2truecard, f)

with open('pattern2training.pkl', 'rb') as f:
    pattern2training = pickle.load(f)

def prepare_pattern_workload(path):
    pattern2training = {}
    with open('pattern2totalmem.pkl', 'rb') as f:
        pattern2totalnum = pickle.load(f)
    pattern2truecard = {}
    # pattern2totalmem = {}
min_card_log = 100000000
max_card_log = 0
for pattern in tqdm(pattern2totalnum.keys()):
    if pattern not in pattern2training:
        continue
    pattern2training[pattern] = np.array(pattern2training[pattern])
    if pattern2totalnum[pattern] < min_card_log:
        min_card_log = pattern2totalnum[pattern]
    if pattern2totalnum[pattern] > max_card_log:
        max_card_log = pattern2totalnum[pattern]
    pattern2training[pattern] = []
    for i in range(100):
        pattern2training[pattern].append(pattern2totalnum[pattern])
    # pattern2totalmem[pattern] = pattern2totalnum[pattern]
    pattern2truecard[pattern] = np.array(pattern2truecard[pattern])
    pattern2truecard[pattern] = np.log(pattern2truecard[pattern] * pattern2totalnum[pattern] + 1.0)

    return pattern2training, pattern2truecard, min_card_log, max_card_log, pattern2totalmem



def prepare_pattern_workload(path):
    pattern2training = {}
    with open('pattern2totalmem.pkl', 'rb') as f:
        pattern2totalnum = pickle.load(f)
    minmax = pd.read_csv(min_max_file)
    minmax = minmax.set_index('name')
    min_card_log = 999999999999.0
    max_card_log = 0.0
    with open(path + '.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tables = sorted([x.split(' ')[1] for x in line.split('#')[0].split(',')])
            local_cols = []
            vecs = []
            for col_name in minmax.index:
                if col_name.split('.')[0] in tables:
                    local_cols.append(col_name)
                    vecs.append(0.0)
                    vecs.append(1.0)
            conds = [x for x in line.split('#')[2].split(',')]
            for i in range(int(len(conds) / 3)):
                attr = conds[i * 3]
                op = conds[i * 3 + 1]
                value = conds[i * 3 + 2]
                idx = local_cols.index(attr)
                maximum = float(minmax.loc[attr]['max'])
                minimum = float(minmax.loc[attr]['min'])
                distinct_num = minmax.loc[attr]['num_unique_values']
                if op == '=':
                    offset = (maximum - minimum) / distinct_num / 2.0
                    upper = ((float(value) + offset) - minimum) / (maximum - minimum)
                    lower = (float(value) - offset - minimum) / (maximum - minimum)
                elif op == '<':
                    upper = (float(value) - minimum) / (maximum - minimum)
                    lower = 0.0
                elif op == '>':
                    upper = 1.0
                    lower = (float(value) - minimum) / (maximum - minimum)
                else:
                    raise Exception(op)
                if upper < vecs[idx * 2 + 1]:
                    vecs[idx * 2 + 1] = upper
                if lower > vecs[idx * 2]:
                    vecs[idx * 2] = lower
            key = '_'.join(tables)
            card = float(line.split('#')[-1])
            if key not in pattern2truecard:
                full_tables = sorted([x.split(' ')[0] for x in line.split('#')[0].split(',')])
                datas = []
                for t in full_tables:
                    print(t)
                    if t != 'title':
                        datas.append(pd.read_csv('{}/{}.csv'.format(datasets_dir, t), quotechar='"', escapechar='\\',
                                                 error_bad_lines=False, low_memory=False).groupby(
                            ['movie_id']).size().reset_index(name='counts').set_index('movie_id'))
                total = 0
                for id in tqdm(pd.read_csv('{}/title.csv'.format(datasets_dir))['id'].tolist()):
                    num = 1
                    for d in datas:
                        if id in d.index:
                            num *= d.loc[id, 'counts']
                        else:
                            num = 0
                            break
                    total += num
                print('joins {} is {}'.format(key, total))
                pattern2totalnum[key] = total
                with open('pattern2totalnum.pkl', 'wb') as f:
                    pickle.dump(pattern2totalnum, f)
            if key in pattern2training:
                pattern2training[key].append(vecs)
                pattern2truecard[key].append([card / float(pattern2totalnum[key])])
            else:
                pattern2training[key] = [vecs]
                pattern2truecard[key] = [[card / float(pattern2totalnum[key])]]

    return pattern2training, pattern2truecard, min_card_log, max_card_log


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Quick Sel Preprocessing')
    parser.add_argument('--raw-file', type=str, help='sqls to be parsed',
                        default='../../data/tpch/tpch_100M/tpch_100M.sql')
    parser.add_argument('--min-max-file', type=str, help='Min Max',
                        default='../../data/tpch/tpch_100M/min_max.csv')
    parser.add_argument('--datasets-dir', type=str, help='datasets_dir',
                        default='../../data/tpch/tpch_100M')
    parser.add_argument('--output-dir', type=str, help='output_dir',)
    args = parser.parse_args()

    min_max_file = args.min_max_file
    path = args.raw_file
    datasets_dir = args.datasets_dir
    pattern2training, pattern2truecard, _, _ = prepare_pattern_workload(path)
    for k, v in pattern2training.items():
        print(k, len(v))
        data = np.concatenate((v, pattern2truecard[k]), axis=1)
        np.savetxt('./test/java/edu/illinois/quicksel/resources/{}/{}.assertion'.format(args.output_dir, k), data,
                   delimiter=",")
        vecs = []
        for _ in range(int(len(v[0]) / 2)):
            vecs.append(0.0)
            vecs.append(1.0)
        vecs.append(1.0)
        np.savetxt('./test/java/edu/illinois/quicksel/resources/{}/{}.permanent'.format(args.output_dir, k),
                   np.array([vecs]), delimiter=",")
