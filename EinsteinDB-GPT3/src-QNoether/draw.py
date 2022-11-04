import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sys
import os
def main():
    filelist = ["qnoether.txt", "random.txt"]
    metric_name = "throughput"
    draw_lines(filelist, metric_name)

if __name__ == '__main__':
    main()
def draw_lines(filelist, metric_name):
    '''
    multiple lines on the same metric (y) with increasing iterations (x)
    :param filelist:
    :param metric_name:
    :return: 1 (succeed)/0 (fail)
    '''

    # read data
    data = []
    for file in filelist:
        data.append(pd.read_csv(file))

    # draw
    rcParams['figure.figsize'] = 10, 6

    ''' Load Data: [qnoether] '''
    col_list = ["iteration", metric_name]
    df = pd.read_csv("training-results/" + filelist[0], usecols=col_list, sep="\t")

    x_qnoether = list(df[col_list[0]])
    x_qnoether = [int(x) for x in x_qnoether]
    y_qnoether = list(df[col_list[1]])
    y_qnoether = [float(y) for y in y_qnoether]

    ''' Load Data: [random] '''
    col_list = ["iteration", metric_name]
    df = pd.read_csv("training-results/" + filelist[1], usecols=col_list, sep="\t")


    ''' Load Data: [Random] '''
    col_list = ["iteration", metric_name]
    df = pd.read_csv("training-results/" + filelist[1], usecols=col_list, sep="\t")
    x_random = list(df[col_list[0]])
    x_random = [int(x) for x in x_random]
    y_random = list(df[col_list[1]])
    y_random = [float(y) for y in y_random]

    ''' Draw '''
    plt.plot(x_qnoether, y_qnoether, label="QNoether")
    plt.plot(x_random, y_random, label="Random")
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

    return 1


# Path: EinsteinDB-GPT3/src-QNoether/draw.py
# Compare this snippet from DiabloGPT3/MumfordGrammar/LeanredJoinOrder/JOBDir/CostTraining.py:
#             validate.append(input_list[idx])
#         else:
#             train.append(input_list[idx])
#     return train,validate
#
#
# def QueryLoader(QueryDir):
#     def file_name(file_dir):
#         import os
#         L = []
#         for root, dirs, files in os.walk(file_dir):
#             for file in files:
#                 if os.path.splitext(file)[1] == '.sql':
#                     L.append(os.path.join(root, file))
#         return L
#     QueryList = file_name(QueryDir)
#     return QueryList
#


    ''' figure drawing '''
    mpl.rcdefaults()
    rcParams.update({
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 15,
        #     'figure.autolayout': True,
        'figure.subplot.hspace': 0.45,
        'figure.subplot.wspace': 0.22,
        #     'mathtext.fontset': 'cm',
    })

    fig = plt.figure()

    qid = 1
    ax = fig.add_subplot(1, 1, qid)

    rf = np.array(y_qnoether)
    dt = np.array(y_random)

    x = np.arange(1, max(len(x_random), len(x_qnoether)) + 5)
    # y = np.arange(0.0, 1.0)

    l1, = plt.plot(x_qnoether, rf[:len(x_qnoether)], marker='D', ms=3, linewidth=1)
    l2, = plt.plot(x_random, dt[:len(x_random)], marker='X', ms=3, linewidth=1)

    ax.text(0.5, -0.36,
            f"({chr(ord('a') + qid - 1)}) $D_{{ {qid} }}$",
            horizontalalignment='center', transform=ax.transAxes, fontsize=15, family='serif',
            )
    ax.set_xticks(np.arange(0, len(x), len(x) / 10))
    if metric_name == 'latency':
        y_range = max(max(y_qnoether), max(y_random)) + 5
    elif metric_name == 'throughput':
        y_range = max(max(y_qnoether), max(y_random)) + 100

    ax.set_yticks(np.arange(0, y_range, y_range / 10))
    ax.set_ylim(0, y_range)
    ax.set_xlim(0, len(x))
    ax.set_xlabel('#-Iterations')
    ax.set_ylabel('Performance')

    fig.legend([l1, l2], ['qnoether', 'Random'],
               loc='upper center', ncol=4,
               handlelength=3,
               columnspacing=6.,
               bbox_to_anchor=(0., 0.98, 1., .0),
               bbox_transform=plt.gcf().transFigure,
               fontsize=10,
               )

    plt.savefig('training-results/training.png')

    return 1


if __name__ == '__main__':
    argv = sys.argv
    linelist = argv[1].split(',')
    metric_name = argv[2]
    mark = draw_lines(linelist, metric_name)
    if mark:
        print('Successfully update figure!')
    else:
        print('Fail to update figure!')
