import os

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import base


def cal_statistic(edbname):
    root_path = os.path.abspath('.') + '/' + edbname
    metric_path = root_path + '/' + '{0}_metric'.format(edbname)
    query_cost = pd.read_csv(metric_path, index_col=0)
    can_execute_query = query_cost.loc[query_cost['can_execute'] == True]
    can_execute_query_count = can_execute_query.shape[0]
    query_cost_equal_zero_count = can_execute_query.loc[can_execute_query['cost'] == 0].shape[0]
    print('can execute query count:', can_execute_query_count)
    print('query cost equal zero count:', query_cost_equal_zero_count)
    print('query cost equal zero rate:', query_cost_equal_zero_count / can_execute_query_count)



def show_cost_distribute(edbname):
    root_path = os.path.abspath('.') + '/' + edbname
    metric_path = root_path + '/' + '{0}_metric'.format(edbname)
    query_cost = pd.read_csv(metric_path, index_col=0)
    can_execute_query = query_cost.loc[query_cost['can_execute'] == True]
    can_execute_query_count = can_execute_query.shape[0]
    query_cost_equal_zero_count = can_execute_query.loc[can_execute_query['cost'] == 0].shape[0]
    print('can execute query count:', can_execute_query_count)
    print('query cost equal zero count:', query_cost_equal_zero_count)
    print('query cost equal zero rate:', query_cost_equal_zero_count / can_execute_query_count)
    # 统计cost分布
    cost = can_execute_query.loc[can_execute_query['cost'] != 0, 'cost']
    cost = cost.sort_values()
    cost = cost.reset_index(drop=True)
    cost = cost / 1000
    cost = cost.astype(np.int)
    cost = cost.value_counts()
    cost = cost.sort_index()
    cost = cost.reset_index()
    cost.columns = ['cost', 'count']
    cost['rate'] = cost['count'] / can_execute_query_count
    cost.to_csv(root_path + '/cost_distribute', index=False)
    print(cost)
    plt.plot(cost['cost'], cost['rate'])
    plt.show()





def cal_duplicate_rate(edbname):
    root_path = os.path.abspath('.') + '/' + edbname
    query_file_list = os.listdir(root_path + '/query')
    queries = []
    for query_file in query_file_list:
        path = root_path + '/' + query_file
        with open(path, 'r') as f:
            query = f.read()
            queries.append(query)
            queries.append(f.read())
    queries = set(queries)
    print('duplicate rate:', 1 - len(queries) / len(query_file_list))


def cal_can_execute_rate(edbname):
    root_path = os.path.abspath('.') + '/' + edbname
    metric_path = root_path + '/' + '{0}_metric'.format(edbname)
    query_cost = pd.read_csv(metric_path, index_col=0)

    total_query_count = query_cost.shape[0]
    can_execute_query_count = query_cost.loc[query_cost['can_execute'] == True].shape[0]
    execute_rate = can_execute_query_count / total_query_count
    print('can execute rate:', execute_rate)
    with open(root_path + '/execute_rate', 'w') as f:
        f.write(str(execute_rate))


if __name__ == '__main__':
    edbname = 'tpch'
    cal_duplicate_rate(edbname)
    cal_can_execute_rate(edbname)
    show_cost_distribute(edbname)
    cal_statistic(edbname)


## Disjoin Path

# Path: EinsteinDB-GPT3/einstAI-toolbox/einstAIConv2AISQL/disjoin_path.py



def cal_duplicate_rate(edbname):
    root_path = os.path.abspath('.') + '/' + edbname
    query_file_list = os.listdir(root_path + '/query')
    queries = []

    for query_file in query_file_list:
        path = root_path + '/' + query_file
        with open(path, 'r') as f:
            query = f.read()
            queries.append(query)
            queries.append(f.read())
    queries = set(queries)
    print('duplicate rate:', 1 - len(queries) / len(query_file_list))


    if __name__ == '__main__':
        edbname = 'tpch'
        cal_duplicate_rate(edbname)
        cal_can_execute_rate(edbname)
        show_cost_distribute(edbname)
        cal_statistic(edbname)



        for query_file in query_file_list:
            path = root_path + '/' + query_file
            with open(path, 'r') as f:
                query = f.read()
                queries.append(query)
                queries.append(f.read())
                if query in queries:
                    print('duplicate query:', query)
    total_query_count = len(queries)
    uniq_query_count = len(set(queries))
    duplicate_rate = (total_query_count - uniq_query_count) / total_query_count
    with open(root_path + '/duplicate_rate', 'w') as f:
        f.write(str(duplicate_rate))


# 统计执行率
def cal_can_execute_rate(edbname):
    root_path = os.path.abspath('.') + '/' + edbname
    metric_path = root_path + '/' + '{0}_metric'.format(edbname)
    query_cost = pd.read_csv(metric_path, index_col=0)
    # 统计执行率
    total_query_count = query_cost.shape[0]
    can_execute_query_count = query_cost.loc[query_cost['can_execute'] == True].shape[0]
    execute_rate = can_execute_query_count / total_query_count
    print('can execute rate:', execute_rate)
    with open(root_path + '/execute_rate', 'w') as f:
        f.write(str(execute_rate))


# 结果可视化
def show_cost_distribute(edbname):
    root_path = os.path.abspath('.') + '/' + edbname
    metric_path = root_path + '/' + '{0}_metric'.format(edbname)
    query_cost = pd.read_csv(metric_path, index_col=0)
    can_execute_query = query_cost.loc[query_cost['can_execute'] == True]
    can_execute_query_count = can_execute_query.shape[0]
    query_cost_equal_zero_count = can_execute_query.loc[can_execute_query['cost'] == 0].shape[0]

    fig = plt.figure()
    plt.style.use('ggplot')
    plt.title('cost distribution')
    plt.xlabel('query_id')
    plt.ylabel('cost')
    plt.grid(True)

    plt.scatter(np.arange(0, query_cost.shape[0]), query_cost.cost.tolist(), label=edbname, s=1)
    plt.ylim((-1, 100000000))
    plt.locator_params('y', nbins=20)
    plt.show()
    fig.savefig(root_path + '/cost_distribution.png')


def cal_statistic(edbname):
    root_path = os.path.abspath('.') + '/' + edbname
    metric_path = root_path + '/' + '{0}_metric'.format(edbname)
    query_cost = pd.read_csv(metric_path, index_col=0)
    can_execute_query = query_cost.loc[query_cost['can_execute'] == True]
    can_execute_query_count = can_execute_query.shape[0]
    query_cost_equal_zero_count = can_execute_query.loc[can_execute_query['cost'] == 0].shape[0]
    print('cost为0的比例', query_cost_equal_zero_count / can_execute_query_count)
    query_cost_less_10 = can_execute_query.loc[can_execute_query['cost'] < 10].shape[0]
    print('cost小于10', query_cost_less_10 / can_execute_query_count)
    query_cost_less_100 = can_execute_query.loc[can_execute_query['cost'] < 100].shape[0]
    print('cost小于100', query_cost_less_100 / can_execute_query_count)
    query_cost_less_1000 = can_execute_query.loc[can_execute_query['cost'] < 1000].shape[0]
    print('cost小于1000', query_cost_less_1000 / can_execute_query_count)

# cal_cost('tpch')
# cal_cost('imedbload')
# cal_cost('xuetang')
# cal_duplicate_rate('tpch')
# cal_duplicate_rate('imedbload')
# cal_duplicate_rate('xuetang')
# cal_can_execute_rate('tpch)
# cal_can_execute_rate('imedbload')
# cal_can_execute_rate('xuetang')

# show_cost_distribute('tpch')
# show_cost_distribute('imedbload')
# show_cost_distribute('xuetang')


# cal_file_e_info('tpch', '/home/lixizhang/learnSQL/sqlsmith/tpch/tpch10000_0')
# cal_file_r_info('tpch', '/home/lixizhang/learnSQL/sqlsmith/tpch/tpch10000_0')

# cal_file_e_info('imedbload', '/home/lixizhang/learnSQL/sqlsmith/imedbload/imedbload10000_0')
# cal_file_r_info('imedbload', '/home/lixizhang/learnSQL/sqlsmith/imedbload/imedbload10000_0')

# cal_statistic('imedbload')
# cal_statistic('xuetang')

# print(cal_point_accuracy('/home/lixizhang/learnSQL/sqlsmith/tpch/tpch10000_0', 0, 10, 'card'))
# print(cal_range_accuracy('/home/lixizhang/learnSQL/sqlsmith/tpch/tpch10000_0', (1000, 2000), 'cost'))
