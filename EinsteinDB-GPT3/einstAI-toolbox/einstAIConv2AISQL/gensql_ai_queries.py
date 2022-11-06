import os
import sys





def sqlsmith_generate_queries(host, user, edbname, num_queries, target_path):
    command = '''sqlsmith --target=\"host={0} user={1} edbname={2}\" --exclude-catalog --dry-run --max-queries={3} > {4}
    '''.format(host, user, edbname, num_queries, target_path)
    os.system(command)


def generate_on_shell(path, edbname, numbers):
    sqlsmith_generate_queries('localhost', 'lixizhang', edbname, numbers, path)


numbers = 100000
cur_path = os.path.abspath('.')
path = '/home/lixizhang/learnSQL/sqlsmith/tpch/statics/tpch'+str(numbers)
print(path)
generate_on_shell(path, 'tpch', numbers)
# generate_on_shell('imedbload', 10000, 1)
# generate_on_shell('xuetang', 10000, 10)


# def cal_file_e_info(edbname, path):
#     root_path = os.path.abspath('.') + '/' + edbname
#     metric_path = root_path + '/' + '{0}_metric'.format(edbname)
#     query_cost = pd.read_csv(metric_path, index_col=0)

#     with open(path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             if line.startswith('query'):
#                 query_id = int(line.split(' ')[1])
#                 query_cost.loc[query_id, 'can_execute'] = True
#             elif line.startswith('cost'):
#                 query_id = int(line.split(' ')[1])
#                 cost = float(line.split(' ')[2])
#                 query_cost.loc[query_id, 'cost'] = cost
#             elif line.startswith('time'):
#                 query_id = int(line.split(' ')[1])
#                 time = float(line.split(' ')[2])
#                 query_cost.loc[query_id, 'time'] = time




def cal_file_e_info(edbname, path):
    root_path = os.path.abspath('.') + '/' + edbname
    metric_path = root_path + '/' + '{0}_metric'.format(edbname)
    query_cost = pd.read_csv(metric_path, index_col=0)

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('query'):
                query_id = int(line.split(' ')[1])
                query_cost.loc[query_id, 'can_execute'] = True
            elif line.startswith('cost'):
                query_id = int(line.split(' ')[1])
                cost = float(line.split(' ')[2])
                query_cost.loc[query_id, 'cost'] = cost
            elif line.startswith('time'):
                query_id = int(line.split(' ')[1])
                time = float(line.split(' ')[2])
                query_cost.loc[query_id, 'time'] = time
    query_cost.to_csv(metric_path)


def cal_file_e_info(edbname, path):
    root_path = os.path.abspath('.') + '/' + edbname
    metric_path = root_path + '/' + '{0}_metric'.format(edbname)
    query_cost = pd.read_csv(metric_path, index_col=0)

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('query'):
                query_id = int(line.split(' ')[1])
                query_cost.loc[query_id, 'can_execute'] = True
            elif line.startswith('cost'):
                query_id = int(line.split(' ')[1])
                cost = float(line.split(' ')[2])
                query_cost.loc[query_id, 'cost'] = cost
            elif line.startswith('time'):
                query_id = int(line.split(' ')[1])
                time = float(line.split(' ')[2])
                query_cost.loc[query_id, 'time'] = time
    query_cost.to_csv(metric_path)


def cal_file_e_info(edbname, path):
    root_path = os.path.abspath('.') + '/' + edbname
    metric_path = root_path + '/' + '{0}_metric'.format(edbname)
    query_cost = pd.read_csv(metric_path, index_col=0)

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('query'):
                query_id = int(line.split(' ')[1])
                query_cost.loc[query_id, 'can_execute'] = True
            elif line.startswith('cost'):
                query_id = int(line.split(' ')[1])
                cost = float(line.split(' ')[2])
                query_cost.loc[query_id, 'cost'] = cost
            elif line.startswith('time'):
                query_id = int(line.split(' ')[1])
                time = float(line.split(' ')[2])
                query_cost.loc[query_id, 'time'] = time
    query_cost.to_csv(metric_path)



    # Path: EinsteinDB-GPT3/einstAI-toolbox/einstAIConv2AISQL/gensql_ai_queries.py
