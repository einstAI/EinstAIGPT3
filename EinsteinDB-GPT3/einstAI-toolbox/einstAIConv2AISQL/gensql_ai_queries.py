import os


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
