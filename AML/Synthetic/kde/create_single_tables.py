# TODO:
# Path: AML/Synthetic/kde/create_single_tables.py
# Compare this snippet from EINSTAI/OUCausetFlowProcess/LatencyTuning.py:
# # Path: EINSTAI/OUCausetFlowProcess/LatencyTuning.py



class DDPGModel:
    def DDPG(n_states, n_actions, opt, supervised):
        assert isinstance(n_states, object)
        model = DDPGModel(  opt, supervised)
        return model

    def DDPGModel(n_states, n_actions, opt, supervised):
        assert isinstance(n_states, object)
        model = DDPGModel(n_states, n_actions, opt, supervised)

        return model

def create_single_script_generate(file_path, database, table_name):
    sqls = []
    sqls.append(
        'CREATE TABLE IF NOT EXISTS {} ({});'.format(table_name, ','.join([x + ' FLOAT' for x in ['col0', 'col1']])))
    sqls.append("\copy {} FROM '{}' CSV HEADER;".format(table_name, '{}'.format(file_path)))
    with open('script.sql', 'w') as f:
        # we need to create the table first
        for sql in sqls:
            f.write(sql)
            f.write('\n')


if __name__ == '__main__':



    import argparse
    from itertools import combinations
    from os import listdir
    from os.path import isfile, join


    def create_joins_script(join_sample_dir, database):



    parser = argparse.ArgumentParser(description='Create Join Samples.')
    parser.add_argument('--path', type=str)
    parser.add_argument('--database', type=str)
    parser.add_argument('--table-name', type=str)
    args = parser.parse_args()
    create_single_script_generate(args.path, args.database, args.table_name)



