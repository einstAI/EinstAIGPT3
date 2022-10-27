"""Evaluate estimators (Naru or others) on queries."""

import argparse
import collections
import glob
import os
import pickle
import re
import time

import common
import datasets
import estimators as estimators_lib
import made
import numpy as np
import pandas as pd
import torch
import transformer

# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device', DEVICE)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()

parser.add_argument('--testfilepath', type=str, default='dmv-tiny', help='Dataset.')
parser.add_argument('--version', type=str, default='cols_4_distinct_1000_corr_5_skew_5', help='Dataset.')

parser.add_argument('--table', type=str, default='cols_4_distinct_1000_corr_5_skew_5', help='Dataset.')
parser.add_argument('--alias', type=str, default='cdcs', help='Dataset.')

parser.add_argument('--inference-opts',
                    action='store_true',
                    help='Tracing optimization for better latency.')

parser.add_argument('--num-queries', type=int, default=20, help='# queries.')
parser.add_argument('--dataset', type=str, default='dmv-tiny', help='Dataset.')
parser.add_argument('--err-csv',
                    type=str,
                    default='results.csv',
                    help='Save result csv to what path?')
parser.add_argument('--glob',
                    type=str,
                    help='Checkpoints to glob under models/.')
parser.add_argument('--blacklist',
                    type=str,
                    help='Remove some globbed checkpoint files.')
parser.add_argument('--psample',
                    type=int,
                    default=2000,
                    help='# of progressive samples to use per query.')
parser.add_argument(
    '--column-masking',
    action='store_true',
    help='Turn on wildcard skipping.  Requires checkpoints be trained with ' \
         'column masking.')
parser.add_argument('--order',
                    nargs='+',
                    type=int,
                    help='Use a specific order?')

# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=128,
                    help='Hidden units in FC.')
parser.add_argument('--layers', type=int, default=4, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
parser.add_argument(
    '--inv-order',
    action='store_true',
    help='Set this flag iff using MADE and specifying --order. Flag --order' \
         'lists natural indices, e.g., [0 2 1] means variable 2 appears second.' \
         'MADE, however, is implemented to take in an argument the inverse ' \
         'semantics (element i indicates the position of variable i).  Transformer' \
         ' does not have this issue and thus should not have this flag on.')
parser.add_argument(
    '--input-encoding',
    type=str,
    default='binary',
    help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
parser.add_argument(
    '--output-encoding',
    type=str,
    default='one_hot',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
         'then input encoding should be set to embed as well.')

# Transformer.
parser.add_argument(
    '--heads',
    type=int,
    default=0,
    help='Transformer: num heads.  A non-zero value turns on Transformer' \
         ' (otherwise MADE/ResMADE).'
)
parser.add_argument('--blocks',
                    type=int,
                    default=2,
                    help='Transformer: num blocks.')
parser.add_argument('--dmodel',
                    type=int,
                    default=32,
                    help='Transformer: d_model.')
parser.add_argument('--dff', type=int, default=128, help='Transformer: d_ff.')
parser.add_argument('--transformer-act',
                    type=str,
                    default='gelu',
                    help='Transformer activation.')

# Estimators to enable.
parser.add_argument('--run-sampling',
                    action='store_true',
                    help='Run a materialized sampler?')
parser.add_argument('--run-maxdiff',
                    action='store_true',
                    help='Run the MaxDiff histogram?')
parser.add_argument('--run-bn',
                    action='store_true',
                    help='Run Bayes nets? If enabled, run BN only.')

# Bayes nets.
parser.add_argument('--bn-samples',
                    type=int,
                    default=200,
                    help='# samples for each BN inference.')
parser.add_argument('--bn-root',
                    type=int,
                    default=0,
                    help='Root variable index for chow liu tree.')
# Maxdiff
parser.add_argument(
    '--maxdiff-limit',
    type=int,
    default=30000,
    help='Maximum number of partitions of the Maxdiff histogram.')

args = parser.parse_args()
version = args.version
fmetric = open('../metric_result/' + args.version + '.naru.txt', 'a')


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def preprocess_sql(sql_path):
    output = []
    cols = set([])
    with open(sql_path, 'r') as f:
        for line in f.readlines():
            # print (line)
            sql = ','.join(line.split(',')[:-1])
            cardinality = line.split(',')[-1].strip('\n')
            tables = [x.strip() for x in re.search('FROM(.*)WHERE', sql, re.IGNORECASE).group(1).split(',')]
            joins = [x.strip() for x in
                     re.search('WHERE(.*)', sql, re.IGNORECASE).group(1).split('AND')[0:len(tables) - 1]]
            conditions = [x.strip(' ;\n') for x in
                          re.search('WHERE(.*)', sql, re.IGNORECASE).group(1).split('AND')[len(tables) - 1:]]
            conds = []
            for cond in conditions:
                operator = re.search('([<>=])', cond, re.IGNORECASE).group(1)
                left = cond.split(operator)[0]
                right = cond.split(operator)[1]
                cols.add(left)
                conds.append(left)
                conds.append(operator)
                conds.append(right)
            # print (tables, joins, conds, cardinality)
            output.append(','.join(tables) + '#' + ','.join(joins) + '#' + ','.join(conds) + '#' + cardinality)
    with open(sql_path + '.csv', 'w') as f:
        for line in output:
            f.write(line)
            f.write('\n')


def MakeTable():
    assert args.dataset in ['dmv-tiny', 'dmv']
    if args.dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif args.dataset == 'dmv':
        table = datasets.LoadDmv(args.version + '.csv')  # modify

    oracle_est = estimators_lib.Oracle(table)
    if args.run_bn:
        return table, common.TableDataset(table), oracle_est
    return table, None, oracle_est


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          table,
                          return_col_idx=False):
    s = table.data.iloc[rng.randint(0, table.cardinality)]
    vals = s.values

    if args.dataset in ['dmv', 'dmv-tiny']:
        # Giant hack for DMV.
        vals[6] = vals[6].to_datetime64()

    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    if num_filters == len(all_cols):
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals
        return all_cols, ops, vals

    vals = vals[idxs]
    if return_col_idx:
        return idxs, ops, vals

    return cols, ops, vals


def GenerateQuery(all_cols, rng, table, return_col_idx=False):
    """Generate a random query."""
    num_filters = rng.randint(5, 12)
    cols, ops, vals = SampleTupleThenRandom(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            return_col_idx=return_col_idx)
    return cols, ops, vals


def Query(estimators,
          do_print=True,
          oracle_card=None,
          query=None,
          table=None,
          oracle_est=None):
    assert query is not None
    cols, ops, vals = query

    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    # Actual.
    card = oracle_est.Query(cols, ops,
                            vals) if oracle_card is None else oracle_card
    if card == 0:
        return

    pprint('Q(', end='')
    for c, o, v in zip(cols, ops, vals):
        pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
    pprint('): ', end='')

    pprint('\n  actual {} ({:.3f}%) '.format(card,
                                             card / table.cardinality * 100),
           end='')

    for est in estimators:
        est_card = est.Query(cols, ops, vals)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)
        pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
    pprint()


def ReportEsts(estimators):
    v = -1
    for est in estimators:
        print(est.name, 'max', np.max(est.errs), '99th',
              np.quantile(est.errs, 0.99), '95th', np.quantile(est.errs, 0.95),
              'median', np.quantile(est.errs, 0.5))
        v = max(v, np.max(est.errs))
    return v


def binary_search(x, target, start, end):
    if start >= end:
        return start - 1
    mid = int((start + end) / 2.0)
    if x[mid] < target:
        return binary_search(x, target, mid + 1, end)
    elif x[mid] == target:
        return mid
    else:
        return binary_search(x, target, start, mid)


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if float(labels_unnorm[i]) > 0 and float(preds_unnorm[i]) > 0:
            if preds_unnorm[i] > float(labels_unnorm[i]):
                qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
            else:
                qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))
    fmetric.write("Median: {}".format(np.median(qerror)) + '\n' + "90th percentile: {}".format(
        np.percentile(qerror, 90)) + '\n' + "95th percentile: {}".format(np.percentile(qerror, 95)) + \
                  '\n' + "99th percentile: {}".format(np.percentile(qerror, 99)) + '\n' + "99th percentile: {}".format(
        np.percentile(qerror, 99)) + '\n' + "Max: {}".format(np.max(qerror)) + '\n' + \
                  "Mean: {}".format(np.mean(qerror)) + '\n')
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def print_mse(preds_unnorm, labels_unnorm):
    fmetric.write("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean()) + '\n')
    print("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean()))


def print_mape(preds_unnorm, labels_unnorm):
    fmetric.write("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100) + '\n')
    print("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100))


def print_pearson_correlation(x, y):
    from scipy import stats
    PCCs = stats.pearsonr(x, y)
    fmetric.write("Pearson Correlation: {}".format(PCCs) + '\n\n')
    print("Pearson Correlation: {}".format(PCCs))


def RunN(table,
         cols,
         estimators,
         rng=None,
         num=20,
         log_every=50,
         num_filters=11,
         oracle_cards=None,
         oracle_est=None):
    print(11)
    if rng is None:
        rng = np.random.RandomState(1234)
    ests_new, truths_new = [], []
    last_time = None
    print('estimators', estimators)
    alias2table = {args.alias: args.table}  # modify
    with open(args.testfilepath + version + 'test.sql' + '.csv', 'r') as f:  # modify
        queries = f.readlines()[0:999]  # .Parsing
        testt1 = time.time()
        for query in queries:
            print(query)
            tables = [x.split(' ')[0] for x in query.split('#')[0].split(',')]
            alias = [x.split(' ')[1] for x in query.split('#')[0].split(',')]
            key = '_'.join(sorted(alias))
            conditions = query.split('#')[2].split(',')
            true_card = int(query.split('#')[-1])

            columns = np.asarray(table.columns)
            # print('columns',columns)
            names = [c.name for c in columns]
            # col1 = columns[names.index('col0')]

            cols, ops, vals = [], [], []
            for i in range(int(len(conditions) / 3)):
                x = conditions[i * 3].split('.')
                col = columns[names.index(x[1])]  # alias2table[x[0]]+':'+
                op = conditions[i * 3 + 1]
                val = float(conditions[i * 3 + 2])
                x = col.all_distinct_values
                # print (col.all_distinct_values, val)
                position = binary_search(x, val, 0, len(x))
                if op == '<':
                    if position < len(x) - 1:
                        val = x[position + 1]
                    elif position == 0:
                        op = '='
                        val = x[position]
                elif op == '>':
                    if position >= 0:
                        val = x[position]
                # print (f'Updated:{op}, {val}')
                cols.append(col)
                ops.append(op)
                vals.append(val)

            '''        
            cols, ops ,vals = [],[],[]
            
            cols.append(col1)
            col2 = columns[names.index('col1')]
            cols.append(col2)
            ops.append(conditions[1])
            ops.append(conditions[4])

            i = 0
            x = conditions[i*3].split('.')
            col = col1
            op = conditions[i*3+1]
            val = int(conditions[i*3+2])
            x = col.all_distinct_values
            print (col.all_distinct_values, val)
            position = binary_search(x, val, 0, len(x))
            if op == '<':
                if position < len(x) - 1:
                    val = x[position+1]
                elif position == 0:
                    op = '='
                    val = x[position]
            elif op == '>':
                if position >= 0:
                    val = x[position]
            vals.append(val)
              
            i =1        
            x = conditions[i*3].split('.')
            col = col2
            op = conditions[i*3+1]
            val = int(conditions[i*3+2])
            x = col.all_distinct_values
            print (col.all_distinct_values, val)
            position = binary_search(x, val, 0, len(x))
            if op == '<':
                if position < len(x) - 1:
                    val = x[position+1]
                elif position == 0:
                    op = '='
                    val = x[position]
            elif op == '>':
                if position >= 0:
                    val = x[position]
            vals.append(val)
            
            print(ops)
            '''

            query = cols, ops, vals
            estcard = estimators[0].Query(cols, ops,
                                          vals)  # oracle_est
            print('res:', estcard, true_card)
            ests_new.append(estcard)
            truths_new.append(true_card)
        # print(ests_new)
        testt2 = time.time()
        est = np.array(ests_new)
        true = np.array(truths_new)
        a = true.tolist()
        b = est.tolist()
        c = []
        c.append(a)
        c.append(b)
        c = np.array(c)
        c = np.rot90(c)
        c = np.rot90(c)
        c = np.rot90(c)
        np.savetxt(args.testfilepath + version + 'test.sql.naru.result.csv', c, delimiter=',')  # modify /
        fmetric.write('testtime per:' + str((testt2 - testt1) / 1800) + '\n')
        print('testtime per:', (testt2 - testt1) / 1800)
        print_qerror(np.array(ests_new), np.array(truths_new))
        print_mse(np.array(ests_new), np.array(truths_new))
        print_mape(np.array(ests_new), np.array(truths_new))
        print_pearson_correlation(np.array(ests_new), np.array(truths_new))
    return False


def RunNParallel(estimator_factory,
                 parallelism=2,
                 rng=None,
                 num=20,
                 num_filters=11,
                 oracle_cards=None):
    """RunN in parallel with Ray.  Useful for slow estimators e.g., BN."""
    import ray
    ray.init(redis_password='xxx')

    @ray.remote
    class Worker(object):

        def __init__(self, i):
            self.estimators, self.table, self.oracle_est = estimator_factory()
            self.columns = np.asarray(self.table.columns)
            self.i = i

        def run_query(self, query, j):
            col_idxs, ops, vals = pickle.loads(query)
            Query(self.estimators,
                  do_print=True,
                  oracle_card=oracle_cards[j]
                  if oracle_cards is not None else None,
                  query=(self.columns[col_idxs], ops, vals),
                  table=self.table,
                  oracle_est=self.oracle_est)

            print('=== Worker {}, Query {} ==='.format(self.i, j))
            for est in self.estimators:
                est.report()

        def get_stats(self):
            return [e.get_stats() for e in self.estimators]

    print('Building estimators on {} workers'.format(parallelism))
    workers = []
    for i in range(parallelism):
        workers.append(Worker.remote(i))

    print('Building estimators on driver')
    estimators, table, _ = estimator_factory()
    cols = table.columns

    if rng is None:
        rng = np.random.RandomState(1234)
    queries = []
    for i in range(num):
        col_idxs, ops, vals = GenerateQuery(cols,
                                            rng,
                                            table=table,
                                            return_col_idx=True)
        queries.append((col_idxs, ops, vals))

    cnts = 0
    for i in range(num):
        query = queries[i]
        print('Queueing execution of query', i)
        workers[i % parallelism].run_query.remote(pickle.dumps(query), i)

    print('Waiting for queries to finish')
    stats = ray.get([w.get_stats.remote() for w in workers])

    print('Merging and printing final results')
    for stat_set in stats:
        for e, s in zip(estimators, stat_set):
            e.merge_stats(s)
    time.sleep(1)

    print('=== Merged stats ===')
    for est in estimators:
        est.report()
    return estimators


def MakeBnEstimators():
    table, train_data, oracle_est = MakeTable()
    estimators = [
        estimators_lib.BayesianNetwork(train_data,
                                       args.bn_samples,
                                       'chow-liu',
                                       topological_sampling_order=True,
                                       root=args.bn_root,
                                       max_parents=2,
                                       use_pgm=False,
                                       discretize=100,
                                       discretize_method='equal_freq')
    ]

    for est in estimators:
        est.name = str(est)
    return estimators, table, oracle_est


def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    if args.inv_order:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
                     args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model


def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    results = pd.DataFrame()
    for est in estimators:
        data = {
            'est': [est.name] * len(est.errs),
            'err': est.errs,
            'est_card': est.est_cards,
            'true_card': est.true_cards,
            'query_dur_ms': est.query_dur_ms,
        }
        results = results.append(pd.DataFrame(data))
    if return_df:
        return results
    results.to_csv(path, index=False)


def LoadOracleCardinalities():
    ORACLE_CARD_FILES = {
        'dmv': 'datasets/dmv-2000queries-oracle-cards-seed1234.csv'
    }
    path = ORACLE_CARD_FILES.get(args.dataset, None)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) == 2000, len(df)
        return df.values.reshape(-1)
    return None


def Main():
    import os
    import fnmatch
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    all_ckpts = glob.glob('./models/{}'.format(args.glob))
    for f_name in os.listdir('./models'):
        if fnmatch.fnmatch(f_name, version + '*.pt'):  # filename_match
            all_ckpts = glob.glob('./models/' + f_name)
    # all_ckpts = glob.glob('./models/dmv-2.0MB-model5.623-data5.471-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-colmask-20epochs-seed0.pt')
    print(args.blacklist)
    if args.blacklist:
        all_ckpts = [ckpt for ckpt in all_ckpts if args.blacklist not in ckpt]
    print(all_ckpts)
    selected_ckpts = all_ckpts
    oracle_cards = LoadOracleCardinalities()
    print('ckpts', selected_ckpts)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if not args.run_bn:
        # OK to load tables now
        table, train_data, oracle_est = MakeTable()
        cols_to_train = table.columns

    Ckpt = collections.namedtuple(
        'Ckpt', 'epoch model_bits bits_gap path loaded_model seed')
    parsed_ckpts = []

    for s in selected_ckpts:
        if args.order is None:
            z = re.match('.*?model([\d\.]+)-data([\d\.]+).+seed([\d\.]+).*.pt',
                         s)
        else:
            z = re.match(
                '.*?model([\d\.]+)-data([\d\.]+).+seed([\d\.]+)-order.*.pt', s)
        assert z
        model_bits = float(z.group(1))
        data_bits = float(z.group(2))
        seed = int(z.group(3))
        bits_gap = model_bits - data_bits

        order = None
        if args.order is not None:
            order = list(args.order)

        if args.heads > 0:
            model = MakeTransformer(cols_to_train=table.columns,
                                    fixed_ordering=order,
                                    seed=seed)
        else:
            if args.dataset in ['dmv-tiny', 'dmv']:
                model = MakeMade(
                    scale=args.fc_hiddens,
                    cols_to_train=table.columns,
                    seed=seed,
                    fixed_ordering=order,
                )
            else:
                assert False, args.dataset

        assert order is None or len(order) == model.nin, order
        ReportModel(model)
        print('Loading ckpt:', s)
        model.load_state_dict(torch.load(s))
        model.eval()

        print(s, bits_gap, seed)

        parsed_ckpts.append(
            Ckpt(path=s,
                 epoch=None,
                 model_bits=model_bits,
                 bits_gap=bits_gap,
                 loaded_model=model,
                 seed=seed))

    # Estimators to run.
    if args.run_bn:
        estimators = RunNParallel(estimator_factory=MakeBnEstimators,
                                  parallelism=50,
                                  rng=np.random.RandomState(1234),
                                  num=args.num_queries,
                                  num_filters=None,
                                  oracle_cards=oracle_cards)
    else:
        print(parsed_ckpts)
        for c in parsed_ckpts:
            print(c.loaded_model, table, args.psample, args.column_masking)
        estimators = [
            estimators_lib.ProgressiveSampling(c.loaded_model,
                                               table,
                                               args.psample,
                                               device=DEVICE,
                                               shortcircuit=args.column_masking)
            for c in parsed_ckpts
        ]
        # print(estimators)
        for est, ckpt in zip(estimators, parsed_ckpts):
            est.name = str(est) + '_{}_{:.3f}'.format(ckpt.seed, ckpt.bits_gap)

        if args.inference_opts:
            print('Tracing forward_with_encoded_input()...')
            for est in estimators:
                encoded_input = est.model.EncodeInput(
                    torch.zeros(args.psample, est.model.nin, device=DEVICE))

                # NOTE: this line works with torch 1.0.1.post2 (but not 1.2).
                # The 1.2 version changes the API to
                # torch.jit.script(est.model) and requires an annotation --
                # which was found to be slower.
                est.traced_fwd = torch.jit.trace(
                    est.model.forward_with_encoded_input, encoded_input)

        if args.run_sampling:
            SAMPLE_RATIO = {'dmv': [0.0013]}  # ~1.3MB.
            for p in SAMPLE_RATIO.get(args.dataset, [0.01]):
                estimators.append(estimators_lib.Sampling(table, p=p))

        if args.run_maxdiff:
            estimators.append(
                estimators_lib.MaxDiffHistogram(table, args.maxdiff_limit))

        # Other estimators can be appended as well.

        if len(estimators):
            RunN(table,
                 cols_to_train,
                 estimators,
                 rng=np.random.RandomState(1234),
                 num=args.num_queries,
                 log_every=1,
                 num_filters=None,
                 oracle_cards=oracle_cards,
                 oracle_est=oracle_est)
    print(len(estimators))
    SaveEstimators(args.err_csv, estimators)
    print('...Done, result:', args.err_csv)


if __name__ == '__main__':
    preprocess_sql(args.testfilepath + version + 'test.sql')
    Main()
