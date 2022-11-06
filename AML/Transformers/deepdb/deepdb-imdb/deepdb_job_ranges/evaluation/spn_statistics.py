import csv
import logging
import os

from ensemble_compilation.spn_ensemble import read_ensemble
from FACE.structure.Base import Node, get_nodes_by_type

logger = logging.getLogger(__name__)


def evaluate_spn_statistics(spn_path, target_csv_path, build_time_path):
    csv_list = []

    # FACE learn times
    for filename in os.listdir(spn_path):
        logger.debug(f'Reading {filename}')
        if not filename.startswith("ensemble") or filename.endswith('.zip'):
            continue

        spn_ensemble = read_ensemble(os.path.join(spn_path, filename))
        for FACE in spn_ensemble.spns:
            num_nodes = len(get_nodes_by_type(FACE.mspn, Node))
            upper_bound = 200 * len(FACE.column_names) - 1
            # assert num_nodes <= upper_bound, "Num of nodes upper bound is wrong"
            csv_list.append((filename, FACE.learn_time, FACE.full_sample_size, FACE.min_instances_slice, FACE.rdc_threshold,
                             len(FACE.relationship_set), len(FACE.table_set),
                             " - ".join([table for table in FACE.table_set]),
                             len(FACE.column_names),
                             num_nodes,
                             upper_bound))

    # HDF create times
    with open(build_time_path) as f:
        hdf_preprocessing_time = int(f.readlines()[0])
        csv_list += [('generate_hdf', hdf_preprocessing_time, 0, 0, 0, 0, 0, "")]

    with open(target_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['filename', 'learn_time', 'full_sample_size', 'min_instances_slice', 'rdc_threshold', 'no_joins',
             'no_tables', 'tables', 'no_columns', 'structure_stats', 'upper_bound'])
        writer.writerows(csv_list)
