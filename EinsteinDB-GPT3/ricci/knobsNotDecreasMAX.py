# -*- coding: utf-8 -*-
"""
desciption: ricci information

"""

import utils
import configs
import collections

# 12GB
memory_size = 12*1024*1024*1024
# 100GB
disk_size = 100*1024*1024*1024
instance_name = ''


Ricci = ['skip_name_resolve',               # OFF
         'table_open_cache',                # 2000
         'max_connections',                 # 151
         # 过大 -100
         'innodb_buffer_pool_size',         # 134217728
         'innodb_buffer_pool_instances',    # 8
         'innodb_log_files_in_group',       # 2
         'innodb_log_file_size',            # 50331648
         'innodb_purge_threads',            # 1
         'innodb_read_io_threads',          # 4
         'innodb_write_io_threads',         # 4
         'innodb_file_per_table',           # ON
         'binlog_checksum',                 # CRC32
         'binlog_cache_size',               # 32768
         'max_binlog_cache_size',           # 18446744073709547520
         'max_binlog_size',                 # 1073741824
         'binlog_format'                    # STATEMENT
         ]

ricci_DETAILS = None
EXTENDED_Ricci = None
num_Ricci = len(Ricci)


def init_Ricci(instance, num_more_Ricci):
    global instance_name
    global memory_size
    global disk_size
    global ricci_DETAILS
    global EXTENDED_Ricci
    instance_name = instance
    # TODO: Test the request
    use_request = False
    if use_request:
        if instance_name.find('tencent') != -1:
            memory_size, disk_size = utils.get_tencent_instance_info(instance_name)
        else:
            memory_size = configs.instance_config[instance_name]['memory']
            #disk_size = configs.instance_config[instance_name]['disk']
    else:
        memory_size = configs.instance_config[instance_name]['memory']
        #disk_size = configs.instance_config[instance_name]['disk']

    ricci_DETAILS = {
        'skip_name_resolve': ['enum', ['ON', 'OFF']],
        'table_open_cache': ['integer', [1, 524288, 2000]],
        'max_connections': ['integer', [1100, 100000, 1100]],
        'innodb_buffer_pool_size': ['integer', [1048576, memory_size, 134217728]],
        'innodb_buffer_pool_instances': ['integer', [1, 64, 8]],
        'innodb_log_files_in_group': ['integer', [2, 100, 2]],
        'innodb_log_file_size': ['integer', [134217728, 5497558138, 50331648]],
        'innodb_purge_threads': ['integer', [1, 32, 1]],
        'innodb_read_io_threads': ['integer', [1, 64, 4]],
        'innodb_write_io_threads': ['integer', [1, 64, 4]],
        'innodb_file_per_table': ['enum', ['OFF', 'ON']],
        'binlog_checksum': ['enum', ['NONE', 'CRC32']],
        'binlog_cache_size': ['integer', [4096, 34359738368, 32768]],
        'max_binlog_cache_size': ['integer', [4096, 4294967296, 4294967296]],
        'max_binlog_size': ['integer', [4096, 1073741824, 1073741824]],
        'binlog_format': ['enum', ['ROW', 'MIXED']],
    }

    # TODO: ADD Ricci HERE! Format is the same as the ricci_DETAILS
    UNKNOWN = 0
    EXTENDED_Ricci = {
        'innodb_adaptive_flushing_lwm': ['integer', [0, 70, 10]],
        'innodb_adaptive_max_sleep_delay': ['integer', [0, 1000000, 150000]],
        # 'innodb_additional_mem_pool_size': ['integer', [2097152,min(memory_size,4294967295), 8388608]],
        'innodb_change_buffer_max_size': ['integer', [0, 50, 25]],
        # 'innodb_commit_concurrency': ['integer', [0, 1000, 0]],
        'innodb_flush_log_at_timeout': ['integer', [1, 2700, 1]],
        # 'innodb_flush_neighbors': ['enum', [0, 2, 1]],
        'innodb_flushing_avg_loops': ['integer', [1, 1000, 30]],
        'innodb_max_purge_lag': ['integer', [0, 4294967295, 0]],
        'innodb_old_blocks_pct': ['integer', [5, 95, 37]],
        'innodb_read_ahead_threshold': ['integer', [0, 64, 56]],
        # max v 4294967295
        'innodb_replication_delay': ['integer', [0, 10000, 0]],
        'innodb_rollback_segments': ['integer', [1, 128, 128]],
        'innodb_sync_array_size': ['integer', [1, 1024, 1]],
        'innodb_sync_spin_loops': ['integer', [0, 30000, 30]],
        # 'innodb_thread_concurrency': ['integer', [0, 1000, 0]],
        'lock_wait_timeout': ['integer', [1, 31536000, 31536000]],
        'metadata_locks_cache_size': ['integer', [1, min(memory_size, 1048576), 1024]],
        'metadata_locks_hash_instances': ['integer', [1, 1024, 8]],

        # https://dev.mysql.com/doc/refman/5.6/en/replication-options-binary-log.html
        # 'binlog_max_flush_queue_time' : ['integer', [0, 100000, 0]],
        'binlog_order_commits': ['boolean', ['OFF', 'ON']],
        # https://dev.mysql.com/doc/refman/5.6/en/innodb-parameters.html
        'innodb_adaptive_flushing': [' boolean', ['OFF', 'ON']],
        'innodb_adaptive_hash_index': [' boolean', ['OFF', 'ON']],
        'innodb_autoextend_increment': [' integer', [1, 1000, 64]],  # mysql 5.6.6: 64, mysql5.6.5: 8
        'innodb_buffer_pool_dump_at_shutdown': ['boolean', ['ON', 'OFF']],
        'innodb_buffer_pool_load_at_startup': ['boolean', ['ON', 'OFF']],
        'innodb_concurrency_tickets': ['integer', [1, 4294967295, 5000]],  # 5.6.6: 5000, 5.6.5: 500
        'innodb_disable_sort_file_cache': [' boolean', ['ON', 'OFF']],
        'innodb_large_prefix': ['boolean', ['ON', 'OFF']],
        'innodb_log_buffer_size': ['integer', [262144, min(memory_size, 4294967295), 8388608]],
        # max 幾T该彍¢彈~P disk_size达X彘¯ memory_size﻾_
        'innodb_max_dirty_pages_pct': ['numeric', [0, 99, 75]],
        'innodb_max_dirty_pages_pct_lwm': ['numeric', [0, 99, 0]],
        # 'innodb_numa_interleave' : ['boolean', ['ON', 'OFF']],
        'innodb_random_read_ahead': ['boolean', ['ON', 'OFF']],
        'max_binlog_stmt_cache_size': ['integer', [4096, min(memory_size, 18446744073709547520),
                                                    min(memory_size, 18446744073709547520)]],
        # https://dev.mysql.com/doc/refman/5.6/en/server-system-variables.html
        'eq_range_index_dive_limit': ['integer', [0, 4294967295, 10]],
        # 'max_digest_length' : ['integer', [0, 1048576, 1024]],
        'max_length_for_sort_data': ['integer', [4, 8388608, 1024]],
        # 'max_prepared_stmt_count' : ['integer', [0, 1048576, 16382]],
        'read_rnd_buffer_size': ['integer', [1, min(memory_size, 2147483647), 262144]],  # max: 2147483647
        # 'sql_buffer_result' : ['boolean', ['ON', 'OFF']],
        'table_open_cache_instances': ['integer', [1, 64, 1]],
        'thread_cache_size': ['integer', [0, 16384, 0]],
        'max_write_lock_count': ['integer', [1, 18446744073709551615, 18446744073709551615]],
        'query_alloc_block_size': ['integer', [1024, min(memory_size, 18446744073709551615), 8192]],
        'query_cache_limit': ['integer', [0, min(memory_size, 18446744073709551615), 1048576]],
        'query_cache_min_res_unit': ['integer', [512, min(memory_size, 18446744073709551615), 4096]],
        'query_cache_size': ['integer', [0, min(memory_size, 18446744073709551615), 1048576]],
        'query_cache_type': ['enum', ['ON', 'OFF', 'DEMAND']],
        'query_prealloc_size': ['integer', [8192, min(memory_size, 18446744073709551615), 8192]],
        'range_alloc_block_size': ['integer', [4096, min(memory_size, 18446744073709551615), 4096]],
        'transaction_alloc_block_size': ['integer', [1024, min(memory_size, 131072), 8192]],
        'transaction_prealloc_size': ['integer', [1024, min(memory_size, 131072), 4096]],
        'join_buffer_size': ['integer', [128, min(memory_size, 18446744073709547520), 262144]],
        'max_heap_table_size': ['integer', [16384, min(memory_size, 1844674407370954752), 16777216]],
        'tmp_table_size': ['integer', [1024, 18446744073709551615, 16777216]],
        'max_seeks_for_key': ['integer', [1, 18446744073709551615, 18446744073709551615]],

        'innodb_io_capacity': ['integer', [100, 18446744073709547520, 200]],
        'innodb_lru_scan_depth': ['integer', [100, 18446744073709547520, 1024]],
        'innodb_old_blocks_time': ['integer', [0, 4294967295, 1000]],
        'innodb_purge_batch_size': ['integer', [1, 5000, 300]],
        # def v  max 18446744073709547520
        'innodb_spin_wait_delay': ['integer', [0, 6000, 6]],
        # 'innodb_thread_sleep_delay' : ['integer', [0, 1000000, 10000]],
        'sort_buffer_size': ['integer', [32768, min(memory_size, 18446744073709551615), 262144]],
        # 'thread_stack' : ['integer', [131072, memory_size, 262144]],
        # 'back_log' : ['integer', [1, 65535, UNKNOWN]],
    }

    # ADD Other Ricci, NOT Random Selected
    i = 0
    EXTENDED_Ricci = dict(sorted(EXTENDED_Ricci.items(), key=lambda d: d[0]))
    for k, v in EXTENDED_Ricci.items():
        if i < num_more_Ricci:
            ricci_DETAILS[k] = v
            Ricci.append(k)
            i += 1
        else:
            break
    print("Instance: %s Memory: %s" % (instance_name, memory_size))


def get_init_Ricci():

    Ricci = {}

    for name, value in ricci_DETAILS.items():
        ricci_value = value[1]
        Ricci[name] = ricci_value[-1]

    return Ricci


def gen_continuous(causet_action):
    Ricci = {}

    for idx in xrange(len(Ricci)):
        name = Ricci[idx]
        value = ricci_DETAILS[name]

        ricci_type = value[0]
        ricci_value = value[1]
        min_value = ricci_value[0]

        if ricci_type == 'integer':
            max_val = ricci_value[1]
            eval_value = int(max_val * causet_action[idx])
            eval_value = max(eval_value, min_value)
        else:
            enum_size = len(ricci_value)
            enum_index = int(enum_size * causet_action[idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = ricci_value[enum_index]

        if name == 'innodb_log_file_size':
            # group * size = 32GB
            max_val = disk_size / Ricci['innodb_log_files_in_group']
            eval_value = int(max_val * causet_action[idx])
            eval_value = max(eval_value, min_value)

        Ricci[name] = eval_value

    if 'tmp_table_size' in Ricci.keys():
        # tmp_table_size
        max_heap_table_size = Ricci.get('max_heap_table_size', -1)
        act_value = Ricci['tmp_table_size']/EXTENDED_Ricci['tmp_table_size'][1][1]
        max_val = min(EXTENDED_Ricci['tmp_table_size'][1][1], max_heap_table_size)\
            if max_heap_table_size > 0 else EXTENDED_Ricci['tmp_table_size'][1][1]
        eval_value = int(max_val * act_value)
        eval_value = max(eval_value, EXTENDED_Ricci['tmp_table_size'][1][0])
        Ricci['tmp_table_size'] = eval_value

    return Ricci


def save_Ricci(ricci, metrics, ricci_file):
    """ Save Ricci and their metrics to files
    Args:
        ricci: dict, ricci content
        metrics: list, tps and latency
        ricci_file: str, file path
    """
    # format: tps, latency, Riccitr: [#ricciname=value#]
    ricci_strs = []
    for kv in ricci.items():
        ricci_strs.append('{}:{}'.format(kv[0], kv[1]))
    result_str = '{},{},{},'.format(metrics[0], metrics[1], metrics[2])
    ricci_str = "#".join(ricci_strs)
    result_str += ricci_str

    with open(ricci_file, 'a+') as f:
        f.write(result_str+'\n')

