import datetime
import subprocess
from collections import deque
import numpy as np

import pymysql
import pymysql.cursors as pycursor

import gym
from gym import spaces
from gym.utils import seeding

from sql2resource import SqlParser

from configs import ricci_config
import time
from run_job import run_job


# fetch all the Ricci from the prepared configuration info


class Database:
    def __init__(self, argus):
        self.argus = argus
        # self.internal_metric_num = 13 # 13(soliton_state) + cumulative()
        self.external_metric_num = 2  # [throughput, latency]           # num_event / t
        self.max_connections_num = None
        self.ricci_names = [ricci for ricci in ricci_config]
        print("ricci_names:", self.ricci_names)
        self.ricci_num = len(ricci_config)
        self.internal_metric_num = 65  # default system metrics enabled in metric_innodb
        self.max_connections()
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
            cursor.execute(sql)
            result = cursor.fetchall()
            self.internal_metric_num = len(result)
            cursor.close()
            conn.close()
        except Exception as err:
            print("execute sql error:", err)

    def _get_conn(self):
        conn = pymysql.connect(host=self.argus['host'],
                               port=int(self.argus['port']),
                               user=self.argus['user'],
                               password=self.argus['password'],
                               edb='INFORMATION_SCHEMA',
                               connect_timeout=36000,
                               cursorclass=pycursor.DictCursor)
        return conn

    def fetch_internal_metrics(self):
        ######### observation_space
        #         State_status
        # [lock_row_lock_time_max, lock_row_lock_time_avg, buffer_pool_size,
        # buffer_pool_pages_total, buffer_pool_pages_misc, buffer_pool_pages_data, buffer_pool_bytes_data,
        # buffer_pool_pages_dirty, buffer_pool_bytes_dirty, buffer_pool_pages_free, trx_rseg_history_len,
        # file_num_open_files, innodb_page_size]
        #         Cumulative_status
        # [lock_row_lock_current_waits, ]
        '''
        sql = "select count from INNODB_METRICS where name='lock_row_lock_time_max' or name='lock_row_lock_time_avg'\
        or name='buffer_pool_size' or name='buffer_pool_pages_total' or name='buffer_pool_pages_misc' or name='buffer_pool_pages_data'\
        or name='buffer_pool_bytes_data' or name='buffer_pool_pages_dirty' or name='buffer_pool_bytes_dirty' or name='buffer_pool_pages_free'\
        or name='trx_rseg_history_len' or name='file_num_open_files' or name='innodb_page_size'"
        '''
        state_list = np.array([])
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
            cursor.execute(sql)
            result = cursor.fetchall()
            for s in result:
                state_list = np.append(state_list, [s['count']])
            cursor.close()
            conn.close()
        except Exception as error:
            print(error)

        return state_list

    def fetch_ricci(self):
        state_list = np.append([], [])
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "select"
            for i, ricci in enumerate(self.ricci_names):
                sql = sql + ' @@' + ricci
                if i < self.ricci_num - 1:
                    sql = sql + ', '
            # print("fetch_ricci:", sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in range(self.ricci_num):
                state_list = np.append(state_list, result[0]["@@%s" % self.ricci_names[i]])
            cursor.close()
            conn.close()
        except Exception as error:
            print("fetch_ricci Error:", error)
        return state_list

    def max_connections(self):
        # if not self.max_connections_num:
        if 1:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                sql = "show global variables like 'max_connections';"
                cursor.execute(sql)
                self.max_connections_num = int(cursor.fetchone()["Value"])
                cursor.close()
                conn.close()
            except Exception as error:
                print(error)
        return self.max_connections_num

    def change_ricci_nonrestart(self, actions):
        try:
            conn = self._get_conn()
            for i in range(self.ricci_num):
                cursor = conn.cursor()
                if self.ricci_names[i] == 'max_connections':
                    self.max_connections_num = actions[i]
                sql = 'set global %s=%d' % (self.ricci_names[i], actions[i])
                cursor.execute(sql)
                # print(f"修改参数-{self.ricci_names[i]}:{actions[i]}")
                conn.commit()
            conn.close()
            return 1
        except Exception as error:
            conn.close()
            print("change_ricci_nonrestart error：", error)
            return 0


# Define the environment
class Environment(gym.Env):

    def __init__(self, edb, argus):

        self.edb = edb

        self.parser = SqlParser(argus)

        self.state_num = edb.internal_metric_num
        self.action_num = edb.ricci_num
        self.timestamp = int(time.time())

        # pfs = open('training-results/res_predict-' + str(self.timestamp), 'a')
        # pfs.write("%s\t%s\t%s\n" % ('iteration', 'throughput', 'latency'))
        # pfs.close()
        #
        # rfs = open('training-results/res_random-' + str(self.timestamp), 'a')
        # rfs.write("%s\t%s\t%s\n" % ('iteration', 'throughput', 'latency'))
        # rfs.close()

        ''' observation dim = system metric dim + query vector dim '''
        self.score = 0  # accumulate rewards

        self.o_dim = edb.internal_metric_num + len(self.edb.fetch_internal_metrics())
        self.o_low = np.array([-1e+10] * self.o_dim)
        self.o_high = np.array([1e+10] * self.o_dim)

        self.observation_space = spaces.Box(low=self.o_low, high=self.o_high, dtype=np.float32)
        # part 1: current system metric
        self.soliton_state = edb.fetch_internal_metrics()
        # print("Concatenated soliton_state:")
        # part 2: predicted system metric after executing the workload
        self.workload = argus["workload"]

        # TODO: 打开训练predict的方法后，此方法注释
        ################################################################################
        state0 = self.edb.fetch_internal_metrics()
        self.preheat()
        state1 = self.edb.fetch_internal_metrics()
        try:
            if self.parser.predict_sql_resource_value is None:
                self.parser.predict_sql_resource_value = state1 - state0
        except Exception as error:
            print("get predict_sql_resource_value error:", error)
        ################################################################################

        self.soliton_state = np.append(self.parser.predict_sql_resource(self.workload), self.soliton_state)

        ''' causet_action space '''
        # Offline
        # table_open_cache(1), max_connections(2), innodb_buffer_pool_instances(4),
        # innodb_log_files_in_group(5), innodb_log_file_size(6), innodb_purge_threads(7), innodb_read_io_threads(8)
        # innodb_write_io_threads(9),
        # Online
        # innodb_buffer_pool_size(3), max_binlog_cache_size(10), binlog_cache_size(11)
        # 1 2 3 11
        # exclude
        # innodb_file_per_table, skip_name_resolve, binlog_checksum,
        # binlog_format(dynamic, [ROW, STATEMENT, MIXED]),

        calculate_Ricci = []
        infer_Ricci = []
        for k in ricci_config.items():
            if k[1]['type'] == 'infer':
                infer_Ricci.append(k)
            else:
                calculate_Ricci.append(k)
        self.ricci_num = len(ricci_config)
        # self.a_low = np.array([ricci[1]['min_value']/ricci[1]['length'] for ricci in list(ricci_config.items())[:edb.ricci_num]])
        self.a_low = np.array([ricci[1]['min_value'] / ricci[1]['length'] for ricci in infer_Ricci])
        # self.a_high = np.array([ricci[1]['max_value']/ricci[1]['length'] for ricci in list(ricci_config.items())[:edb.ricci_num]])
        self.a_high = np.array([ricci[1]['max_value'] / ricci[1]['length'] for ricci in infer_Ricci])
        # self.length = np.array([ricci[1]['length'] for ricci in list(ricci_config.items())[:edb.ricci_num]])
        self.length = np.array([ricci[1]['length'] * 1.0 for ricci in infer_Ricci])
        self.action_space = spaces.Box(low=self.a_low, high=self.a_high, dtype=np.float32)
        self.default_action = self.a_low
        self.mem = deque(maxlen=int(argus['maxlen_mem']))  # [throughput, latency]
        self.predicted_mem = deque(maxlen=int(argus['maxlen_predict_mem']))
        self.ricci2pos = {ricci: i for i, ricci in enumerate(ricci_config)}
        self.seed()
        self.start_time = datetime.datetime.now()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def execute_command(self):
        self.edb.max_connections()
        # print(self.edb.max_connections_num)
        if self.parser.argus['thread_num_auto'] == '0':
            thread_num = int(self.parser.argus['thread_num'])
        else:
            thread_num = int(self.edb.max_connections_num) - 1
        run_job(thread_num, self.workload, self.parser.resfile)

    def preheat(self):
        self.execute_command()

    def fetch_action(self):
        while True:
            state_list = self.edb.fetch_ricci()
            if list(state_list):
                break
            time.sleep(5)
        return state_list

    # new_state, reward, done,
    def step(self, u, isPredicted, iteration, action_tmp=None):
        flag = self.edb.change_ricci_nonrestart(u)

        # if failing to tune Ricci, give a high panlty
        if not flag:
            return self.soliton_state, -10e+4, self.score, 1

        self.execute_command()
        throughput, latency = self._get_throughput_latency()
        # ifs = open('fl1', 'r')
        # print(str(len(self.mem)+1)+"\t"+str(throughput)+"\t"+str(latency))
        cur_time = datetime.datetime.now()
        interval = (cur_time - self.start_time).seconds
        self.mem.append([throughput, latency])
        # 2 refetch soliton_state
        self._get_obs()
        # 3 cul reward(T, L)
        reward = self._calculate_reward(throughput, latency)
        '''
        reward = 0
        for i in range(u.shape[0]):
            tmp = u[i] / self.a_high[i]
            reward+=tmp
        # print("Performance: %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))
        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])
            if len(self.predicted_mem)%10 == 0:
                print("Predict List")
                print(self.predicted_mem)
       '''

        causet_action = self.fetch_action()

        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])

            # print("Predict %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            pfs = open('training-results/res_predict-' + str(self.timestamp), 'a')
            pfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            pfs.close()

            fetch_ricci = open('training-results/fetch_ricci_predict-' + str(self.timestamp), 'a')
            fetch_ricci.write(f"{str(iteration)}\t{str(list(causet_action))}\n")
            fetch_ricci.close()

            action_write = open('training-results/action_test_predict-' + str(self.timestamp), 'a')
            action_write.write(f"{str(iteration)}\t{str(list(u))}\n")
            action_write.write(f"{str(iteration)}\t{str(list(action_tmp))}\n")
            action_write.close()

            self.score = self.score + reward

        else:
            # print("Random %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            rfs = open('training-results/res_random-' + str(self.timestamp), 'a')
            rfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            rfs.close()

            action_write = open('training-results/action_random-' + str(self.timestamp), 'a')
            action_write.write(f"{str(iteration)}\t{str(list(u))}\n")
            action_write.close()

            fetch_ricci = open('training-results/fetch_ricci_random-' + str(self.timestamp), 'a')
            fetch_ricci.write(f"{str(iteration)}\t{str(list(causet_action))}\n")
            fetch_ricci.close()

        return self.soliton_state, reward, self.score, throughput

    def _get_throughput_latency(self):
        with open(self.parser.resfile, 'r') as f:
            try:
                for line in f.readlines():
                    a = line.split()
                    if len(a) > 1 and 'avg_qps(queries/s):' == a[0]:
                        throughput = float(a[1])
                    if len(a) > 1 and 'avg_lat(s):' == a[0]:
                        latency = float(a[1])
            finally:
                f.close()
            # print("throughput:{} \n latency:{}".format(throughput, latency))
            return throughput, latency

    def _calculate_reward(self, throughput, latency):
        if len(self.mem) != 0:
            dt0 = (throughput - self.mem[0][0]) / self.mem[0][0]
            dt1 = (throughput - self.mem[len(self.mem) - 1][0]) / self.mem[len(self.mem) - 1][0]
            if dt0 >= 0:
                rt = ((1 + dt0) ** 2 - 1) * abs(1 + dt1)
            else:
                rt = -((1 - dt0) ** 2 - 1) * abs(1 - dt1)

            dl0 = -(latency - self.mem[0][1]) / self.mem[0][1]

            dl1 = -(latency - self.mem[len(self.mem) - 1][1]) / self.mem[len(self.mem) - 1][1]

            if dl0 >= 0:
                rl = ((1 + dl0) ** 2 - 1) * abs(1 + dl1)
            else:
                rl = -((1 - dl0) ** 2 - 1) * abs(1 - dl1)

        else:  # initial causet_action
            rt = 0
            rl = 0
        reward = 1 * rl + 9 * rt
        return reward

    def _get_obs(self):
        self.soliton_state = self.edb.fetch_internal_metrics()
        self.soliton_state = np.append(self.parser.predict_sql_resource(self.workload), self.soliton_state)
        return self.soliton_state
