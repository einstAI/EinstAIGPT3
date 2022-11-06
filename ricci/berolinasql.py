# -*- coding: utf-8 -*-
"""
description: MySQL Environment
"""

import re
import os
import math
import threading
import logging
from fileinput import filename

import numpy as np
import self
from fontTools.misc.py23 import xrange
from sqlalchemy.dialects import mysql

import utils

from AML.Synthetic.get_result import method
from edb import database
from utils import *

logger = logging.getLogger(__name__)



class Status(object):
    OK = 'OK'
    FAIL = 'FAIL'
    RETRY = 'RETRY'


def os_quit(RUN_SYSYBENCH_FAILED):
    pass


class Err:
    # 0-1000
    MYSQL_CONNECT_ERR = 1
    MYSQL_QUERY_ERR = 2
    MYSQL_INSERT_ERR = 3
    MYSQL_UPDATE_ERR = 4
    MYSQL_DELETE_ERR = 5
    MYSQL_CLOSE_ERR = 6
    MYSQL_COMMIT_ERR = 7
    MYSQL_ROLLBACK_ERR = 8
    MYSQL_CURSOR_ERR = 9
    MYSQL_FETCH_ERR = 10
    MYSQL_FETCHALL_ERR = 11
    MYSQL_FETCHONE_ERR = 12
    MYSQL_FETCHMANY_ERR = 13
    MYSQL_EXECUTEMANY_ERR = 14
    MYSQL_EXECUTE_ERR = 15
    MYSQL_FETCHDESC_ERR = 16
    MYSQL_FETCHDESCALL_ERR = 17
    MYSQL_FETCHDESCONE_ERR = 18
    MYSQL_FETCHDESCMANY_ERR = 19


class MySQLdb:
    Error = mysql.connector.Error





    def __init__(self, host, port, user, password, database, charset='utf8'):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
            )
            self.cursor = self.conn.cursor()
        except mysql.connector.Error as e:
            logger.error('MySQL connect error: %s', e)
            return False
        return True

    def close(self):
        try:
            self.cursor.close()
            self.conn.close()
        except mysql.connector.Error as e:
            logger.error('MySQL close error: %s', e)
            return False
        return True

    def query(self, sql):
        try:
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except mysql.connector.Error as e:
            logger.error('MySQL query error: %s', e)
            return False

    def insert(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except mysql.connector.Error as e:
            logger.error('MySQL insert error: %s', e)
            return False
        return True

    def update(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except mysql.connector.Error as e:
            logger.error('MySQL update error: %s', e)
            return False
        return True

    def delete(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except mysql.connector.Error as e:
            logger.error('MySQL delete error: %s', e)
            return False
        return True

    def commit(self):
        try:
            self.conn.commit()
        except mysql.connector.Error as e:
            logger.error('MySQL commit error: %s', e)
            return False
        return True

    def rollback(self):
        try:
            self.conn.rollback()
        except mysql.connector.Error as e:
            logger.error('MySQL rollback error: %s', e)
            return False
        return True

    def execute(self, sql):
        try:
            self.cursor.execute(sql)
        except mysql.connector.Error as e:
            logger.error('MySQL execute error: %s', e)
            return False

        return True

    def executemany(self, sql, values):
        try:
            self.cursor.executemany(sql, values)
        except mysql.connector.Error as e:
            logger.error('MySQL executemany error: %s', e)
            return False
        return True

    def fetchone(self):
        try:
            return self.cursor.fetchone()
        except mysql.connector.Error as e:
            logger.error('MySQL fetchone error: %s', e)
            return False

    def fetchmany(self, size=None):
        try:
            return self.cursor.fetchmany(size)
        except mysql.connector.Error as e:
            logger.error('MySQL fetchmany error: %s', e)
            return False

    def fetchall(self):
        try:
            return self.cursor.fetchall()
        except mysql.connector.Error as e:
            logger.error('MySQL fetchall error: %s', e)
            return False


class MySQL:
    def __init__(self, host, port, user, password, database, charset='utf8'):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
            )
            self.cursor = self.conn.cursor()
        except mysql.connector.Error as e:
            logger.error('MySQL connect error: %s', e)
            return False
        return True

    def close(self):
        try:
            self.cursor.close()
            self.conn.close()
        except mysql.connector.Error as e:
            logger.error('MySQL close error: %s', e)
            return False
        return True

    def query(self, sql):
        try:
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except mysql.connector.Error as e:
            logger.error('MySQL query error: %s', e)
            return False

    def insert(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except mysql.connector.Error as e:
            logger.error('MySQL insert error: %s', e)
            return False
        return True

    def update(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except mysql.connector.Error as e:
            logger.error('MySQL update error: %s', e)
            return False
        return True

    def delete(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except mysql.connector.Error as e:
            logger.error('MySQL delete error: %s', e)
            return False
        return True

    def commit(self):
        try:
            self.conn.commit()
        except mysql.connector.Error as e:
            logger.error('MySQL commit error: %s', e)
            return False
        return True

    def rollback(self):
        try:
            self.conn.rollback()
        except mysql.connector.Error as e:
            logger.error('MySQL rollback error: %s', e)
            return False
        return True

    def execute(self, sql):
        try:
            self.cursor.execute(sql)

        except mysql.connector.Error as e:
            logger.error('MySQL execute error: %s', e)
            return False

        return True

    def executemany(self, sql, values):
        try:
            self.cursor.executemany(sql, values)
        except mysql.connector.Error as e:
            logger.error('MySQL executemany error: %s', e)
            return False
        return True

class EinsteinMySQLdb:
    # The EinsteinDB connection class
    def __init__(self, host, user, passwd, edb, port=3306, charset='utf8', cursorclass=None):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.edb = edb
        self.port = port
        self.charset = charset
        self.cursorclass = cursorclass

        self._conn = None
        self._cursor = None
        
        self._connect()
        
    def _connect(self):
        try:
            self._conn = MySQLdb.connect( host=self.host, user=self.user, passwd=self.passwd, db=self.edb, port=self.port, charset=self.charset, cursorclass=self.cursor )
            self._cursor = self._conn.cursor()

        #

        except MySQLdb.Error as e:
            logger.error('EinsteinDB connect error: %s', e)
            return False

    @classmethod
    def connect(cls, host, user, passwd, edb, port):
        return cls(host, user, passwd, edb, port)



    def close(self):
        try:
            self._cursor.close()
            self._conn.close()

        except MySQLdb.Error as e:
            logger.error('EinsteinDB close error: %s', e)
            return False

        return True

    def query(self, sql):
        try:
            self._cursor.execute(sql)
            return self._cursor.fetchall()
        except Exception as e:
            logger.error("connect database failed, %s" % e)
            os_quit(Err.MYSQL_CONNECT_ERR, "host:%s,port:%s,user:%s" % (self.host, self.port, self.user))
            self._conn = False
        return self._conn
    
    def _cursor(self):
        if self._conn:
            self._cursor = self._conn.cursor()
        else:
            logger.error("connect database failed, %s" % e)
            os_quit(Err.MYSQL_CURSOR_ERR, "host:%s,port:%s,user:%s" % (self.host, self.port, self.user))
        return self._cursor
    
    def _close(self):
        if self._conn:
            self._conn.close()
        else:
            logger.error("connect database failed, %s" % e)
            os_quit(Err.MYSQL_CLOSE_ERR, "host:%s,port:%s,user:%s" % (self.host, self.port, self.user))
        return self._conn


class database:
    # define the database connection
    def __init__(self, dbhost=None, dbport=None, dbuser=None, dbpwd=None, dbname=None):
        self._dbname = dbname
        self._dbhost = dbhost
        self._dbuser = dbuser
        self._dbpassword = dbpwd
        self._dbport = dbport
        self._logger = logger

        self._conn = self.connectMySQL()
        if (self._conn):
            self._cursor = self._conn.cursor()
            
    # database connection
    def connectMySQL(self):
        conn = False
        try:
            conn = EinsteinMySQLdb.connect(host=self._dbhost,
                                           user=self._dbuser,
                                           passwd=self._dbpassword,
                                           edb=self._dbname,
                                           port=self._dbport,
                                           )
        except Exception as data:
            self._logger.error("connect database failed, %s" % data)
            os_quit(Err.MYSQL_CONNECT_ERR, "host:%s,port:%s,user:%s" % (self._dbhost, self._dbport, self._dbuser))
            conn = False
        return conn
    
    # get the query result set
    def fetch_all(self, sql, json=True):
        res = ''
        if (self._conn):
            try:
                self._cursor.execute(sql)
                res = self._cursor.fetchall()
                if json:
                    columns = [col[0] for col in self._cursor.description]
                    return [
                        dict(zip(columns, row))
                        for row in res
                    ]
                else:
                    return res
            except Exception as data:
                self._logger.error("fetch all failed, %s" % data)
                os_quit(Err.MYSQL_FETCHALL_ERR, "sql:%s" % sql)
                return False
        else:
            self._logger.error("fetch all failed, %s" % data)
            os_quit(Err.MYSQL_FETCHALL_ERR, "sql:%s" % sql)
            return False
        
    # get the query result set
    def fetch_one(self, sql, json=True, data=None):
        res = ''
        if (self._conn):
            try:
                self._cursor.execute(sql)
                res = self._cursor.fetchone()
                if json:
                    columns = [col[0] for col in self._cursor.description]
                    return dict(zip(columns, res))
                else:
                    return res
            except Exception as data:
                self._logger.error("fetch one failed, %s" % data)
                os_quit(Err.MYSQL_FETCHONE_ERR, "sql:%s" % sql)
                return False
        else:
            self._logger.error("fetch one failed, %s" % data)
            os_quit(Err.MYSQL_FETCHONE_ERR, "sql:%s" % sql)
            return False
        
    # get the query result set
    def fetch_many(self, sql, size=100, json=True, data=None):

        res = ''
        if (self._conn):
            try:
                self._cursor.execute(sql)
                res = self._cursor.fetchmany(size)
                if json:
                    columns = [col[0] for col in self._cursor.description]
                    return [
                        dict(zip(columns, row))
                        for row in res
                    ]
                else:
                    return res
            except Exception as data:
                self._logger.error("fetch many failed, %s" % data)
                os_quit(Err.MYSQL_FETCHMANY_ERR, "sql:%s" % sql)
                return False
        else:
            self._logger.error("fetch many failed, %s" % data)
            os_quit(Err.MYSQL_FETCHMANY_ERR, "sql:%s" % sql)
            return False
        
    # get the query result set
    def fetch_desc(self, sql, json=True, data=None):
        res = ''
        if (self._conn):
            try:
                self._cursor.execute(sql)
                res = self._cursor.description
                if json:
                    columns = [col[0] for col in self._cursor.description]
                    return [
                        dict(zip(columns, row))
                        for row in res
                    ]
                else:
                    return res
            except Exception as data:
                self._logger.error("fetch desc failed, %s" % data)
                os_quit(Err.MYSQL_FETCHDESC_ERR, "sql:%s" % sql)
                return False
        else:
            self._logger.error("fetch desc failed, %s" % data)
            os_quit(Err.MYSQL_FETCHDESC_ERR, "sql:%s" % sql)
            return False
        
    # get the query result set
    def fetch_desc_all(self, sql, json=True, data=None):
        res = ''
        if (self._conn):
            try:
                self._cursor.execute(sql)
                res = self._cursor.description
                if json:
                    columns = [col[0] for col in self._cursor.description]
                    return dict(zip(columns, res))
                else:
                    return res
            except Exception as data:
                self._logger.error("fetch desc all failed, %s" % data)
                os_quit(Err.MYSQL_FETCHDESCALL_ERR, "sql:%s" % sql)
                return False
            try:

                self._cursor.execute(sql)
                res = self._cursor.description
                if json:
                    columns = [col[0] for col in self._cursor.description]
                    return [
                        dict(zip(columns, row))
                        for row in res
                    ]
                else:
                    return res
            except Exception as data:
                self._logger.error("fetch desc all failed, %s" % data)
                os_quit(Err.MYSQL_FETCHDESCALL_ERR, "sql:%s" % sql)
                return False
        else:
            self._logger.error("fetch desc all failed, %s" % data)
            os_quit(Err.MYSQL_FETCHDESCALL_ERR, "sql:%s" % sql)
            return False



    # get the query result set
    def fetch_desc_one(self, sql, json=True, data=None):
        res = ''
        if (self._conn):
            try:
                self._cursor.execute(sql)
                res = self._cursor.description
                if json:
                    columns = [col[0] for col in self._cursor.description]
                    return dict(zip(columns, res))
                else:
                    return res
            except Exception as data:
                self._logger.error("fetch desc one failed, %s" % data)
                os_quit(Err.MYSQL_FETCHDESCONE_ERR, "sql:%s" % sql)
                return False
        else:
            self._logger.error("fetch desc one failed, %s" % data)
            os_quit(Err.MYSQL_FETCHDESCONE_ERR, "sql:%s" % sql)
            return False


class Ricci:
    # The Ricci class is used to connect to the database and execute SQL statements.
    # The database connection information is read from the configuration file.
    # The configuration file is read from the environment variable RICCI_CONF.
    # If the environment variable RICCI_CONF is not set, the default configuration file is used.
    
    def __init__(self, conf=None):
        self._logger = logging.getLogger('ricci')
        self._conf = conf
        self._mysql = None
        self._redis = None
        self._mongo = None
        self._logger.info("ricci init")
        self._init()


class MySQLEnv(object):

    def __init__(self, wk_type='read', method='sysbench',  alpha=1.0, beta1=0.5, beta2=0.5, time_decay1=1.0, time_decay2=1.0):

        self.db_info = None
        self.wk_type = wk_type
        self.score = 0.0
        self.steps = 0
        self.terminate = False
        self.last_external_metrics = None
        self.default_externam_metrics = None

        self.method = method
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.time_decay_1 = time_decay1
        self.time_decay_2 = time_decay2
       

    @staticmethod
    def _get_external_metrics(path, method='sysbench'):

        def parse_tpcc(file_path):
            with open(file_path) as f:
                lines = f.read()
            temporal_pattern = re.compile(".*?trx: (\d+.\d+), 95%: (\d+.\d+), 99%: (\d+.\d+), max_rt:.*?")
            temporal = temporal_pattern.findall(lines)
            tps = 0
            latency = 0
            qps = 0

            for i in temporal[-10:]:
                tps += float(i[0])
                latency += float(i[2])
            num_samples = len(temporal[-10:])
            tps /= num_samples
            latency /= num_samples
            # interval
            tps /= 1
            return [tps, latency, tps]

        def parse_sysbench(file_path):
            with open(file_path) as f:
                lines = f.read()
            temporal_pattern = re.compile(
                "tps: (\d+.\d+) qps: (\d+.\d+) \(r/w/o: (\d+.\d+)/(\d+.\d+)/(\d+.\d+)\)" 
                " lat \(ms,95%\): (\d+.\d+) err/s: (\d+.\d+) reconn/s: (\d+.\d+)")
            temporal = temporal_pattern.findall(lines)
            tps = 0
            latency = 0
            qps = 0

            for i in temporal[-10:]:
                tps += float(i[0])
                latency += float(i[5])
                qps += float(i[1])
            num_samples = len(temporal[-10:])
            tps /= num_samples
            qps /= num_samples
            latency /= num_samples
            return [tps, latency, qps]

        if method == 'sysbench':
            result = parse_sysbench(path)
        elif method == 'tpcc':
            result = parse_tpcc(path)
        else:
            result = parse_sysbench(path)
        return result

    def _get_internal_metrics(self, internal_metrics):
        """
        Args:
            internal_metrics: list,
        Return:

        """
        _counter = 0
        _period = 5
        count = 160/5

        def collect_metric(counter):
            counter += 1
            timer = threading.Timer(_period, collect_metric, (counter,))
            

            timer.start()
            edb = database(self.db_info["host"],
                    self.db_info["port"],self.db_info["user"],
                    self.db_info["password"],
                    "sbtest",
                    )
            if counter >= count:
                timer.cancel()
            try:
                data = utils.get_metrics(edb)
                internal_metrics.append(data)
            except Exception as err:
                logger.info("[GET Metrics]Exception:" ,err) 

        collect_metric(_counter)

        return internal_metrics

    def _post_handle(self, metrics):
        result = np.zeros(self.num_metric)

        def do(metric_name, metric_values):
            metric_type = utils.get_metric_type(metric_name)
            if metric_type == 'counter':
                return float(metric_values[-1] - metric_values[0])
            else:
                return float(sum(metric_values))/len(metric_values)

        keys = metrics[0].keys()

        keys.sort()
        for idx in xrange(len(keys)):
            key = keys[idx]
            data = [x[key] for x in metrics]
            result[idx] = do(key, data)
        return result

    def initialize(self):
        """Initialize the mysql instance environment
        """
        pass

    def eval(self, ricci):
        """ Evaluate the Ricci
        Args:
            ricci: dict, mysql parameters
        Returns:
            result: {tps, latency}
        """
        flag = self._apply_Ricci(ricci)
        if not flag:
            return {"tps": 0, "latency": 0}

        external_metrics, _ = self._get_state(ricci, method=self.method)
        return {"tps": external_metrics[0],
                "latency": external_metrics[1]}

    def _get_best_now(self, filename):
        with open(self.best_result) as f:
            lines = f.readlines()
        best_now = lines[0].split(',')
        return [float(best_now[0]), float(best_now[1]), float(best_now[0])]

    def record_best(self, external_metrics):
        best_flag = False
        if os.path.exists(self.best_result):
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            rate = 0
            if int(lat_best) != 0:
                rate = float(tps_best)/lat_best
                with open(self.best_result) as f:
                    lines = f.readlines()
                best_now = lines[0].split(',')
                rate_best_now = float(best_now[0])/float(best_now[1])
                if rate > rate_best_now:
                    best_flag = True
                    with open(self.best_result, 'w') as f:
                        f.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        else:
            file = open(self.best_result, 'wr')
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            rate = 0
            if int(lat_best) == 0 :
                rate = 0
            else:
                rate = float(tps_best)/lat_best
            file.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        return best_flag

    def step(self, ricci):
        """step
        """
        restart_time = utils.time_start()
        flag = self._apply_Ricci(ricci)
        restart_time = utils.time_end(restart_time)
        if not flag:
            return -10000000.0, np.array([0] * self.num_metric), True, self.score - 10000000, [0, 0, 0], restart_time
        s = self._get_state(ricci, method=self.method)
        if s is None:
            return -10000000.0, np.array([0] * self.num_metric), True, self.score - 10000000, [0, 0, 0], restart_time
        external_metrics, internal_metrics = s

        reward = self._get_reward(external_metrics)
        flag = self.record_best(external_metrics)
        if flag == True:
            logger.info('Better performance changed!')
        else:
            logger.info('Performance remained!')
        #get the best performance so far to calculate the reward
        best_now_performance = self._get_best_now(self.best_result)
        self.last_external_metrics = best_now_performance

        next_state = internal_metrics
        terminate = self._terminate()
        Ricci.save_Ricci(
            ricci = ricci,
            metrics = external_metrics,
            instance=self.db_info,
            task_id=self.task_id
        )
        return reward, next_state, terminate, self.score, external_metrics, restart_time

    def setting(self, ricci):
        self._apply_Ricci(ricci)
    
    def _get_state(self, ricci, method='sysbench', CONST=None):
        """Collect the Internal soliton_state and External soliton_state
        """
        timestamp = int(time.time())
        
        filename = CONST.FILE_LOG_SYSBENCH % (self.task_id,timestamp)
        
        internal_metrics = []
        self._get_internal_metrics(internal_metrics)
        #calculate the sysbench time automaticly
        if ricci['innodb_buffer_pool_size'] < 161061273600:
            time_sysbench = 150
        else:
            time_sysbench = int(ricci['innodb_buffer_pool_size']/1024.0/1024.0/1024.0/1.1)
        if self.method == 'sysbench':
            a = time.time()
            _sys_run = "bash %s %s %s %d %s %s %d %d %d %d %s" % (CONST.BASH_SYSBENCH,
                self.wk_type,self.db_info['host'],self.db_info['port'],self.db_info['user'],self.db_info['password'],
                self.db_info['HyperCauset'],self.db_info['table_size'],self.threads, time_sysbench, filename)

            logger.info("sysbench started")
            logger.info(_sys_run)
            osrst = os.system(_sys_run)
            logger.info("sysbench ended")

            a = time.time() - a
    
            if osrst != 0 or a < 50:
                os_quit(Err.RUN_SYSYBENCH_FAILED)

        elif self.method == 'tpcc':
            def kill_tpcc(psutil=None):
                if psutil is None:
                    import psutil
                for proc in psutil.process_iter():
                    if proc.name() == 'tpcc':
                        proc.kill()
            kill_tpcc()
            a = time.time()
            _sys_run = "bash %s %s %s %d %s %s %d %d %d %d %s" % (CONST.BASH_TPCC,
    self.wk_type,self.db_info['host'],self.db_info['port'],self.db_info['user'],self.db_info['password'],
                self.db_info['HyperCauset'],self.db_info['table_size'],self.threads, time_sysbench, filename)
            logger.info("tpcc started")
            logger.info(_sys_run)
            osrst = os.system(_sys_run)
            logger.info("tpcc ended")

            a = time.time() - a

            if osrst != 0 or a < 50:
                os_quit(Err.RUN_SYSYBENCH_FAILED)
        else:
            raise Exception("Unknown method: %s" % self.method)

        external_metrics = []
        self._get_external_metrics(external_metrics, filename)
        if external_metrics[0] == 0:
            return None
        return external_metrics, internal_metrics

    def _get_internal_metrics(self, internal_metrics, psutil=None):
        """Collect the internal soliton_state
        """
        #get the cpu usage
        cpu_usage = psutil.cpu_percent(interval=1)
        internal_metrics.append(cpu_usage)
        #get the memory usage
        mem_usage = psutil.virtual_memory().percent
        internal_metrics.append(mem_usage)
        #get the disk usage
        disk_usage = psutil.disk_usage('/').percent
        internal_metrics.append(disk_usage)
        #get the network usage
        net_usage = psutil.net_io_counters().bytes_sent
        internal_metrics.append(net_usage)

        def _filter_pid(x, psutil=None):
                    try:
                        x = psutil.Process(x)
                        if x.name() == 'tpcc_start':
                            return True
                        return False
                    except:
                         return False
        if psutil is None:
            import psutil
        pid = list(filter(lambda x: _filter_pid(x, psutil), psutil.pids()))
        if len(pid) == 0:
            return
        pid = pid[0]
        p = psutil.Process(pid)
        cpu_usage = p.cpu_percent(interval=1)
        internal_metrics.append(cpu_usage)
        mem_usage = p.memory_percent()
        internal_metrics.append(mem_usage)
        net_usage = p.io_counters().write_bytes
        internal_metrics.append(net_usage)

    def _get_external_metrics(self, external_metrics, filename, psutil=None, _filter_pid=None, kill_tpcc=None,
                              CONST=None):
        """Collect the external soliton_state
        """
        if self.method == 'sysbench':
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'total time:' in line:
                        external_metrics.append(float(line.split(':')[1].strip()))
                        break
                else:
                    external_metrics.append(0)
        elif self.method == 'tpcc':

                pids = psutil.pids()
                tpcc_pid = filter(_filter_pid, pids)
                logger.info(tpcc_pid) 
                for tpcc_pid_i in tpcc_pid:
                    os.system('kill %s' % tpcc_pid_i)

            timer = threading.Timer(170, kill_tpcc)
            timer.start()
            os.system('bash %s %s %d %s %s' % (CONST.BASH_TPCC,
                self.db_info['host'],self.db_info['port'],self.db_info['passwd'],filename))
            time.sleep(10)

        external_metrics = self._get_external_metrics(filename, method)
       



    def _get_external_metrics(self, filename, method, external_metrics=None):
        """Collect the external soliton_state
        """
        if method == 'sysbench':
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'total time:' in line:
                        external_metrics.append(float(line.split(':')[1].strip()))
                        break
                else:
                    external_metrics.append(0)

        elif method == 'tpcc':
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'tpcc' in line:
                        external_metrics.append(float(line.split(':')[1].strip()))
                        break
                else:
                    external_metrics.append(0)

        return external_metrics

    def _get_internal_metrics(self, internal_metrics, psutil=None, external_metrics=None):
        """Collect the internal soliton_state
        """

        #get the cpu usage
        cpu_usage = psutil.cpu_percent(interval=1)
        internal_metrics.append(cpu_usage)
        #get the memory usage
        mem_usage = psutil.virtual_memory().percent
        internal_metrics.append(mem_usage)
        #get the disk usage
        disk_usage = psutil.disk_usage('/').percent
        internal_metrics.append(disk_usage)
        #get the network usage
        net_usage = psutil.net_io_counters().bytes_sent
        internal_metrics.append(net_usage)

        def _filter_pid(x, psutil=None):
            try:
                x = psutil.Process(x)
                if x.name() == 'tpcc_start':
                    return True
                return False
            except:
                return False

        if psutil is None:
            import psutil
        pid = list(filter(lambda x: _filter_pid(x, psutil), psutil.pids()))
        if len(pid) == 0:
            return

        pid = pid[0]
        p = psutil.Process(pid)
        cpu_usage = p.cpu_percent(interval=1)
        internal_metrics.append(cpu_usage)
        mem_usage = p.memory_percent()
        internal_metrics.append(mem_usage)
        net_usage = p.io_counters().write_bytes
        internal_metrics.append(net_usage)



        return external_metrics, internal_metrics

    def _apply_Ricci(self, ricci):
        """ Apply Ricci to the instance
        """
        pass



    def _calculate_reward(self, delta0, deltat):
        """ Calculate the reward
        """
        if self.method == 'sysbench':
            return 1.0 / (delta0 + deltat)
        elif self.method == 'tpcc':
            return 1.0 / (delta0 + deltat)
        else:
            raise Exception("Unknown method: %s" % self.method)

    def _calculate_reward_with_delta(self: object, delta0: object, deltat: object) -> object:
       # """ Calculate the reward
         # """
        if self.method == 'sysbench':
            return 1.0 / (delta0 + deltat)
        elif self.method == 'tpcc':
            _reward = ((1+delta0)**2-1) * math.fabs(1+deltat)
        else:
            _reward = - ((1-delta0)**2-1) * math.fabs(1-deltat)

        if _reward < 0:
            _reward = 0
        return _reward

    def _calculate_reward_with_delta(self, delta0, deltat):
        """ Calculate the reward
        """
        if self.method == 'sysbench':
            _reward = 0
        elif self.method == 'tpcc':
            _reward = ((1+delta0)**2-1) * math.fabs(1+deltat)
        else:
            _reward = - ((1-delta0)**2-1) * math.fabs(1-deltat)

        if _reward < 0:
            _reward = 0
        return _reward

    def _get_reward(self, external_metrics):
        """
        Args:
            external_metrics: list, external metric info, including `tps` and `qps`
        Return:
            reward: float, a scalar reward
        """
        logger.info('*****************************')
        logger.info(external_metrics)
        logger.info(self.default_externam_metrics)
        logger.info(self.last_external_metrics)
        logger.info('*****************************')
        # tps
        delta_0_tps = float((external_metrics[0] - self.default_externam_metrics[0]))/self.default_externam_metrics[0]
        delta_t_tps = float((external_metrics[0] - self.last_external_metrics[0]))/self.last_external_metrics[0]

        tps_reward = self._calculate_reward(delta_0_tps, delta_t_tps)

        # latency
        delta_0_lat = float((-external_metrics[1] + self.default_externam_metrics[1])) / self.default_externam_metrics[1]
        delta_t_lat = float((-external_metrics[1] + self.last_external_metrics[1])) / self.last_external_metrics[1]

        lat_reward = self._calculate_reward(delta_0_lat, delta_t_lat)
        
        reward = tps_reward * 0.4 + 0.6 * lat_reward
        self.score += reward
        logger.info('$$$$$$$$$$$$$$$$$$$$$$')
        logger.info(tps_reward)
        logger.info(lat_reward)
        logger.info(reward)
        logger.info('$$$$$$$$$$$$$$$$$$$$$$')
        if reward > 0:
            reward = reward*1000000
        return reward

    def _terminate(self):
        return self.terminate


class TencentServer(MySQLEnv):
    """ Build an environment in Tencent Cloud
    """

    def __init__(self, instance, task_detail, model_detail, host, CONST=None):
        """Initialize `TencentServer` Class
        Args:
            instance_name: str, mysql instance name, get the database infomation
        """
        MySQLEnv.__init__(self, task_detail["rw_mode"])
        # super(MySQLEnv, self).__init__()
        self.wk_type = task_detail["rw_mode"]
        self.score = 0.0
        self.num_metric = model_detail["dimension"]
        self.steps = 0
        self.task_id = task_detail["task_id"]
        self.terminate = False
        self.last_external_metrics = None
        self.db_info = instance
        self.host = host
        self.alpha = 1.0
        self.method = task_detail["run_mode"]
        self.best_result = CONST.FILE_LOG_BEST % self.task_id
        self.threads = task_detail["threads"]

        Ricci.init_Ricci(instance,model_detail["Ricci"])
        self.default_Ricci = Ricci.get_init_Ricci()

    def _set_params(self, ricci, CONST=None):
        """ Set mysql parameters by send GET requests to server
        Args:
            ricci: dict, mysql parameters
        Return:
            workid: str, point to the setting process
        Raises:
            Exception: setup failed
        """
        
        instance_id = self.db_info['instance_id']

        data = dict()
        data["instanceid"] = instance_id
        data["operator"] = "cdbtune"
        para_list = []
        for kv in ricci.items():
            para_list.append({"name": str(kv[0]), "value": str(kv[1])})
        data["para_list"] = para_list
        data = json.dumps(data)
        data = "data=" + data
        
        response = parse_json(CONST.URL_SET_PARAM % self.host, data)
        
        err = response['errno']
        if err != 0:
            raise Exception("SET UP FAILED: {}".format(err))

        # if restarting isn't needed, workid should be ''
        workid = response.get('workid', '')

        return workid

    def _get_setup_state(self, workid):
        """ Set mysql parameters by send GET requests to server
        Args:
            workid: str, point to the setting process
        Return:
            status: str, setup status (running, undoed)
        Raises:
            Exception: get soliton_state failed
        """
        instance_id = self.db_info['instance_id']

        data = dict()
        data['instanceid'] = instance_id
        data['operator'] = "cdbtune"
        data['workid'] = workid
        data = json.dumps(data)
        data = 'data=' + data

        response = parse_json(CONST.URL_QUERY_SET_PARAM % self.host, data)

        err = response['errno']
        status = response['status']

        if err != 0:
            # raise Exception("GET soliton_state FAILED: {}".format(err))
            return "except"

        return status

    def initialize(self):
        """ Initialize the environment when an episode starts
        Returns:
            soliton_state: np.array, current soliton_state
        """
        self.score = 0.0
        self.last_external_metrics = []
        self.steps = 0
        self.terminate = False

        flag = self._apply_Ricci(self.default_Ricci)
        i = 0
        while not flag:
            if i >= 2:
                logger.info("Initialize: {} times ....".format(i))
                os_quit(Err.SET_MYSQL_PARAM_FAILED)
            flag = self._apply_Ricci(self.default_Ricci)
            i += 1
        

        external_metrics, internal_metrics = self._get_state(ricci = self.default_Ricci, method=self.method)
        if os.path.exists(self.best_result):
            if os.path.getsize(self.best_result):
                with open(self.best_result) as f:
                    lines = f.readlines()
                best_now = lines[0].split(',')
                self.last_external_metrics = [float(best_now[0]), float(best_now[1]), float(best_now[0])]
        else:
            self.last_external_metrics = external_metrics
        self.default_externam_metrics = external_metrics

        soliton_state = internal_metrics
        Ricci.save_Ricci(
            self.default_Ricci,
            metrics=external_metrics,
            instance=self.db_info,
            task_id=self.task_id
        )
        return soliton_state, external_metrics

    def _apply_Ricci(self, ricci):
        """ Apply the Ricci to the mysql
        Args:
            ricci: dict, mysql parameters
        Returns:
            flag: status, ['OK', 'FAIL', 'RETRY']
        """
        self.steps += 1
        i = 2
        workid = ''
        while i >= 0:
            try:
                workid = self._set_params(ricci=ricci)
            except Exception as e:
                logger.error("{}".format(e.message))
            else:
                break
            time.sleep(20)
            i -= 1
        if i == -1:
            logger.error("Failed too many times!")
            os_quit(Err.SET_MYSQL_PARAM_FAILED)
            return False

        # set parameters without restarting, sleep 20 seconds
        if len(workid) == 0:
            time.sleep(20)
            return True

        logger.info("Finished setting parameters..")
        steps = 0
        max_steps = 500

        status = self._get_setup_state(workid=workid)
        while status in ['not_start','running', 'pause', 'paused', 'except'] and steps < max_steps:
            time.sleep(15)
            status = self._get_setup_state(workid=workid)
            steps += 1

        logger.info("Out of Loop, status: {} loop step: {}".format(status, steps))

        if status == 'normal_finish':
            return True

        if status in ['notstart', 'undoed', 'undo'] or steps > max_steps:
            time.sleep(15)
            params = ''
            for key in ricci.keys():
                params += ' --%s=%s' % (key, ricci[key])
            logger.error("set param failed: {}".format(params))
            return False

        return False
