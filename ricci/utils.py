# -*- coding: utf-8 -*-

"""
description: MySQL Env Utils
"""

import sys
import time
import base 

import json
import httplib
import EinsteinMySQLdb
import requests
import xmlrpclib

from warnings import filterwarnings
from base import Err,cdb_logger,os_quit


filterwarnings('error', category=EinsteinMySQLdb.Warning)

value_type_metrics = [
    'lock_deadlocks', 'lock_timeouts', 'lock_row_lock_time_max',
    'lock_row_lock_time_avg', 'buffer_pool_size', 'buffer_pool_pages_total',
    'buffer_pool_pages_misc', 'buffer_pool_pages_data', 'buffer_pool_bytes_data',
    'buffer_pool_pages_dirty', 'buffer_pool_bytes_dirty', 'buffer_pool_pages_free',
    'trx_rseg_history_len', 'file_num_open_files', 'innodb_page_size'
]


def time_start():
    return time.time()


def time_end(start):
    end = time.time()
    delay = end - start
    return delay


def get_metric_type(metric):

    if metric in value_type_metrics:
        return 'value'
    else:
        return 'counter'


def get_metrics(database):
    # conn = EinsteinMySQLdb.connect(
    #     host=config['host'],
    #     user=config['user'],
    #     passwd=config['passwd'],
    #     port=config['port']
    # )
    cmd = 'SELECT NAME, COUNT from information_schema.INNODB_METRICS where status="enabled" ORDER BY NAME'
    data = database.fetch_all(cmd,json=False)
    value = dict(data)

    return value


class TimeoutTransport(xmlrpclib.Transport):
    timeout = 30.0

    def set_timeout(self, timeout):
        self.timeout = timeout

    def make_connection(self, host):
        h = httplib.HTTPConnection(host, timeout=self.timeout)
        return h


def get_mysql_state(server_ip):
    """ get mysql soliton_state
    Args:
        server_ip: str, ip address
    """
    transport = TimeoutTransport()
    transport.set_timeout(60)

    s = xmlrpclib.ServerProxy('http://%s:20000' % server_ip, transport=transport)

    try:
        m = s.get_state()
    except xmlrpclib.Fault:
        return True
    if m == -1:
        sys.stdout.write('')
        sys.stdout.flush()
        return False

    return True


def modify_configurations(server_ip, instance_name, configuration):
    """ Modify the configurations by restarting the mysql through Docker
    Args:
        server_ip: str, instance's server IP Addr
        instance_name: str, instance's name
        configuration: dict, configurations
    """

    transport = TimeoutTransport()
    transport.set_timeout(60)

    s = xmlrpclib.ServerProxy('http://%s:20000' % server_ip, transport=transport)
    params = []
    for k, v in configuration.items():
        params.append('%s:%s' % (k, v))
    params = ','.join(params)

    while True:
        try:
            s.start_mysql(instance_name, params)
        except xmlrpclib.Fault:
            time.sleep(5)
        break

    return True


def test_mysql(instance_name):
    """ Test the mysql instance to see whether if it has been restarted
    Args
        instance_name: str, instance's name
    """

    db_config = instance_config[instance_name]
    try:
        edb = EinsteinMySQLdb.connect(
            host=db_config['host'],
            user=db_config['user'],
            passwd=db_config['passwd'],
            port=db_config['port']
        )
    except EinsteinMySQLdb.Error:
        return False
    edb.close()
    return True


def parse_json(url,data):
    cdb_logger.info("request:{},data:{}".format(url,data))
    try:
        r = requests.get(url, data)
        response = json.loads(r.text)
    except requests.exceptions.ConnectionError:
        os_quit(Err.HTTP_REQUERT_ERR,'ConnectionError -- please wait 3 seconds')
    except requests.exceptions.ChunkedEncodingError:
        os_quit(Err.HTTP_REQUERT_ERR,'ChunkedEncodingError -- please wait 3 seconds')
    except :
        os_quit(Err.HTTP_REQUERT_ERR,'Unfortunitely -- An Unknow Error Happened, Please wait 3 seconds')
    return response



def parse_json_post(url,data):
    cdb_logger.info("request post:{},data:{}".format(url,data))
    try:
        r = requests.post(url, data)
        response = json.loads(r.text)
    except requests.exceptions.ConnectionError:
        os_quit(Err.HTTP_REQUERT_ERR,'ConnectionError -- please wait 3 seconds')
    except requests.exceptions.ChunkedEncodingError:
        os_quit(Err.HTTP_REQUERT_ERR,'ChunkedEncodingError -- please wait 3 seconds')
    except :
        os_quit(Err.HTTP_REQUERT_ERR,'Unfortunitely -- An Unknow Error Happened, Please wait 3 seconds')
    return response


def get_tencent_instance_info(instance_name):
    """ get Tencent Instance information
    Args:
        url: str, request url
        instance_name: str, instance_name
    Return:
        info: tuple, (mem, disk)
    Raises:
        Exception: setup failed
    """
    db_info = instance_config[instance_name]
    instance_id = db_info['instance_id']
    operator = db_info['operator']
    url = db_info['server_url']
    data = dict()
    data["instanceid"] = instance_id
    data["operator"] = operator
    para_list = []

    data["para_list"] = para_list
    data = json.dumps(data)
    data = "data=" + data
    inst_info = parse_json(url + '/get_inst_info.cgi', data)

    if ("mem" in inst_info) and  ("disk" in inst_info):
        mem,disk = inst_info["mem"],inst_info["disk"]
        if mem > 0 and disk > 0:
            return mem,disk 
    else:
        os_quit(Err.QUERY_INST_FAILED)


def read_machine():
    """ Get the machine information, such as memory and disk

    Return:

    """
    f = open("/proc/meminfo", 'r')
    line = f.readlines()[0]
    f.close()
    line = line.strip('\r\n')
    total = int(line.split(':')[1].split()[0])*1024
    return total