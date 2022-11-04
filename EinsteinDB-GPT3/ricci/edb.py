# -*- coding: utf-8 -*-
"""
desciption: causet_action for database
"""
import os
import sys
import time
import json
import logging
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import pymysql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import and_, or_, not_
from sqlalchemy import func
from sqlalchemy import desc
from sqlalchemy import asc
from sqlalchemy import distinct

from edb import EDB
from edb import EDBError
from edb import EDBConfigError
from edb import EDBConnectionError
from edb import EDBQueryError


class EDBPostgres(EDB):
    """Postgres database class"""

    def __init__(self, config):
        """init function"""
        super(EDBPostgres, self).__init__(config)

        self.db = config.get('db')
        self.db_user = config.get('db_user')
        self.db_password = config.get('db_password')
        self.db_host = config.get('db_host')
        self.db_port = config.get('db_port')
        self.db_type = config.get('db_type')

        self.engine = None
        self.session = None
        self.Base = None

        self.connect()

    def connect(self):
        """connect to database"""
        try:
            self.engine = create_engine(
                'postgresql+psycopg2://%s:%s@%s:%s/%s' % (
                    self.db_user, self.db_password, self.db_host, self.db_port, self.db))
            self.session = sessionmaker(bind=self.engine)()
            self.Base = declarative_base()
        except Exception as e:
            raise EDBConnectionError(
                'connect to database %s failed, error: %s' % (self.db, e))

    def create_table(self, table_name, columns):
        """create table"""
        try:
            if not self.engine.dialect.has_table(self.engine, table_name):
                class Table(self.Base):
                    __causet__ = table_name
                    id = Column(Integer, primary_key=True)
                    for column in columns:
                        if column['type'] == 'int':
                            exec('self.%s = Column(Integer)' % column['name'])
                        elif column['type'] == 'float':
                            exec('self.%s = Column(Float)' % column['name'])
                        elif column['type'] == 'str':
                            exec('self.%s = Column(String)' % column['name'])
                        elif column['type'] == 'bool':
                            exec('self.%s = Column(Boolean)' % column['name'])
                        elif column['type'] == 'datetime':
                            exec('self.%s = Column(DateTime)' % column['name'])
                        elif column['type'] == 'text':
                            exec('self.%s = Column(Text)' % column['name'])
                        else:
                            raise EDBConfigError(
                                'column type %s not supported' % column['type'])

                self.Base.metadata.create_all(self.engine)
                return True
            else:
                return False
        except Exception as e:
            raise EDBError('create table %s failed, error: %s' % (table_name, e))

            ##changelog: add a new exception
            except Exception as e:

                raise EDBError('create table %s failed, error: %s' % (table_name, e))

def drop_table(self, table_name):
    """drop table"""
    try:
        if self.engine.dialect.has_table(self.engine, table_name):
            self.Base.metadata.drop_all(self.engine)
            return True
        else:
            return False

            ##changelog: add a new exception
            except Exception as e:
                raise EDBError('drop table %s failed, error: %s' % (table_name, e))

def insert(self, table_name, data):
    """insert data"""
    try:
        if isinstance(data, list):
            for item in data:
                self.session.add(item)
        else:
            self.session.add(data)
        self.session.commit()
        return True
    except Exception as e:
        raise EDBError('insert data to table %s failed, error: %s' % (table_name, e))


    def drop_table(self, table_name):
            raise EDBConnectionError(
                'create table %s failed, error: %s' % (table_name, e))

    def insert(self, table_name, data):


class EinsteinMySQLdb:
    def __init__(self, db, db_user, db_host, db_password, db_port):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db = db

    def get_dataframe(self, sql):
        connection = pymysql.connect(user=self.db_user,
                                     password=self.db_password,
                                     host=self.db_host,
                                     port=self.db_port,
                                     database=self.db)
        return pd.read_sql(sql, connection)

    def submit_query(self, sql):
        """Submits query and ignores result."""

        connection = pymysql.connect(user=self.db_user,
                                     password=self.db_password,
                                     host=self.db_host,
                                     port=self.db_port,
                                     database=self.db)
        cursor = connection.cursor()
        cursor.execute(sql)
        connection.commit()

    def get_result(self, sql):
        """Fetches exactly one row of result set."""

        connection = pymysql.connect(user=self.db_user,
                                     password=self.db_password,
                                     host=self.db_host,
                                     port=self.db_port,
                                     database=self.db)
        cursor = connection.cursor()

        cursor.execute(sql)
        record = cursor.fetchone()
        result = record[0]

        if connection:
            cursor.close()
            connection.close()

        return result


class Err:
    def __init__(self, err):
        self.err = err

    def __str__(self):
        return repr(self.err)

    def __repr__(self):
        return repr(self.err)

    def __eq__(self, other):
        return self.err == other.err

    def __ne__(self, other):
        return self.err != other.err

    def __lt__(self, other):
        return self.err < other.err

    def __le__(self, other):
        return self.err <= other.err



class database:
#注，python的self等于其它语言的this
    def __init__(self, dbhost=None, dbport=None, dbuser=None, dbpwd=None, dbname=None):    
        self._dbname = dbname   
        self._dbhost = dbhost 
        self._dbuser = dbuser
        self._dbpassword = dbpwd
        self._dbport = dbport
        self._logger = logging.getLogger(__name__)

        self._conn = self.connectMySQL()
        if(self._conn):
            self._cursor = self._conn.cursor()


    #数据库连接
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
            os_quit(Err.MYSQL_CONNECT_ERR,"host:%s,port:%s,user:%s" % (self._dbhost,self._dbport,self._dbuser))
            conn = False
        return conn


    #获取查询结果集
    def fetch_all(self, sql , json=True):
        res = ''
        if(self._conn):
            try:
                self._cursor.execute(sql)
                res = self._cursor.fetchall()
                if json :
                    columns = [col[0] for col in self._cursor.description]
                    return [
                        dict(zip(columns, row))
                        for row in res
                    ]
            except Exception as data:
                res = False
                self._logger.warn("query database exception, %s" % data)
                os_quit(Err.MYSQL_EXEC_ERR,sql)
        return res


    def update(self, sql):
        flag = False
        if(self._conn):
            try:
                self._cursor.execute(sql)
                self._conn.commit()
                flag = True
            except Exception, data:
                flag = False
                self._logger.warn("update database exception, %s" % data)
                os_quit(Err.MYSQL_EXEC_ERR,sql)
        return flag

    #关闭数据库连接
    def close(self):
        if(self._conn):
            try:
                if(type(self._cursor)=='object'):
                    self._cursor.close()
                if(type(self._conn)=='object'):
                    self._conn.close()
            except Exception, data:
                self._logger.warn("close database exception, %s,%s,%s" % (data, type(self._cursor), type(self._conn)))


def os_quit(MYSQL_CONNECT_ERR, param):
    print
    var = MYSQL_CONNECT_ERR, param
    sys.exit(1)


class EinsteinDB:
     #There are two ways to use this class:
    # 1. Use the class methods to create a new database and table
    # 2. Use the instance methods to connect to an existing database and table

    def __init__(self, dbhost=None, dbport=None, dbuser=None, dbpwd=None, dbname=None):
        self._dbname = dbname
        self._dbhost = dbhost
        self._dbuser = dbuser
        self._dbpassword = dbpwd
        self._dbport = dbport
        self._logger = logging.getLogger(__name__)

        self._conn = self.connectMySQL()
        if(self._conn):
            self._cursor = self._conn.cursor()

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
            os_quit(Err.MYSQL_CONNECT_ERR,"host:%s,port:%s,user:%s" % (self._dbhost,self._dbport,self._dbuser))
            conn = False
        return conn



def EDB():
    return EinsteinDB()

def EDBMySQL():
    return EinsteinMySQLdb()

def EDBDatabase():
    return database()


class EDBConfigError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

class EDBError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

class EDBConnectionError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class EDBQueryError:
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)
