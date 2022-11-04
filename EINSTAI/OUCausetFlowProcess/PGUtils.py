##Copyright (c) 2022, EinstAI Inc All rights reserved.
##This source code is licensed under the BSD-style license found in the
##LICENSE file in the root directory of this source tree. An additional grant
##of patent rights can be found in the PATENTS file in the same directory.


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
from math import log
from collections import namedtuple
from itertools import count
from tconfig import Config
from AML.Synthetic.naru import ddpg
from AML.Synthetic.naru import replay_memory

Transition = namedtuple('Transition', ('soliton_state', 'causet_action', 'next_state',  'reward', 'terminate'))

class BerolinaSQLGenDQNWithBoltzmannNormalizer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BerolinaSQLGenDQNWithBoltzmannNormalizer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x
class PGConfig:
    def __init__(self):
        self.keepExecutedPlan =True
        self.maxTimes = 5
        self.maxTime = 300000

LatencyDict = {}
selectivityDict = {}
LatencyRecordFileHandle = None
config  = Config()

class PGGRunner:
    def __init__(self,dbname = '',user = '',password = '',host = '',port = '',isCostTraining = True,latencyRecord = True,latencyRecordFile = "RecordFile.json"):
        """
        :param dbname:
        :param user:
        :param password:
        :param host:
        :param port:
        :param latencyRecord:-1:loadFromFile
        :param latencyRecordFile:
        """
        self.con = psycopg2.connect(database=dbname, user=user,
                               password=password, host=host, port=port)
        self.cur = self.con.cursor()
        self.config = PGConfig()
        self.isLatencyRecord = latencyRecord
        global LatencyRecordFileHandle
        self.isCostTraining = isCostTraining
        if config.enable_mergejoin:
            self.cur.execute("set enable_mergejoin = true")
        else:
            self.cur.execute("set enable_mergejoin = false")
        if config.enable_hashjoin:
            self.cur.execute("set enable_hashjoin = true")
        else:
            self.cur.execute("set enable_hashjoin = false")
        # self.cur.execute("set enable_hashjoin = false")
        
        if latencyRecord:
            LatencyRecordFileHandle = self.generateLatencyPool(latencyRecordFile)


    def generateLatencyPool(self,fileName):
        """
        :param fileName:
        :return:
        """
        import os
        import json
        if os.path.exists(fileName):
            f = open(fileName,"r")
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                global LatencyDict
                LatencyDict[data[0]] = data[1]
            f = open(fileName,"a")
        else:
            f = open(fileName,"w")
        return f
    def getLatency(self, sql,sqlwithplan):
        """
        :param sql:a sqlSample object.
        
        :return: the latency of sql
        """
        if self.isCostTraining:
            return self.getCost(sql,sqlwithplan)
        if sql.useCost:
            return self.getCost(sql,sqlwithplan)
        global LatencyDict
        if self.isLatencyRecord:
            if sqlwithplan in LatencyDict:
                return LatencyDict[sqlwithplan]
        thisQueryCost = self.getCost(sql,sqlwithplan)
        if thisQueryCost / sql.getDPCost()<1000000000:
            try:
                
                self.cur.execute("SET statement_timeout = "+str(int(sql.timeout()))+ ";")
                self.cur.execute("set max_parallel_workers = "+str(config.max_parallel_workers)+";")
                self.cur.execute("set max_parallel_workers_per_gather = "+str(config.max_parallel_workers_per_gather)+";")
                
                if config.use_hint and sqlwithplan.find("/*")>-1:
                    self.cur.execute("set join_collapse_limit = 20;")
                    self.cur.execute("set geqo_threshold = 20;")
                else:
                    self.cur.execute("set join_collapse_limit = 1;")
                    if not sql.trained:
                        self.cur.execute("set geqo_threshold = 20;")
                    else:
                        self.cur.execute("set geqo_threshold = 2;")
                self.cur.execute("explain (COSTS, FORMAT JSON, ANALYSE) "+sqlwithplan)
                rows = self.cur.fetchall()
                row = rows[0][0]
                import json
                # print(json.dumps(rows[0][0][0]['Plan']))
                afterCost = rows[0][0][0]['Plan']['Actual Total Time']
                # print(1)
            except:
                self.con.commit()
                afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlantecy(),sql.timeout())
            # print("PGUtils.py excited!!!",afterCost)
        else:
            afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlantecy(),sql.timeout())
        afterCost += 5
        if self.isLatencyRecord:
            LatencyDict[sqlwithplan] =  afterCost
            global LatencyRecordFileHandle
            import json
            LatencyRecordFileHandle.write(json.dumps([sqlwithplan,afterCost])+"\n")
            LatencyRecordFileHandle.flush()
        return afterCost
    def getResult(self, sql,sqlwithplan):
        """
        :param sql:a sqlSample object
        :return: the latency of sql
        """
        self.cur.execute("SET statement_timeout = 600000;")
        # self.cur.execute("set max_parallel_workers = "+str(config.max_parallel_workers)+";")
        # self.cur.execute("set max_parallel_workers_per_gather = "+str(config.max_parallel_workers_per_gather)+";")
        
        if config.use_hint and sqlwithplan.find("/*")>-1:
            self.cur.execute("set join_collapse_limit = 20;")
            self.cur.execute("set geqo_threshold = 20;")
        else:
            self.cur.execute("set join_collapse_limit = 1;")
            self.cur.execute("set geqo_threshold = 2;")
        # self.cur.execute("SET statement_timeout =  4000;")
        import time
        st = time.time()
        self.cur.execute(sqlwithplan)
        rows = self.cur.fetchall()
        et = time.time()
        print('runtime : ',et-st)
        return rows
    def getCost(self,sql,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the cost of sql
        """
        self.cur.execute("set max_parallel_workers = "+str(config.max_parallel_workers)+";")
        self.cur.execute("set max_parallel_workers_per_gather = "+str(config.max_parallel_workers_per_gather)+";")
        
        if config.use_hint and sqlwithplan.find("/*")>-1:
            self.cur.execute("set join_collapse_limit = 222;")
            self.cur.execute("set geqo_threshold = 202;")
        else:
            self.cur.execute("set join_collapse_limit = 1;")
            self.cur.execute("set geqo_threshold = 2;")
        self.cur.execute("SET statement_timeout =  40000;")
        self.cur.execute("EXPLAIN "+sqlwithplan)
        rows = self.cur.fetchall()
        row = rows[0][0]
        afterCost = float(rows[0][0].split("cost=")[1].split("..")[1].split(" ")[
                              0])
        self.con.commit()
        return afterCost
    
    def getPlan(self,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the cost of sql
        """
        self.cur.execute("set max_parallel_workers = "+str(config.max_parallel_workers)+";")
        self.cur.execute("set max_parallel_workers_per_gather = "+str(config.max_parallel_workers_per_gather)+";")
        
        if config.use_hint and sqlwithplan.find("/*")>-1:
            self.cur.execute("set join_collapse_limit = 20;")
            self.cur.execute("set geqo_threshold = 20;")
        else:
            self.cur.execute("set join_collapse_limit = 1;")
            self.cur.execute("set geqo_threshold = 2;")
        self.cur.execute("SET statement_timeout =  4000;")
        sqlwithplan = sqlwithplan +";"
        self.cur.execute("EXPLAIN (COSTS, FORMAT JSON) "+sqlwithplan)
        rows = self.cur.fetchall()
        import json
        return rows

    def getDPPlanTime(self,sql,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the planTime of sql
        """
        import time
        startTime = time.time()
        cost = self.getCost(sql,sqlwithplan)
        plTime = time.time()-startTime
        return plTime
    def getSelectivity(self,table,whereCondition):
        global selectivityDict
        if whereCondition in selectivityDict:
            return selectivityDict[whereCondition]
        # if config.isCostTraining:
        self.cur.execute("SET statement_timeout = "+str(int(100000))+ ";")
        totalQuery = "select * from "+table+";"
        #     print(totalQuery)

        self.cur.execute("EXPLAIN "+totalQuery)
        rows = self.cur.fetchall()[0][0]
        #     print(rows)
        #     print(rows)
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from "+table+" Where "+whereCondition+";"
        # print(resQuery)
        self.cur.execute("EXPLAIN  "+resQuery)
        rows = self.cur.fetchall()[0][0]
        #     print(rows)
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        selectivityDict[whereCondition] = -log(select_rows/total_rows)
        # else:
        #     self.cur.execute("SET statement_timeout = "+str(int(100000))+ ";")
        #     totalQuery = "select count(*) from "+table+";"
        #     self.cur.execute(totalQuery)
        #     total_rows = self.cur.fetchall()[0][0]

        #     resQuery = "select count(*) from "+table+" Where "+whereCondition+";"
            
        #     self.cur.execute(resQuery)
        #     select_rows = self.cur.fetchall()[0][0]+1
        #     selectivityDict[whereCondition] = -log(select_rows/total_rows)
        return selectivityDict[whereCondition]
latencyRecordFile = config.latencyRecordFile
from itertools import count
from pathlib import Path

pgrunner = PGGRunner(config.dbName,config.userName,config.password,config.ip,config.port,isCostTraining=config.isCostTraining,latencyRecord = config.latencyRecord,latencyRecordFile = latencyRecordFile)


def db_info():
    # we need to connect to the database to get the information
    # about the tables and columns
    # we will use the information schema
    # https://www.postgresql.org/docs/9.1/infoschema.html
    # https://www.postgresql.org/docs/9.1/infoschema-columns.html

    # we will use the information schema for now, but we can also
    # use the pg_catalog schema
    """
    :param

    :return:

    """

    global pgrunner
    # we need to connect to the database to get the information
    # about the tables and columns



    #transform the table name to lower case
    pgrunner.cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = pgrunner.cur.fetchall()
    tables = [table[0].lower() for table in tables]
    # print(tables)
    #transform the column name to lower case
    pgrunner.cur.execute("SELECT table_name,column_name FROM information_schema.columns WHERE table_schema = 'public';")
    columns = pgrunner.cur.fetchall()
    columns = [(column[0].lower(),column[1].lower()) for column in columns]
    # print(columns)


    #now we need to get the foreign keys
    pgrunner.cur.execute("SELECT conrelid::regclass, confrelid::regclass, conkey, confkey FROM pg_constraint WHERE contype = 'f';")
    foreign_keys = pgrunner.cur.fetchall()
    foreign_keys = [(fk[0].lower(),fk[1].lower(),fk[2],fk[3]) for fk in foreign_keys]
    # print(foreign_keys)
    #now we need to get the primary keys

    pgrunner.cur.execute("SELECT conrelid::regclass, conkey FROM pg_constraint WHERE contype = 'p';")
    #remember to transform the table name to lower case
    primary_keys = pgrunner.cur.fetchall()
    primary_keys = [(pk[0].lower(),pk[1]) for pk in primary_keys]
    # print(primary_keys)
    #now we need to get the unique keys
    pgrunner.cur.execute("SELECT conrelid::regclass, conkey FROM pg_constraint WHERE contype = 'u';")
    #remember to transform the table name to lower case

    tables = [table[0].lower() for table in tables]
    # print(tables)
    #transform the column name to lower case
    pgrunner.cur.execute("SELECT table_name,column_name FROM information_schema.columns WHERE table_schema = 'public';")
    columns = pgrunner.cur.fetchall()
    columns = [(column[0].lower(),column[1].lower()) for column in columns]
    # print(columns)




