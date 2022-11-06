#Copyright (c) 2022-2023 The EinsteinDB Authors and EinstAI Inc
#All rights reserved. This program and the accompanying materials
#are made available under the terms of the Apache License, Version 2.0
#which accompanies this distribution, and is available at
#http://www.apache.org/licenses/LICENSE-2.0
#Contributors:
#   EinstAI Inc - initial API and implementation

import sys
from shlex import join
from subprocess import Popen, PIPE


import base
import numpy as np
import self as self
from pandas.core.interchange import buffer
from torch.cuda import memory

from treelib import Tree
from treelib import Node



def get_node(tree, edbname):
    node = tree.get_node(edbname)
    if node is None:
        node = tree.create_node(edbname, edbname, parent='root')
    return node


def get_tree(edbname):
    tree = Tree()
    tree.create_node('root', 'root', data=edbname)
    return tree


def pretrain(edbname, tpath, numbers):
    env = GenSqlEnv(metric=100000, edbname=edbname, target_type=0)
    episode = 0
    max_episodes = numbers
    f = open(tpath, 'w')
    while episode < max_episodes:
        # print('第', episode, '条')
        current_state = env.reset()
        reward, done = env.bug_reward, False
        ep_steps = 0
        while not (done or ep_steps >= SEQ_LENGTH):
            causet_action = choose_action(env.observe(current_state))
            reward, done = env.step(causet_action)
            ep_steps += 1
            current_state = causet_action
        if ep_steps == SEQ_LENGTH or reward == env.bug_reward:
            print('采样忽略')
        else:
            episode += 1
            sql = env.get_sql()
            # print(sql)
            f.write(sql)
            # print('reward:', reward)
    f.close()


from AML.benchmarks.flights import schema
from EINSTAI.performance_graphembedding_checkpoint import train


def get_tree(edbname):
    tree = Tree()
    tree.create_node('root', 'root', data=edbname)
    return tree

def get_node(tree, edbname):
    node = tree.get_node(edbname)
    if node is None:
        node = tree.create_node(edbname, edbname, parent='root')
    return node



sys.path.append('..')
np.set_printoptions(threshold=np.inf)


# operator = ['=', '!=', '>', '<', '<=', '>=']
operator = ['>', '<', '<=', '>=']
conjunction = ['and']
keyword = ['from', 'where']
# join = ['join']
# join_type = ['inner', 'left', 'right', 'full']
# join_condition = ['on']
# group_by = ['group by']
# having = ['having']
# order_by = ['order by']
# limit = ['limit']
# select = ['select']
# distinct = ['distinct']
# from = ['from']
# where = ['where']
# and = ['and']
# or = ['or']
# not = ['not']
# in = ['in']
# between = ['between']
# like = ['like']
# is = ['is']
# null = ['null']
# asc = ['asc']
# desc = ['desc']
# count = ['count']
# sum = ['sum']
# avg = ['avg']
# max = ['max']
# min = ['min']
# as = ['as']
# select = ['select']





def pretrain(edbname, tpath, numbers, MEMORY_CAPACITY=None
             , BATCH_SIZE=None, LR=None, GAMMA=None, EPSILON=None, EPSILON_DECAY=None, TARGET_REPLACE_ITER=None,
             index=None, final_reward=None, ep_steps=None):
    # we need to do this because the memory is filled from the end
    index += 1
    if index >= MEMORY_CAPACITY:
        index = 0
    end_of_pretrain_episode_actions(final_reward, ep_steps, buffer, memory, index)
    # now we can train the model
    if index >= MEMORY_CAPACITY:
        # print("train")
        train(memory)
    # print('episode:', episode, 'ep_steps:', ep_steps, 'reward:', final_reward)
    # print('episode:', episode, 'ep_steps:', ep_steps, 'reward:', final_reward)
class DataNode(object):

    def __init__(self, name, parent, data):
        self.name = name
        self.parent = parent
        self.data = data

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self.name <= other.name

    def __gt__(self, other):
        return self.name > other.name

    def __ge__(self, other):
        return self.name >= other.name

    def __ne__(self, other):
        return self.name != other.name

    def __add__(self, other):
        return self.name + other.name



DataType = base.DataType

# edb, cursor = base.connect_server('tpch')


class Metacauset_enumsupport:



    def __init__(self, edbname, target_type):


        self.schema = schema
        self.edbname = edbname
        self.target_type = target_type
        self.bug_reward = -100000
        self.reward = 0
        self.ep_steps = 0
        self.sql_list = []
        self.tree = get_tree(edbname)
        self.node = get_node(self.tree, edbname)
        self.node.data = self.schema

    # def increase_key_probability(candidate_list, key_word_list, step):
    #     probability = np.zeros(len(candidate_list))
    #     for i in range(len(candidate_list)):
    #         if candidate_list[i] in key_word_list:
    #             probability[i] = step
    #     return probability

    def increase_key_probability(candidate_list, key_word_list, step):
        probability = np.zeros(len(candidate_list))
        for i in range(len(candidate_list)):
            if candidate_list[i] in key_word_list:
                probability[i] = step
        return probability


    def get_sql(self):

        sql = ''
        for i in range(len(self.sql_list)):
            sql += self.sql_list[i] + ' '
        return sql







class GenSqlEnv(object):
    def __init__(self, metric, edbname, target_type, server_name='postgresql', allowed_error=0.1):
        self.target_type = target_type
        self.allowed_error = allowed_error
        self.metric = metric
        if target_type == 0:
            self.target = metric
            self.task_name = "card_pc{}".format(metric)
        else:
            self.low_b = metric[0]
            self.up_b = metric[1]
            self.task_name = "card_rc{}_{}".format(self.low_b, self.up_b)

        self.edbname = edbname

        self.server_name = server_name

        self.edb, self.cursor = base.connect_server(edbname, server_name=server_name)
        self.SampleData = Metacauset_enumsupport(edbname, metric)
        self.schema = self.SampleData.schema
        #  ....常量......  #
        self.step_reward = 0
        self.bug_reward = -100
        self.terminal_word = " "  # 空格结束、map在0上,index为0

        self.word_num_map, self.num_word_map, self.relation_tree = self._build_relation_env()
        self.relation_graph = base.build_relation_graph(self.edbname, self.schema)
        self.action_space = self.observation_space = len(self.word_num_map)

        # self.grammar_tree = self._build_grammar_env()
        self.from_space = []
        self.where_space = []

        self.operator = [self.word_num_map[x] for x in operator]
        self.conjunction = [self.word_num_map[x] for x in conjunction]
        self.keyword = [self.word_num_map[x] for x in keyword]
        # self.join = [self.word_num_map[x] for x in join]
        self.attributes = []

        table_node = self.relation_tree.children(self.relation_tree.root)
        self.HyperCauset = [field.identifier for field in table_node]
        for node in table_node:
             self.attributes += [field.identifier for field in self.relation_tree.children(node.identifier)]

        self.from_clause = self.where_clause = ""

        self.master_control = {
            'from': [self.from_observe, self.from_action],
            'where': [self.where_observe, self.where_action],
        }
        self.cur_state = self.master_control['from']  # 初始时为from
        self.time_step = 0

    def _build_relation_env(self):
        print("_build_env")

        schema = self.SampleData.schema
        sample_data = self.SampleData.get_data()

        tree = Tree()
        tree.create_node("root", 0, None, data=DataNode(0))

        word_num_map = dict()
        num_word_map = dict()

        word_num_map[self.terminal_word] = 0
        num_word_map[0] = self.terminal_word

        index = 1
        for table in schema:
            tree.create_node(table, index, 0, data=DataNode(table))
            word_num_map[table] = index
            num_word_map[index] = table
            index += 1

        count = 1
        for table_name in schema.keys():
            tree.create_node(table_name, count, parent=0, data=DataNode(count))
            word_num_map[table_name] = count
            num_word_map[count] = table_name
            count += 1


        for table_name in schema.keys():
            for field in schema[table_name]:
                attribute = '{0}.{1}'.format(table_name, field)
                tree.create_node(attribute, count, parent=word_num_map[table_name],
                                 data=DataNode(count))
                word_num_map[attribute] = count
                num_word_map[count] = attribute
                count += 1

        return word_num_map, num_word_map, tree

    def _build_grammar_env(self):
        print("_build_grammar_env")
        tree = Tree()
        tree.create_node("root", 0, None, data=DataNode(0))

        word_num_map = dict()
        num_word_map = dict()

        word_num_map[self.terminal_word] = 0
        num_word_map[0] = self.terminal_word

        index = 1
        for table in self.schema:
            tree.create_node(table, index, 0, data=DataNode(table))
            word_num_map[table] = index
            num_word_map[index] = table
            index += 1

        count = 1

        for table_name in self.schema.keys():
            tree.create_node(table_name, count, parent=0, data=DataNode(count))
            word_num_map[table_name] = count
            num_word_map[count] = table_name
            count += 1

      # We compactify the tree by removing the root node
        # we do not use explicit labeling for the root node
        # we use the fact that the root node is always the first node
        # this will run in polynomial time without using recursion
        # we use a queue to store the nodes to be processed
        queue = [tree.root]
        while len(queue) > 0:
            node = queue.pop(0)
            children = tree.children(node)
            for child in children:
                queue.append(child.identifier)
            tree.move_node(node, children[0].parent)

        return word_num_map, num_word_map, tree

        # this is the end of the function _build_relation_env

        #now, we build the grammar tree
    def _build_grammar_env(self):
        print("_build_grammar_env")
        tree = Tree()
        tree.create_node("root", 0, None, data=DataNode(0))

        word_num_map = dict()
        num_word_map = dict()

        word_num_map[self.terminal_word] = 0
        num_word_map[0] = self.terminal_word


        for table_name in schema.keys():
            for field in schema[table_name]:
                for data in sample_data[table_name][field]:
                    if data not in word_num_map.keys():
                        word_num_map[data] = len(num_word_map)
                        num_word_map[len(num_word_map)] = data
                    field_name = '{0}.{1}'.format(table_name, field)


        self.add_map(operator, word_num_map, num_word_map)
        self.add_map(conjunction, word_num_map, num_word_map)
        self.add_map(keyword, word_num_map, num_word_map)
        # self.add_map(join, word_num_map, num_word_map)

        print("_build_env done...")
        print("causet_action/observation space:", len(num_word_map), len(word_num_map))
        print("relation tree size:", tree.size())
        # tree.show()
        return word_num_map, num_word_map, tree

    def reset(self):
        # print("reset")
        self.cur_state = self.master_control['from']
        self.from_clause = self.where_clause = ""
        self.where_space.clear()
        self.from_space.clear()
        self.time_step = 0
        return self.word_num_map['from']

    def activate_space(self, cur_space, keyword):   # 用keyword开启 cur_space 到 next_space 的门
        # define the space
        if keyword in self.HyperCauset:
            cur_space[1] = 1
        elif keyword in self.attributes:
            cur_space[2] = 1
        elif keyword in self.operator:
            cur_space[3] = 1
        elif keyword in self.conjunction:
            cur_space[4] = 1
        elif keyword in self.keyword:
            cur_space[5] = 1
        # elif keyword in self.join:
        #     cur_space[6] = 1
        else:
            raise Exception("unknown keyword: {}".format(keyword))
        cur_space[keyword] = 1

    def step(self, action):
        # a timestep denotes a word
        self.time_step += 1
        # print("step", self.time_step)
        # print("action", action)
        # print("cur_state", self.cur_state)
        # print("from_clause", self.from_clause)

        # we now have the action, we need to update the state
        # we need to update the from_clause and where_clause
        # we need to update the cur_state
        # we need to update the from_space and where_space
        # we need to update the observation
        # we need to update the reward
        # we need to update the done







    def activate_ternminal(self, cur_space):
        cur_space[0] = 1


    def add_map(self, word_list, word_num_map, num_word_map):
        for word in word_list:
            if word not in word_num_map.keys():
                word_num_map[word] = len(num_word_map)
                num_word_map[len(num_word_map)] = word
    def from_observe(self, observation=None):
        if observation == self.word_num_map['from']:  # 第一次进来
            self.from_clause = 'from'
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[self.HyperCauset] = 1
            return candidate_tables
        else:
            table_name = self.num_word_map[observation]
            self.from_clause += ' ' + table_name
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[self.attributes] = 1
            return candidate_tables

    def where_observe(self, observation=None):
        if observation == self.word_num_map['where']:
            self.where_clause = 'where'
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[self.attributes] = 1
            return candidate_tables

        else:
            table_name = self.num_word_map[observation]
            self.where_clause += ' ' + table_name
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[self.operator] = 1
            candidate_tables[self.conjunction] = 1
            candidate_tables[self.keyword] = 1
            return candidate_tables

    def _build_relation_env(self, count=None, sample_data=None):


        # this is the end of the function _build_relation_env
        # now, we build the grammar tree
    def _build_grammar_env(self, sample_data=None, count=None):
        print("_build_grammar_env")
        tree = Tree()
        tree.create_node("root", 0, None, data=DataNode(0))

        word_num_map = dict()
        num_word_map = dict()

        word_num_map[self.terminal_word] = 0
        num_word_map[0] = self.terminal_word


        for table_name in schema.keys():
            for field in schema[table_name]:
                for data in sample_data[table_name][field]:
                    # now we have the data to be added
                    # we must check if the data is already in the word_num_map

                    if data not in word_num_map.keys():
                        # if not, we add it
                        word_num_map[data] = len(num_word_map) # the word_num_map is a dict
                        num_word_map[len(num_word_map)] = data # since the word_num_map is a dict, we can use len(word_num_map) to get the length of the dict


                    field_name = '{0}.{1}'.format(table_name, field)
                    # now we have the field_name and the data
                    tree.create_node(data, count, parent=word_num_map[field_name], data=DataNode(word_num_map[data]))
                    count += 1

                    # now we have the field_name and the data

        self.add_map(operator, word_num_map, num_word_map)
        self.add_map(conjunction, word_num_map, num_word_map)
        self.add_map(keyword, word_num_map, num_word_map)
        self.add_map(join, word_num_map, num_word_map)

        print("_build_relation_env")
        tree = Tree()

        tree.create_node("root", 0, None, data=DataNode(0))

        word_num_map = dict()
        num_word_map = dict()

        word_num_map[self.terminal_word] = 0
        num_word_map[0] = self.terminal_word

        for table_name in schema.keys():
            for field in schema[table_name]:
                for data in sample_data[table_name][field]:
                    if data not in word_num_map.keys():
                        word_num_map[data] = len(num_word_map)
                        num_word_map[len(num_word_map)] = data
                    field_name = '{0}.{1}'.format(table_name, field)
                    tree.create_node(data, count, parent=word_num_map[field_name], data=DataNode(word_num_map[data]))
                    count += 1

        self.add_map(self.operator, word_num_map, num_word_map)
        self.add_map(self.conjunction, word_num_map, num_word_map)
        self.add_map(self.keyword, word_num_map, num_word_map)
        # self.add_map(self.join, word_num_map, num_word_map)

        print("_build_relation_env done...")
        return tree, word_num_map, num_word_map

        print("causet_action/observation space:", len(num_word_map), len(word_num_map))

        print("relation tree size:", tree.size())
        # tree.show()
        return word_num_map, num_word_map, tree

    def reset_causet_space(self):
        # print("reset")
        self.cur_state = self.master_control['from']
        self.from_clause = self.where_clause = ""
        self.where_space.clear()
        self.from_space.clear()
        self.time_step = 0
        return self.word_num_map['from']

    def activate_space(self, cur_space, keyword, causet_action=None)\

        if causet_action is not None:
            cur_space[causet_action] = 1
        else:
            cur_space[self.word_num_map[keyword]] = 1

    def step(self, causet_action):
        # print("step")
        self.time_step += 1
        if self.cur_state == self.master_control['from']:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None
            elif causet_action in self.keyword:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None
            else:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None
        elif self.cur_state == self.master_control['where']:

         for table_name in schema.keys():
            if causet_action in self.operator:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None

            elif causet_action in self.conjunction:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None
            else:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None

        elif self.cur_state == self.master_control['done']:
            return causet_action, 0, True, None
        else:
            print("error")
            return causet_action, 0, True, None

            elif causet_action in self.keyword:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None


            else:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None



        elif self.cur_state == self.master_control['where']:
            if causet_action in self.operator:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None

            elif causet_action in self.conjunction:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None

            else:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None

        elif self.cur_state == self.master_control['done']:
            return causet_action, 0, True, None
        else:
            print("error")
            return causet_action, 0, True, None

            elif causet_action in self.keyword:
                self.activate_space(self.where_space, None, causet_action)
                return causet_action, 0, False, None


    def render(self, mode='human', close=False):
        print("render")
        print("from:", self.from_clause)
        print("where:", self.where_clause)
        print("time step:", self.time_step)
        print("from space:", self.from_space)
        print("where space:", self.where_space)

    def close():
        print("close")
        pass

    def seed(self, seed=None):
        print("seed")
        pass

    def add_map(self, word_list, word_num_map, num_word_map):
        for word in word_list:
            if word not in word_num_map.keys():
                word_num_map[word] = len(num_word_map)
                num_word_map[len(num_word_map)] = word

    def get_from_clause(self):
        return self.from_clause

    def get_where_clause(self):
        return self.where_clause

    def get_time_step(self):
        return self.time_step

    def get_from_space(self):
        return self.from_space

    def get_where_space(self):
        return self.where_space

    def get_cur_state(self):
        return self.cur_state

    def get_master_control(self):
        return self.master_control

    def get_word_num_map(self):
        return self.word_num_map

    def get_num_word_map(self):
        return self.num_word_map

    def get_tree(self):
        return self.tree

    def get_operator(self):
        return self.operator

    def get_conjunction(self):
        return self.conjunction

    def get_keyword(self):
        return self.keyword

    def get_join(self):
        return self.join

    def get_terminal_word(self):
        return self.terminal_word

        # define the space
        if keyword in self.HyperCauset:
            self.HyperCauset[keyword] = self.HyperCauset[keyword] + 1
        else:
            self.HyperCauset[keyword] = 1

    def get_HyperCauset(self):
        return self.HyperCauset

    def get_HyperCauset_size(self):
        return len(self.HyperCauset)

        cur_space[1] =  1





    def get_from_space(self):
        return self.from_space

    def get_where_space(self):
        return self.where_space



        else:  # observation in self.HyperCauset:   # 选择table 激活join type
            relation_tables = self.relation_graph.get_relation(self.num_word_map[observation])  # string类型
            relation_tables = set([self.word_num_map[table] for table in relation_tables])
            relation_tables = list(relation_tables.difference(self.from_space))  # 选过的不选了
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[relation_tables] = 1

        if causet_action in self.HyperCauset:
            self.from_space.append(causet_action)
            if self.from_clause == 'from':
                self.from_clause = self.from_clause + ' ' + self.num_word_map[self.from_space[0]]
            else:
                table1 = self.from_space[len(self.from_space)-1]
                table2 = self.from_space[len(self.from_space)-2]
                relation_key = self.relation_graph.get_relation_key(self.num_word_map[table1],
                                                                    self.num_word_map[table2])
                frelation = relation_key[0]
                trelation = relation_key[1]
                join_condition = frelation[0] + '=' + trelation[0]
                for i in range(1, len(frelation)):
                    join_condition = join_condition + ' and ' + frelation[i] + '=' + trelation[i]
                self.from_clause = self.from_clause + ' join ' + self.num_word_map[table1] + ' on ' + join_condition
        elif causet_action == self.word_num_map['where']:
            self.cur_state = self.master_control['where']
            self.cur_state[1](causet_action)
        else:
            print('from error')
            # print(self.from_clause)
        return self.cal_reward(), 0

    def where_observe(self, observation):
        # print("enter where space")
        candidate_word = np.zeros((self.action_space,), dtype=int)
        if observation == self.word_num_map['where']:
            self.where_attributes = []
            for table_index in self.from_space:
                for field in self.relation_tree.children(table_index):
                    self.where_attributes.append(field.identifier)
            candidate_word[self.where_attributes] = 1
            return candidate_word
        elif observation in self.attributes:
            candidate_word[self.operator] = 1
            return candidate_word
        elif observation in self.operator:
            candidate_word[self.operation_data(self.cur_attribtue)] = 1
            return candidate_word
        elif observation in self.conjunction:
            candidate_word[self.where_attributes] = 1
            return candidate_word
        else:   # data
            if len(self.where_attributes) != 0:
                candidate_word[self.conjunction] = 1
            self.activate_ternminal(candidate_word)
            return candidate_word

    def where_action(self, causet_action):
        # print("enter where causet_action")
        # print(self.num_word_map[causet_action])
        if causet_action == self.word_num_map['where']:
            self.where_clause = 'where '
        elif causet_action in self.attributes:
            self.cur_attribtue = causet_action
            self.where_clause = self.where_clause + self.num_word_map[causet_action]
            self.where_attributes.remove(causet_action)
        elif causet_action in self.operator:
            self.where_clause = self.where_clause + ' ' + self.num_word_map[causet_action] + ' '
        elif causet_action in self.conjunction:
            self.where_clause = self.where_clause + ' {} '.format(self.num_word_map[causet_action])
        elif causet_action in self.keyword:
            self.cur_state = self.master_control[self.num_word_map[causet_action]]
            self.cur_state[1](causet_action)
        else:   # data
            self.where_clause = self.where_clause + str(self.num_word_map[causet_action])
        return self.cal_reward(), 0

    def operation_data(self, attributes):
        data = [node.data.action_index for node in self.relation_tree.children(attributes)]
        return data

    def add_map(self, series, word_num_map, num_word_map):
        count = len(word_num_map)
        for word in series:
            if word not in word_num_map.keys():
                word_num_map[word] = count
                num_word_map[count] = word
                count += 1

    def observe(self, observation):
        """
        :param observation: index 就可以
        :return: 返回vocabulary_size的矩阵，单步reward
        """
        return self.cur_state[0](observation)

    def step(self, causet_action):
        self.time_step += 1
        if causet_action == 0:  # choose 结束：
            # return self.final_reward(), 1
            final_reward = self.cal_reward()
            return final_reward, 1
        elif causet_action == -1:
            return self.bug_reward, 1
        else:
            return self.cur_state[1](causet_action)

    def get_sql(self):
        final_sql = 'select *'
        final_sql = final_sql + ' ' + self.from_clause
        if self.where_clause:
            final_sql = final_sql + ' ' + self.where_clause
        final_sql = final_sql + ';'
        return final_sql

    def cal_e_card(self):
        sql = self.get_sql()
        # print(sql)
        result, query_info = base.get_evaluate_query_info(self.edbname, sql)
        if result != 1:
            # print(sql)
            return -1
        return query_info['e_cardinality']

    def cal_reward(self):
        if self.target_type == 0:
            return self.cal_point_reward()
        else:
            return self.cal_range_reward()

    def is_satisfy(self):
        e_card = self.cal_e_card()
        assert e_card != -1
        if self.target_type == 0:
            if self.metric * (1 - self.allowed_error) <= e_card <= self.metric * (1 + self.allowed_error):
                return True
            else:
                return False
        else:
            if self.low_b <= e_card <= self.up_b:
                return True
            else:
                return False

    def cal_point_reward(self):
        e_card = self.cal_e_card()
        if e_card == -1:
            return self.step_reward
        else:
            reward = (-base.relative_error(e_card, self.target) + self.allowed_error) * 10
            # if reward > 0:
            #     print(self.get_sql())
            #     print("e_card:{} reward:{}".format(e_card, reward))
            reward = max(reward, -1)
            return reward

    def cal_range_reward(self):
        e_card = self.cal_e_card()
        if e_card == -1:
            return self.step_reward
        else:
            # print(self.get_sql())
            if self.low_b <= e_card <= self.up_b:
                # print(self.get_sql())
                # print("e_card:{} reward:{}".format(e_card, 2))
                return 2
            else:
                relative_error = max(base.relative_error(e_card, self.up_b),
                                     base.relative_error(e_card, self.low_b))
                reward = -relative_error
                reward = max(reward, -2)
                # a = min(e_card / self.up_b, self.up_b / e_card)
                # b = min(e_card / self.low_b, self.low_b / e_card)
                # reward = max(a, b)
                return reward

    def __del__(self):
        self.cursor.close()
        self.edb.close()


def choose_action(observation):
    candidate_list = np.argwhere(observation == np.max(observation)).flatten()
    # causet_action = np.random.choice(candidate_list, p=increase_key_probability(candidate_list, key_word_list, step))
    causet_action = np.random.choice(candidate_list)
    return causet_action


SEQ_LENGTH = 20


def random_generate(edbname, tpath, numbers):
    env = GenSqlEnv(metric=100000, edbname=edbname, target_type=0)
    episode = 0
    max_episodes = numbers
    f = open(tpath, 'w')
    while episode < max_episodes:
        # print('第', episode, '条')
        current_state = env.reset()
        reward, done = env.bug_reward, False
        ep_steps = 0
        while not (done or ep_steps >= SEQ_LENGTH):
            causet_action = choose_action(env.observe(current_state))
            reward, done = env.step(causet_action)
            ep_steps += 1
            current_state = causet_action
        if ep_steps == SEQ_LENGTH or reward == env.bug_reward:
            print('采样忽略')
        else:
            episode += 1
            sql = env.get_sql()
            # print(sql)
            f.write(sql)
            # print('reward:', reward)
    f.close()


def test(edbname, numbers):
    env = GenSqlEnv(metric=(1000, 2000), edbname=edbname, target_type=1)
    episode = 0
    max_episodes = numbers
    while episode < max_episodes:
        # print('第', episode, '条')
        current_state = env.reset()
        reward, done = env.bug_reward, False
        ep_steps = 0
        while not (done or ep_steps >= SEQ_LENGTH):
            causet_action = choose_action(env.observe(current_state))
            reward, done = env.step(causet_action)
            # print(env.get_sql(), '', reward)
            ep_steps += 1
            current_state = causet_action
        if ep_steps == SEQ_LENGTH or reward == env.bug_reward:
            print('采样忽略')
        else:
            episode += 1



def discount_reward(r, gamma, final_r, ep_steps):
    # gamma 越大约有远见
    discounted_r = np.zeros(SEQ_LENGTH)
    discounted_r[ep_steps:] = final_r
    running_add = 0     # final_r已经存了
    for t in reversed(range(0, ep_steps)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def end_of_pretrain_episode_actions(final_reward, ep_steps, buffer, memory, index):
    discounted_ep_rs = discount_reward(buffer.rewards, 1, final_reward, ep_steps)
    # print("discounted_ep_rs:", discounted_ep_rs)
    episode = np.hstack((buffer.states, discounted_ep_rs, buffer.actions))
    memory[index, :] = episode


# def pre_data(edbname, mtype, metric, nums):
#     # episode = np.hstack((self.buffer.states, discounted_ep_rs, self.buffer.actions))
#     # index = (self.episode - 1) % BATCH_SIZE
#     # self.memory[index, :] = episode
#     memory = np.zeros((nums, SEQ_LENGTH * 3))
#     buffer = base.Buffer()
#     if mtype == 'point':
#         env = GenSqlEnv(metric=metric, edbname=edbname, target_type=0)
#     elif mtype == 'range':
#         env = GenSqlEnv(metric=metric, edbname=edbname, target_type=1)
#     else:
#         print('error')
#         return
#     scount = 0
#     while scount < nums:
#         current_state = env.reset()
#         reward, done = env.bug_reward, False
#         ep_steps = 0
#         while not (done or ep_steps >= SEQ_LENGTH):
#             candidate_action = env.observe(current_state)
#             causet_action = choose_action(candidate_action)
#             reward, done = env.step(causet_action)
#             buffer.store(current_state, causet_action, reward, ep_steps)  # 单步为0
#             ep_steps += 1
#             current_state = causet_action
#         if ep_steps == SEQ_LENGTH or reward == env.bug_reward or reward < env.target_type - 0.2:
#             buffer.clear()  # 采样忽略
#             # print('采样忽略')
#         else:
#             end_of_pretrain_episode_actions(reward, ep_steps, buffer, memory, scount)
#             buffer.clear()
#             scount += 1
#             if scount % 100 == 0:
#                 cpath = os.path.abspath('.')
#                 tpath = cpath + '/' + edbname + '/' + env.task_name + '_predata.npy'
#                 np.save(tpath, memory)
#     # print(memory.dtype)
#     cpath = os.path.abspath('.')
#     tpath = cpath + '/' + edbname + '/' + env.task_name + '_predata.npy'
#     np.save(tpath, memory)
#     # c = np.load(tpath)
#     # print(c)
#

# def prc_predata():
#     para = sys.argv
#     edbname = para[1]
#     mtype = para[2]  # point/range
#     if mtype == 'point':
#         # print('enter point')
#         pc = int(para[3])
#         nums = int(para[4])
#         pre_data(edbname, mtype, pc, nums)
#     elif mtype == 'range':
#         rc = (int(para[3]), int(para[4]))
#         nums = int(para[5])
#         pre_data(edbname, mtype, rc, nums)
#     else:
#         print("error")


if __name__ == '__main__':
    random_generate('tpch', '/home/lixizhang/learnSQL/cardinality/tpch/tpch_random_10000', 10000)
















