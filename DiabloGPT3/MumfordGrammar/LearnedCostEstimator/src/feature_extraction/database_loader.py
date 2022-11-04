import pandas as pd
import numpy as np
import random
import time
import torch
from math import log
from torch.nn import init

from DQN import DQN, ENV
from ImportantConfig import Config
from JOBParser import DB
from PGUtils import PGRunner
from TreeLSTM import SPINN
from sqlSample import sqlInfo


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db_info = DB("data/DB/", "data/DB/", "data/DB/")
    pgrunner = PGRunner(db_info)
    QueryDir = "data/SQL/"
    sql_list = QueryLoader(QueryDir)
    train_list, val_list = k_fold(sql_list, 5, 0)
    DQN = DQN(device)
    DQN.load_model()
    for i_episode in range(100):
        print("Epoch:", i_episode)
        train_list = resample_sql(train_list)
        for sql in train_list:
            #         sql = val_list[i_episode%len(train_list)]
            pg_cost = sql.getDPlantecy()
            #         continue
            env = ENV(sql, db_info, pgrunner, device)

            for t in count():
                action_list, chosen_action, all_action = DQN.select_action(env, need_random=False)

                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left, right)

                reward, done = env.reward()
                if done:
                    mrc = max(np.exp(reward * log(1.5)) / pg_cost - 1, 0)
                    DQN.update_policy(env, mrc)
                    break
        DQN.save_model()
        DQN.update_target()
        print("Epoch:", i_episode, "Done")

    DQN.save_model()

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(config.schemaFile, "r") as f:
    createSchema = "".join(f.readlines())

db_info = DB(createSchema)



def load_dataset(dir_path):
    data = dict()
    data["aka_name"] = pd.read_csv(dir_path + '/aka_name.csv', header=None)
    data["aka_title"] = pd.read_csv(dir_path + '/aka_title.csv', header=None)
    data["cast_info"] = pd.read_csv(dir_path + '/cast_info.csv', header=None)
    data["char_name"] = pd.read_csv(dir_path + '/char_name.csv', header=None)
    data["company_name"] = pd.read_csv(dir_path + '/company_name.csv', header=None)
    data["company_type"] = pd.read_csv(dir_path + '/company_type.csv', header=None)
    data["comp_cast_type"] = pd.read_csv(dir_path + '/comp_cast_type.csv', header=None)
    data["complete_cast"] = pd.read_csv(dir_path + '/complete_cast.csv', header=None)
    data["info_type"] = pd.read_csv(dir_path + '/info_type.csv', header=None)
    data["keyword"] = pd.read_csv(dir_path + '/keyword.csv', header=None)
    data["kind_type"] = pd.read_csv(dir_path + '/kind_type.csv', header=None)
    data["link_type"] = pd.read_csv(dir_path + '/link_type.csv', header=None)
    data["movie_companies"] = pd.read_csv(dir_path + '/movie_companies.csv', header=None)
    data["movie_info"] = pd.read_csv(dir_path + '/movie_info.csv', header=None)
    data["movie_info_idx"] = pd.read_csv(dir_path + '/movie_info_idx.csv', header=None)
    data["movie_keyword"] = pd.read_csv(dir_path + '/movie_keyword.csv', header=None)
    data["movie_link"] = pd.read_csv(dir_path + '/movie_link.csv', header=None)
    data["name"] = pd.read_csv(dir_path + '/name.csv', header=None)
    data["person_info"] = pd.read_csv(dir_path + '/person_info.csv', header=None)
    data["role_type"] = pd.read_csv(dir_path + '/role_type.csv', header=None)
    data["title"] = pd.read_csv(dir_path + '/title.csv', header=None)

    aka_name_column = {
        'id': 0,
        'person_id': 1,
        'name': 2,
        'imdb_index': 3,
        'name_pcode_cf': 4,
        'name_pcode_nf': 5,
        'surname_pcode': 6,
        'md5sum': 7
    }

    aka_title_column = {
        'id': 0,
        'movie_id': 1,
        'title': 2,
        'imdb_index': 3,
        'kind_id': 4,
        'production_year': 5,
        'phonetic_code': 6,
        'episode_of_id': 7,
        'season_nr': 8,
        'episode_nr': 9,
        'note': 10,
        'md5sum': 11
    }

    cast_info_column = {
        'id': 0,
        'person_id': 1,
        'movie_id': 2,
        'person_role_id': 3,
        'note': 4,
        'nr_order': 5,
        'role_id': 6
    }

    char_name_column = {
        'id': 0,
        'name': 1,
        'imdb_index': 2,
        'imdb_id': 3,
        'name_pcode_nf': 4,
        'surname_pcode': 5,
        'md5sum': 6
    }

    comp_cast_type_column = {
        'id': 0,
        'kind': 1
    }

    company_name_column = {
        'id': 0,
        'name': 1,
        'country_code': 2,
        'imdb_id': 3,
        'name_pcode_nf': 4,
        'name_pcode_sf': 5,
        'md5sum': 6
    }

    company_type_column = {
        'id': 0,
        'kind': 1
    }

    complete_cast_column = {
        'id': 0,
        'movie_id': 1,
        'subject_id': 2,
        'status_id': 3
    }

    info_type_column = {
        'id': 0,
        'info': 1
    }

    keyword_column = {
        'id': 0,
        'keyword': 1,
        'phonetic_code': 2
    }

    kind_type_column = {
        'id': 0,
        'kind': 1
    }

    link_type_column = {
        'id': 0,
        'link': 1
    }

    movie_companies_column = {
        'id': 0,
        'movie_id': 1,
        'company_id': 2,
        'company_type_id': 3,
        'note': 4
    }

    movie_info_idx_column = {
        'id': 0,
        'movie_id': 1,
        'info_type_id': 2,
        'info': 3,
        'note': 4
    }

    movie_keyword_column = {
        'id': 0,
        'movie_id': 1,
        'keyword_id': 2
    }

    movie_link_column = {
        'id': 0,
        'movie_id': 1,
        'linked_movie_id': 2,
        'link_type_id': 3
    }

    name_column = {
        'id': 0,
        'name': 1,
        'imdb_index': 2,
        'imdb_id': 3,
        'gender': 4,
        'name_pcode_cf': 5,
        'name_pcode_nf': 6,
        'surname_pcode': 7,
        'md5sum': 8
    }

    role_type_column = {
        'id': 0,
        'role': 1
    }

    title_column = {
        'id': 0,
        'title': 1,
        'imdb_index': 2,
        'kind_id': 3,
        'production_year': 4,
        'imdb_id': 5,
        'phonetic_code': 6,
        'episode_of_id': 7,
        'season_nr': 8,
        'episode_nr': 9,
        'series_years': 10,
        'md5sum': 11
    }

    movie_info_column = {
        'id': 0,
        'movie_id': 1,
        'info_type_id': 2,
        'info': 3,
        'note': 4
    }

    person_info_column = {
        'id': 0,
        'person_id': 1,
        'info_type_id': 2,
        'info': 3,
        'note': 4
    }
    data["aka_name"].columns = aka_name_column
    data["aka_title"].columns = aka_title_column
    data["cast_info"].columns = cast_info_column
    data["char_name"].columns = char_name_column
    data["company_name"].columns = company_name_column
    data["company_type"].columns = company_type_column
    data["comp_cast_type"].columns = comp_cast_type_column
    data["complete_cast"].columns = complete_cast_column
    data["info_type"].columns = info_type_column
    data["keyword"].columns = keyword_column
    data["kind_type"].columns = kind_type_column
    data["link_type"].columns = link_type_column
    data["movie_companies"].columns = movie_companies_column
    data["movie_info"].columns = movie_info_column
    data["movie_info_idx"].columns = movie_info_idx_column
    data["movie_keyword"].columns = movie_keyword_column
    data["movie_link"].columns = movie_link_column
    data["name"].columns = name_column
    data["person_info"].columns = person_info_column
    data["role_type"].columns = role_type_column
    data["title"].columns = title_column
    return data
