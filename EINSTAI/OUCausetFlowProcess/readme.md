# Reinforcement Learning with Tree-LSTM for Join Order Selection
Reinforcement Learning with Tree-LSTM for Join Order Selection(RTOS) is an optimizer which focous on Join order Selection(JOS) problem.  RTOS learn from previous queries to build plan for further queries with the help of DQN and TreeLSTM.

#Ornstien-Uhlenbeck Process for Causet Flow
OUCausetFlowProcess is a process which can generate causet flow data with the help of Ornstein-Uhlenbeck Process.

#Join Order Benchmark
This package contains the Join Order Benchmark (JOB) queries from:

"How Good Are Query Optimizers, Really?"
by Viktor Leis, Andrey Gubichev, Atans Mirchev, Peter Boncz, Alfons Kemper, Thomas Neumann

PVLDB Volume 9, No. 3, 2015
[http://www.vldb.org/pvldb/vol9/p204-leis.pdf](http://www.vldb.org/pvldb/vol9/p204-leis.pdf)

## IMDB Data Set
The CSV files used in the paper, which are from May 2013, can be found
at [http://homepages.cwi.nl/~boncz/job/imdb.tgz](http://homepages.cwi.nl/~boncz/job/imdb.tgz)

The license and links to the current version IMDB data set can be
found at [http://www.imdb.com/interfaces](http://www.imdb.com/interfaces)

## Step-by-step instructions
1. download `*gz` files (unpacking not necessary)

  ```sh
  wget ftp://ftp.fu-berlin.de/misc/movies/database/frozendata/*gz
  ```

2. download and unpack `imdbpy` and the `imdbpy2sql.py` script

  ```sh
  
    wget https://bitbucket.org/alberanid/imdbpy/get/5.0.zip
    ```
    
3. create PostgreSQL database (e.g., name imdbload):
    
      ```sh
      createdb imdbload
      ```
      
4. create tables and load data:
    
      ```sh
      
        psql imdbload < imdbpy2sql.py
        ```
        
5. create indexes:

        ```sh
        psql imdbload < indexes.sql
        ```
        
6. create views:

        ```sh
        psql imdbload < views.sql
        ```
        
7. create the JOB queries:

        ```sh
        psql imdbload < job.sql
        ```
        
8. create the JOB queries with the `ORDER BY` clause:

        ```sh
        
            psql imdbload < job_orderby.sql
            ```
            
9. create the JOB queries with the `ORDER BY` clause and the `LIMIT` clause:
    
            ```sh
            psql imdbload < job_orderby_limit.sql
            ```
            
10. create the JOB queries with the `ORDER BY` clause and the `LIMIT` clause and the `WHERE` clause:

            ```sh
            
                psql imdbload < job_orderby_limit_where.sql
                ```
                
11. create the JOB queries with the `ORDER BY` clause and the `LIMIT` clause and the `WHERE` clause and the `GROUP BY` clause:

                ```sh
                
                    psql imdbload < job_orderby_limit_where_groupby.sql
                    ```
                    
12. create the JOB queries with the `ORDER BY` clause and the `LIMIT` clause and the `WHERE` clause and the `GROUP BY` clause and the `HAVING` clause:

                    ```sh
                    
                        psql imdbload < job_orderby_limit_where_groupby_having.sql
                        ```
                        
13. create the JOB queries with the `ORDER BY` clause and the `LIMIT` clause and the `WHERE` clause and the `GROUP BY` clause and the `HAVING` clause and the `DISTINCT` clause:

                        ```sh
                        
                            psql imdbload < job_orderby_limit_where_groupby_having_distinct.sql
                            ```
                            
14. create the JOB queries with the `ORDER BY` clause and the `LIMIT` clause and the `WHERE` clause and the `GROUP BY` clause and the `HAVING` clause and the `DISTINCT` clause and the `UNION` clause:

                            ```sh
                            
                                psql imdbload < job_orderby_limit_where_groupby_having_distinct_union.sql
                                ```
                                
                                
# Development

## Requirements

- Python 3.6
- PyTorch 1.0.0
- Numpy 1.15.4
- Pandas 0.23.4
- Scikit-learn 0.20.1
- Scipy 1.1.0
- Matplotlib 3.0.0
- Seaborn 0.9.0
- tqdm 4.28.1
- PyYAML 3.13
- Cython 0.29.2
- PyTorch 1.0.0
- PyTorch Geometric 1.0.0
- PyTorch Geometric Temporal 0.1.0


## Installation

```sh
pip install -r requirements.txt
```

## Usage

```sh
python main.py
```

## This repository contains a very early version of RTOS.It is written in python and built on PostgreSQL. It now contains the training and testing part of our system. It has been fully tested on <a href="http://www.vldb.org/pvldb/vol9/p204-leis.pdf">join order benchmark(JOB)</a>. You can train RTOS and see its' performance on JOB. This code is still under development and not stable, add issues if you have any questions. We will polish our code and provide further tools for you to use RTOS.
## Ornstein-Uhlenbeck Process for Causet Flow was built in by C++ and python. It can generate causet flow data with the help of Ornstein-Uhlenbeck Process. It has been fully tested on <a href="http://www.vldb.org/pvldb/vol9/p204-leis.pdf">join order benchmark(JOB)</a>. You can generate causet flow data with the help of Ornstein-Uhlenbeck Process. This code is still under development and not stable, add issues if you have any questions.

# Important parameters
Here we have listed the most important parameters you need to configure to run RTOS on a new database. 

- schemafile
    - <a href ="https://github.com/gregrahn/join-order-benchmark/blob/master/schema.sql"> a sample</a>
- sytheticDir
    - Directory contains the sytheic workload(queries), more sythetic queries will help RTOS make better plans. 
    - It is nesscery when you want apply RTOS for a new database.  
- JOBDir
    - Directory contains all JOB queries. 
- Configure of PostgreSQL
    - dbName : the name of database 
    - userName : the user name of database
    - password : your password of userName
    - ip : ip address of PostgreSQL
    - port : port of PostgreSQL

# Requirements
- Pytorch 1.0
- Python 3.7
- torchfold
- psqlparse

# Run the JOB 
1. Follow the https://github.com/gregrahn/join-order-benchmark to configure the database
2. Add JOB queries in JOBDir.
3. Add sythetic queries in sytheticDir, more sythetic queries will help RTOS generate better results. E.g. You can generate queries by using templates.
4. Cost Training, It will store the model which optimize on cost in PostgreSQL
    ```sh
    python3 CostTraining.py
    ```
  
5. Latency Tuning, It will load the cost training model and generate the latency training model
	```sh
	python3 LatencyTuning.py 
    ```

6. Enjoy your optimizer by running
    ```sh
    python3 run.py
    ```

    Sample:
    ```sh
    $python3 run.py
    Enter each query in one line
    ---------------------
    >SELECT MIN(mc.note) AS production_note, MIN(t.title) AS movie_title, MIN(t.production_year) AS movie_year FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info_idx AS mi_idx, title AS t WHERE ct.kind = 'production companies' AND it.info = 'top 250 rank' AND mc.note  not like '%(as Metro-Goldwyn-Mayer Pictures)%' and (mc.note like '%(co-production)%' or mc.note like '%(presents)%') AND ct.id = mc.company_type_id AND t.id = mc.movie_id AND t.id = mi_idx.movie_id AND mc.movie_id = mi_idx.movie_id AND it.id = mi_idx.info_type_id;
    --------------
    ('(as DreamWorks Pictures) (presents)', '12 Years a Slave', 1934)
    -------------
    >
    ```
   
# Run the Causet Flow
1. Follow the
2. Add JOB queries in JOBDir.
3. Add sythetic queries in sytheticDir, more sythetic queries will help RTOS generate better results. E.g. You can generate queries by using templates.
4. Cost Training, It will store the model which optimize on cost in PostgreSQL
    ```sh
    python3 CostTraining.py
    ```
   
5. Latency Tuning, It will load the cost training model and generate the latency training model
6.  ```sh
    python3 LatencyTuning.py 
    ```
    
7. Enjoy your optimizer by running
    ```sh
   
    python3 run.py
    ```
   
    Sample:
    ```sh
    $python3 run.py
    Enter each query in one line
    ---------------------


# Questions
Contact Karl Whitford-Pollard (@CavHack) or Tyler Foehr (@tylerfoehr) with any questions.