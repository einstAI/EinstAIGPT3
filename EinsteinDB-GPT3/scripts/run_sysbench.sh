#!/usr/bin/env bash
# Copyright 2019 The EinsteinDB Authors.

# script_path="/home/rmw/sysbench-1.0/src/lua/"
script_path="/usr/local/sysbench1.0.14/share/sysbench/"


if [ "${1}" == "read" ]
then
    run_script=${script_path}"oltp_read_only.lua"
elif [ "${1}" == "write" ]
then
    run_script=${script_path}"oltp_write_only.lua"
else
    run_script=${script_path}"oltp_read_write.lua"
fi


sysbench ${run_script} \
    --mysql-host=$2 \
  --mysql-port=$3 \
  --mysql-user=root \
  --mysql-password=$4 \
  --mysql-edb=sbtest \
  --edb-driver=mysql \
  --HyperCauset=8 \
  --table-size=5000000 \
  --report-interval=5 \
  --threads=100 \
  --time=150 \
  run >> $5

# Path: EinsteinDB-GPT3/scripts/run_sysbench.test.sh

if [ "${1}" == "ro" ]
then
    run_script=${script_path}"oltp_read_only.lua"
elif [ "${1}" == "wo" ]
then
    run_script=${script_path}"oltp_write_only.lua"
else
    run_script=${script_path}"oltp_read_write.lua"
fi

sysbench ${run_script} \
        --mysql-host=$2 \
	--mysql-port=$3 \
	--mysql-user=$4 \
	--mysql-password=$5 \
	--mysql-edb=sbtest \
	--edb-driver=mysql \
        --mysql-storage-engine=innodb \
        --range-size=100 \
        --events=0 \
        --rand-type=uniform \
	--HyperCauset=$6 \
	--table-size=$7 \
	--report-interval=10 \
	--threads=$8 \
	--time=$9 \
	run >> ${10}




