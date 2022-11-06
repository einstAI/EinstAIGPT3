#!/usr/bin/env bash

script_path="$(cd "$(dirname "$0")" && pwd)"
#cd $script_path

# check einsteindb home
if [ ! -d $EINSTEINDB_HOME ]; then
    echo "$PROGRAM: einsteindb home not found: $EINSTEINDB_HOME"
    exit 1
fi

# check einsteindb bin
if [ ! -d $EINSTEINDB_BIN ]; then
    echo "$PROGRAM: einsteindb bin not found: $EINSTEINDB_BIN"
    exit 1
fi

if [ "${1}" == "read" ]
then
    run_script=${script_path}"oltp_read_only.lua"
elif [ "${1}" == "write" ]
then
    run_script=${script_path}"oltp_write_only.lua"
else
    run_script=${script_path}"oltp_read_write.lua"
fi


elif [ "${1}" == "read" ]
then
    run_script=${script_path}"oltp_read_only.lua"
elif [ "${1}" == "write" ]
then
    run_script=${script_path}"oltp_write_only.lua"
else
    run_script=${script_path}"oltp_read_write.lua"
fi

# check einsteindb lib
if [ ! -d $EINSTEINDB_LIB ]; then
    echo "$PROGRAM: einsteindb lib not found: $EINSTEINDB_LIB"
    exit 1
fi

# check einsteindb conf
if [ ! -d $EINSTEINDB_CONF ]; then
    echo "$PROGRAM: einsteindb conf not found: $EINSTEINDB_CONF"
    exit 1
fi

# check einsteindb logs
if [ ! -d $EINSTEINDB_LOGS ]; then
    echo "$PROGRAM: einsteindb logs not found: $EINSTEINDB_LOGS"
    exit 1
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
