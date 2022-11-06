#!/bin/sh

PROGRAM=$(basename $0)
EINSTEINDB_HOME=$(cd $(dirname $0)/..; pwd)
EINSTEINDB_BIN=$EINSTEINDB_HOME/bin
EINSTEINDB_LIB=$EINSTEINDB_HOME/lib
EINSTEINDB_CONF=$EINSTEINDB_HOME/conf
EINSTEINDB_LOGS=$EINSTEINDB_HOME/logs
EINSTEINDB_PID=$EINSTEINDB_HOME/einsteindb.pid






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