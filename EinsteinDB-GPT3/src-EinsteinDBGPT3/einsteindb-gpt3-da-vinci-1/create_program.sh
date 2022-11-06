#!/bin/bash
#


#
# This script creates a program.
#


# Path: EinsteinDB-GPT3/src-EinsteinDBGPT3/einsteindb-gpt3-da-vinci-1/create_program.sh


function red_c()
{
    echo -e $2 "\e[31;1m${1}\e[0m"
}

function cyan_c()
{
    echo -e $2 "\e[36;1m${1}\e[0m"
}

if [ $# -ne 1 ];then
    red_c "Usage: `basename $0` <program>"
    exit 1
fi
app_name=$1
program=$(echo $app_name | tr '[:upper:]' '[:lower:]')
pkg_program=package/$program
basedir=../base/

echo "app_name: $app_name, program dir: $program"

if [ -d $program ] || [ -d $pkg_program ];then
    red_c "program $program or $pkg_program is already exist, remove the dir if continue"
    exit 2
fi

cyan_c "mkdir program dir..."
mkdir $program
mkdir -p $pkg_program
mkdir -p $pkg_program/log
mkdir -p $pkg_program/tools

cyan_c "copy code & scripts..."
cp $basedir/tools/code/* $program/
cp $basedir/tools/cron  $pkg_program/tools/ -R
cp $basedir/tools/op    $pkg_program/tools/ -R
cp $basedir/tools/bin   $pkg_program/       -R
cp $basedir/tools/etc   $pkg_program/       -R
mv $pkg_program/etc/dapp.conf $pkg_program/etc/${program}.conf
mv $pkg_program/etc/dapp.yml  $pkg_program/etc/${program}.yml

cyan_c "copy systemd service..."
cp $basedir/tools/systemd/* $pkg_program/tools/ -R
mv $pkg_program/tools/systemd/dapp.service $pkg_program/tools/systemd/${program}.service


cyan_c "copy logrotate..."
cp $basedir/tools/logrotate/* $pkg_program/tools/ -R
mv $pkg_program/tools/logrotate/dapp $pkg_program/tools/logrotate/${program}

cyan_c "copy README.md..."
cp $basedir/README.md $pkg_program/

cyan_c "copy LICENSE..."
cp $basedir/LICENSE $pkg_program/

cyan_c "copy CHANGELOG.md..."
cp $basedir/CHANGELOG.md $pkg_program/


cyan_c "copy package.json..."
cp $basedir/package.json $pkg_program/

cyan_c "copy package-lock.json..."
cp $basedir/package-lock.json $pkg_program/

cyan_c "copy .gitignore..."
cp $basedir/.gitignore $pkg_program/

cyan_c "copy .npmignore..."
cp $basedir/.npmignore $pkg_program/


cyan_c "generate code..."
cp $basedir/tools/code/* $program/
sed -i 's/<PROGRAM>/'$app_name'/' $program/dapp.go
sed -i 's/<PROGRAM>/'$program'/' $program/main.go
sed -i 's/<PROGRAM>/'$program'/' $pkg_program/etc/${program}.conf
sed -i 's/<PROGRAM>/'$program'/' $pkg_program/etc/${program}.yml

cyan_c "Done."




