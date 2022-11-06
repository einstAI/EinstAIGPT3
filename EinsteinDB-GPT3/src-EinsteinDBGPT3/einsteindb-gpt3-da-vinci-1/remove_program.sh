#!/bin/sh

# Remove the program
rm -rf /usr/local/bin/einsteindb-gpt3-da-vinci-1

# Remove the data
rm -rf /usr/local/share/einsteindb-gpt3-da-vinci-1

# Remove the configuration
rm -rf /usr/local/etc/einsteindb-gpt3-da-vinci-1

# Remove the log
rm -rf /usr/local/var/log/einsteindb-gpt3-da-vinci-1

# Remove the user
userdel einsteindb-gpt3-da-vinci-1

# Remove the group
groupdel einsteindb-gpt3-da-vinci-1

# Remove the service
rm -rf /etc/systemd/system/einsteindb-gpt3-da-vinci-1.service

# Remove the logrotate
rm -rf /etc/logrotate.d/einsteindb-gpt3-da-vinci-1




if [ $# -ne 1 ];then
    echo "Usage: `basename $0` <program>"
    exit 1
fi
program=$(echo $1 | tr '[:upper:]' '[:lower:]')

echo "remove $program"

rm -f $program package/$program -R

echo "Done."



