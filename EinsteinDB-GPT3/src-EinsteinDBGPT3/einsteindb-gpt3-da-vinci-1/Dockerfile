FROM ubuntu:16.04
MAINTAINER einstAI "whtcorpsarpanet@gmail.com"

RUN apt-get -y update
RUN apt-get -y install python-pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server

WORKDIR /home/
ADD EinsteinDB-GPT3.tar.gz /home/
RUN pwd
WORKDIR /home/EinsteinDB-GPT3

# RUN DEBIAN_FRONTEND=noninteractive apt upgrade python-pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
