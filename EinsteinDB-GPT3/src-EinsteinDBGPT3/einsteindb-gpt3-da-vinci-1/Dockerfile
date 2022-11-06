FROM ubuntu:16.04
MAINTAINER einstAI "whtcorpsarpanet@gmail.com"

# Install dependencies




RUN apt-get -y update
RUN apt-get -y install python-pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]

# Path: EinsteinDB-GPT3/src-EinsteinDBGPT3/einsteindb-gpt3-da-vinci-1/app.py
ADD EinsteinDB-GPT3.tar.gz /home/
RUN pwd
WORKDIR /home/EinsteinDB-GPT3

# Install Python dependencies
RUN pip install -r requirements.txt

# Install EinsteinDB-GPT3
RUN python setup.py install






# RUN DEBIAN_FRONTEND=noninteractive apt upgrade python-pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
