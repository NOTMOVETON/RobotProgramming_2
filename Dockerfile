FROM python:3.10.12
RUN apt-get update 
COPY requirements.txt /RacingCarRL/requirements.txt
WORKDIR /RacingCarRL
RUN pip install -r requirements.txt