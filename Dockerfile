FROM pytorch/pytorch:latest

RUN apt-get install -y cmake
RUN pip install gym
