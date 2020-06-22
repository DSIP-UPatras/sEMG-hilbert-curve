FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN pip install --upgrade pip

COPY ./code /code/

WORKDIR /code

