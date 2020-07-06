FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN pip install --upgrade pip
RUN pip install scipy scikit-learn matplotlib pandas

COPY ./code /code/

WORKDIR /code

