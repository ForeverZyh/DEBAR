FROM python:3 
FROM tensorflow/tensorflow:1.13.1-py3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    curl \
    unzip

RUN curl -L -o a.zip 'https://drive.google.com/uc?export=download&id=1GBHFd-fPIBWqJOpIC8ZO8g3F1LoIZYNn'
RUN unzip a.zip

COPY . .
CMD [ "python", "./main.py", "./computation_graphs_and_TP_list/computation_graphs"]