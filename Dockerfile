FROM python:3.9
COPY . /app
WORKDIR /app

CMD python test.py