FROM python:3.9

WORKDIR /nps_app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app ./app

CMD ["python", "./app/index.py"]