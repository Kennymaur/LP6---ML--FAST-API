FROM python:3.9

ADD app.py

COPY requirements.txt ./

RUN -m pip install -qr requirements.txt

CMD ["python", "./app.py"]