FROM python:3.10

COPY . .

RUN true

RUN pip install -r requirements.txt

CMD ["python", "./app.py"]
