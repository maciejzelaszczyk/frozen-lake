FROM python:3.9.7-buster
ENV PYTHONPATH=/.:$PYTHONPATH
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -U -r requirements.txt
WORKDIR /src
CMD ["python", "train.py"]
