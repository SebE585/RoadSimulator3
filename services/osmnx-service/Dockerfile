FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc g++ libgdal-dev gdal-bin libspatialindex-dev libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py /app/app.py

ENV FLASK_APP=app:app
EXPOSE 5002

CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--workers=1", "--threads=4", "--timeout=180", "app:app"]
