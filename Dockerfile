FROM python:3.10-slim

# Системные зависимости для opencv, geopandas, GDAL
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgdal-dev \
    gdal-bin \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Папка для результатов
RUN mkdir -p results data

CMD ["python", "tree_detector.py", "--help"]