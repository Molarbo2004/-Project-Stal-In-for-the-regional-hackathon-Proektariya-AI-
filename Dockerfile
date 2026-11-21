FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PORT=10000 \
    STAL_IN_WORKSPACE=/var/data/work.db

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 stal-in \
    && install -d -o stal-in -g stal-in /var/data

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py Faults.NNA Faults27x7_var ./
COPY faults/ faults/
COPY web/ web/
COPY analysis/primary_5x2.csv analysis/
COPY ["base model/", "base model/"]

USER stal-in
EXPOSE 10000
CMD ["sh", "-c", "exec python -m uvicorn app:app --host 0.0.0.0 --port \"$PORT\""]
