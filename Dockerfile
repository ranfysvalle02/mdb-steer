FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY mdb_steer ./mdb_steer
RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "mdb_steer"]
CMD ["--help"]
