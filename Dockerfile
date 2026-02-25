FROM python:3.12-slim

# Install uv
RUN pip install uv

WORKDIR /app

COPY pyproject.toml .
COPY main.py .
COPY .env .
COPY config.json .

# Install dependencies
RUN uv sync

CMD ["uv", "run", "main.py"]
