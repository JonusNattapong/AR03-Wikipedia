FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY app/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY app/ .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
