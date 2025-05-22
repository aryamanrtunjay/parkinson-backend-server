FROM python:3.9-slim

# Install Poppler and dependencies
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
COPY . .

ENV PORT=5000
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "backend_api:app"]