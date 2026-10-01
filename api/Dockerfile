# Dockerfile for the Flask prediction API (bonus: Deployment with Flask and Docker)
#
# Build:
#   docker build -t cifar10-api -f api/Dockerfile .
#
# Run:
#   docker run -p 5000:5000 cifar10-api
#
# The image expects a trained model at models/cifar10_model.keras and
# models/model_metadata.json - train first with `python src/train.py`,
# or bake a pre-trained model into the image before building.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 5000

ENV PORT=5000

CMD ["python", "api/flask_app.py"]
