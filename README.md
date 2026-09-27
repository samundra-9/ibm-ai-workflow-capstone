# IBM AI Workflow Capstone Project

This project demonstrates an end-to-end AI workflow in production: data ingestion, model training, inference, logging, monitoring, API deployment, and testing.

## Project structure

```text
.
├── app.py
├── model.py
├── logger.py
├── run_tests.py
├── requirements.txt
├── Dockerfile
├── data/
├── models/
├── logs/
├── unittests/
│   ├── __init__.py
│   ├── ApiTests.py
│   ├── ModelTests.py
│   └── LoggerTests.py
└── README.md
```

## Features

- Mock daily revenue generation for multiple countries
- Model comparison between baseline, Random Forest, and Gradient Boosting
- Prediction for a single country or all countries together
- REST API endpoints for prediction, training, and log retrieval
- Logging and drift/performance monitoring
- Unit test suite for application, model, and logging behavior
- Dockerized service deployment

## Local setup

```bash
cd ibm-capstone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the tests

```bash
python run_tests.py
```

## Run the API locally

```bash
python app.py
```

The app listens on:

```text
http://localhost:8080
```

## Test the API with curl

### Train a model

```bash
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{"country": "USA"}'
```

### Predict revenue for a specific country

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-08-15", "country": "USA"}'
```

### Predict revenue for all countries

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-08-15", "country": "all"}'
```

### Retrieve logs

```bash
curl "http://localhost:8080/logs?type=all"
```

## Docker build and run

```bash
docker build -t ibm-ai-capstone .
docker run -p 8080:8080 ibm-ai-capstone
```

## Notes

- The project uses a deterministic mock data generator so it can be trained and tested reliably without needing external datasets.
- All production artifacts are stored under the `models/` directory while logs are stored under `logs/`.
- The test suite uses isolated runtime data and log output paths via the underlying app and model logic.
