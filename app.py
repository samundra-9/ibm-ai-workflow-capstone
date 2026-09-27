import os
from typing import Any, Dict

from flask import Flask, jsonify, request

from logger import get_logs, log_api_request
from model import predict_revenue, train_model

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_route():
    payload = request.get_json(silent=True) or {}
    date = payload.get("date")
    country = payload.get("country", "all")
    try:
        if not date:
            raise ValueError("Field 'date' is required.")
        result = predict_revenue(date, country=country)
        log_api_request("POST", "/predict", 200, payload, remote_addr=request.remote_addr)
        return jsonify({"status": "success", "data": result}), 200
    except Exception as exc:  # pragma: no cover - defensive API handling
        log_api_request("POST", "/predict", 400, payload, remote_addr=request.remote_addr)
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/train", methods=["POST"])
def train_route():
    payload = request.get_json(silent=True) or {}
    country = payload.get("country", "all")
    try:
        result = train_model(country=country)
        log_api_request("POST", "/train", 200, payload, remote_addr=request.remote_addr)
        return jsonify({"status": "success", "data": result}), 200
    except Exception as exc:  # pragma: no cover - defensive API handling
        log_api_request("POST", "/train", 400, payload, remote_addr=request.remote_addr)
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/logs", methods=["GET"])
def logs_route():
    log_type = request.args.get("type", "all")
    try:
        log_data = get_logs(log_type)
        log_api_request("GET", "/logs", 200, {"type": log_type}, remote_addr=request.remote_addr)
        return jsonify({"status": "success", "logs": log_data}), 200
    except Exception as exc:  # pragma: no cover - defensive API handling
        log_api_request("GET", "/logs", 400, {"type": log_type}, remote_addr=request.remote_addr)
        return jsonify({"status": "error", "message": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
