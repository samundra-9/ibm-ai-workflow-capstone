import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = Path(os.getenv("IBM_CAPSTONE_LOG_DIR", BASE_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

API_LOG_PATH = LOG_DIR / "api_requests.log"
MODEL_LOG_PATH = LOG_DIR / "model_metrics.log"


def _configure_logger(name: str, file_name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler = logging.FileHandler(file_name)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


api_logger = _configure_logger("ibm_api_logger", API_LOG_PATH)
model_logger = _configure_logger("ibm_model_logger", MODEL_LOG_PATH)


def log_api_request(method: str, path: str, status_code: int, payload: Dict[str, Any] | None = None, remote_addr: str = "unknown") -> Dict[str, Any]:
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": method,
        "path": path,
        "status_code": status_code,
        "payload": payload or {},
        "remote_addr": remote_addr,
    }
    api_logger.info(json.dumps(entry, default=str))
    return entry


def log_model_metrics(metrics: Dict[str, Any], country: str = "all") -> Dict[str, Any]:
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "country": country,
        "metrics": metrics,
    }
    model_logger.info(json.dumps(entry, default=str))
    return entry


def monitor_performance_and_drift(current_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any] | None = None, country: str = "all") -> Dict[str, Any]:
    payload = {"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z", "country": country, "current_metrics": current_metrics}
    if baseline_metrics:
        drift = {
            "rmse_delta": round(current_metrics.get("rmse", 0.0) - baseline_metrics.get("rmse", 0.0), 4),
            "mae_delta": round(current_metrics.get("mae", 0.0) - baseline_metrics.get("mae", 0.0), 4),
            "r2_delta": round(current_metrics.get("r2", 0.0) - baseline_metrics.get("r2", 0.0), 4),
        }
        payload["drift"] = drift
        payload["drift_detected"] = any(abs(value) > 0.05 for value in drift.values())
        model_logger.warning(json.dumps(payload, default=str))
        return payload

    model_logger.info(json.dumps(payload, default=str))
    return payload


def _read_log_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line.split("|", 3)[-1]))
            except (IndexError, json.JSONDecodeError):
                records.append({"message": line})
    return records


def get_logs(log_type: str = "all") -> Dict[str, List[Dict[str, Any]]]:
    log_map = {
        "api": API_LOG_PATH,
        "model": MODEL_LOG_PATH,
    }
    if log_type == "all":
        return {key: _read_log_file(path) for key, path in log_map.items()}
    if log_type not in log_map:
        raise ValueError("log_type must be one of: api, model, all")
    return {log_type: _read_log_file(log_map[log_type])}


__all__ = [
    "api_logger",
    "model_logger",
    "log_api_request",
    "log_model_metrics",
    "monitor_performance_and_drift",
    "get_logs",
]
