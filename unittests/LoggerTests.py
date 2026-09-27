import os
import unittest
from pathlib import Path

from logger import get_logs, log_api_request, log_model_metrics


class LoggerTests(unittest.TestCase):
    def test_log_api_request(self):
        entry = log_api_request("POST", "/predict", 200, {"date": "2024-08-15", "country": "USA"})
        self.assertEqual(entry["status_code"], 200)

    def test_log_model_metrics(self):
        entry = log_model_metrics({"rmse": 10.2, "mae": 8.6, "r2": 0.84}, country="USA")
        self.assertEqual(entry["country"], "USA")

    def test_get_logs(self):
        logs = get_logs("all")
        self.assertIn("api", logs)
        self.assertIn("model", logs)


if __name__ == "__main__":
    unittest.main()
