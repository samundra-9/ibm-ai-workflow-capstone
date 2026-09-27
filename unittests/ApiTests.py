import os
import tempfile
import unittest
from pathlib import Path

from app import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_predict_endpoint_requires_date(self):
        response = self.app.post("/predict", json={"country": "USA"})
        self.assertEqual(response.status_code, 400)

    def test_predict_endpoint_country_all(self):
        response = self.app.post("/predict", json={"date": "2024-08-15", "country": "all"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("total_prediction", payload["data"])

    def test_train_endpoint(self):
        response = self.app.post("/train", json={"country": "USA"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("best_model", payload["data"])

    def test_logs_endpoint(self):
        response = self.app.get("/logs?type=api")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("logs", payload)


if __name__ == "__main__":
    unittest.main()
