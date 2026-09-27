import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model import COUNTRIES, ingest_data, predict_revenue, train_model


class ModelTests(unittest.TestCase):
    def test_ingest_data_creates_dataset(self):
        df = ingest_data(start_date="2024-01-01", end_date="2024-01-10", countries=["USA", "Canada"])
        self.assertFalse(df.empty)
        self.assertIn("country", df.columns)
        self.assertIn("revenue", df.columns)

    def test_predict_revenue_for_specific_country(self):
        prediction = predict_revenue("2024-08-15", country="USA")
        self.assertIn("country", prediction)
        self.assertIn("prediction", prediction)

    def test_predict_revenue_all_countries(self):
        prediction = predict_revenue("2024-08-15", country="all")
        self.assertEqual(prediction["country"], "all")
        self.assertIn("total_prediction", prediction)

    def test_train_model_for_country(self):
        result = train_model(country="USA")
        self.assertIn("best_model", result)
        self.assertIn("metrics", result)


if __name__ == "__main__":
    unittest.main()
