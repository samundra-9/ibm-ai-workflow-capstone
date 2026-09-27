import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from logger import log_model_metrics, monitor_performance_and_drift

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("IBM_CAPSTONE_DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("IBM_CAPSTONE_MODELS_DIR", BASE_DIR / "models"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["USA", "Canada", "UK", "Germany", "India", "Japan", "Brazil", "Australia"]
HOLIDAY_DATES = {"01-01", "07-04", "11-28", "12-25"}


def ingest_data(start_date: str = "2023-01-01", end_date: str = "2025-12-31", countries: Optional[Iterable[str]] = None) -> pd.DataFrame:
    country_list = list(countries or COUNTRIES)
    data_path = DATA_DIR / "mock_revenue_data.csv"
    if data_path.exists():
        return pd.read_csv(data_path)

    records: List[Dict[str, Any]] = []
    for country in country_list:
        country_index = country_list.index(country)
        for current_date in pd.date_range(start_date, end_date, freq="D"):
            seed = sum(map(ord, f"{current_date.date()}-{country}"))
            rng = np.random.default_rng(seed)
            seasonal_wave = 520 + 160 * np.sin((current_date.dayofyear / 365) * 2 * np.pi + country_index)
            holiday_boost = 180 if current_date.strftime("%m-%d") in HOLIDAY_DATES else 0
            weekend_factor = 1.28 if current_date.dayofweek >= 5 else 1.0
            country_bias = country_index * 200
            trend = 0.32 * current_date.toordinal()
            noise = rng.normal(0, 24)
            revenue = (seasonal_wave + holiday_boost + country_bias + trend) * weekend_factor + noise
            records.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "country": country,
                    "revenue": round(float(revenue), 2),
                    "month": current_date.month,
                    "day_of_week": current_date.dayofweek,
                    "day_of_year": current_date.dayofyear,
                    "is_weekend": int(current_date.dayofweek >= 5),
                    "is_holiday": int(current_date.strftime("%m-%d") in HOLIDAY_DATES),
                    "week_of_year": int(current_date.isocalendar().week),
                    "trend": int((current_date - pd.Timestamp(start_date)).days),
                    "seasonal_sin": float(np.sin(2 * np.pi * current_date.dayofyear / 365)),
                    "seasonal_cos": float(np.cos(2 * np.pi * current_date.dayofyear / 365)),
                }
            )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(data_path, index=False)
    return df


def _feature_columns() -> List[str]:
    return [
        "country_index",
        "month",
        "day_of_week",
        "day_of_year",
        "is_weekend",
        "is_holiday",
        "week_of_year",
        "trend",
        "seasonal_sin",
        "seasonal_cos",
    ]


def _prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["country_index"] = frame["country"].map({country: idx for idx, country in enumerate(COUNTRIES)})
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _model_signature(country: str, model_name: str) -> str:
    return f"{model_name}_{country}.joblib"


def _save_metrics(country: str, metrics: Dict[str, Dict[str, float]], best_model: str) -> None:
    payload = {"country": country, "best_model": best_model, "metrics": metrics}
    with (MODELS_DIR / f"metrics_{country}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_saved_metrics(country: str) -> Optional[Dict[str, Any]]:
    metric_path = MODELS_DIR / f"metrics_{country}.json"
    if metric_path.exists():
        with metric_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def _baseline_model() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def _build_training_configs() -> Dict[str, Any]:
    return {
        "baseline": _baseline_model(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, max_depth=12),
        "gradient_boosting": GradientBoostingRegressor(random_state=42, n_estimators=250, learning_rate=0.05, max_depth=3),
    }


def _evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def train_model(country: str = "all", data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    df = data if data is not None else ingest_data()
    if country == "all":
        training_df = _prepare_training_frame(df)
    else:
        if country not in COUNTRIES:
            raise ValueError(f"Unsupported country '{country}'. Valid countries are: {COUNTRIES + ['all']}")
        training_df = _prepare_training_frame(df[df["country"] == country].copy())

    if training_df.empty:
        raise ValueError(f"No data found for country '{country}'")

    feature_columns = _feature_columns()
    X = training_df[feature_columns]
    y = training_df["revenue"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_metrics: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, Any] = {}
    for model_name, model in _build_training_configs().items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        model_metrics[model_name] = _evaluate_model(model, X_test, y_test)
        joblib.dump(model, MODELS_DIR / _model_signature(country, model_name))
        log_model_metrics(model_metrics[model_name], country=country)

    best_model = max(model_metrics, key=lambda name: model_metrics[name]["r2"])
    _save_metrics(country, model_metrics, best_model)
    baseline = model_metrics.get("baseline", {})
    if baseline:
        monitor_performance_and_drift(model_metrics[best_model], baseline, country=country)

    return {
        "country": country,
        "best_model": best_model,
        "metrics": model_metrics,
        "message": f"Model training completed for {country}.",
    }


def _get_best_model_for_country(country: str) -> str:
    saved_metadata = _load_saved_metrics(country)
    if saved_metadata and saved_metadata.get("best_model"):
        return saved_metadata["best_model"]
    trained = train_model(country=country)
    return trained["best_model"]


def _prediction_row(date: Any, country: str) -> pd.DataFrame:
    date_value = pd.to_datetime(date).normalize()
    row = {
        "country_index": COUNTRIES.index(country),
        "month": int(date_value.month),
        "day_of_week": int(date_value.dayofweek),
        "day_of_year": int(date_value.dayofyear),
        "is_weekend": int(date_value.dayofweek >= 5),
        "is_holiday": int(date_value.strftime("%m-%d") in HOLIDAY_DATES),
        "week_of_year": int(date_value.isocalendar().week),
        "trend": int((date_value - pd.Timestamp("2023-01-01")).days),
        "seasonal_sin": float(np.sin(2 * np.pi * date_value.dayofyear / 365)),
        "seasonal_cos": float(np.cos(2 * np.pi * date_value.dayofyear / 365)),
    }
    return pd.DataFrame([row], columns=_feature_columns())


def _predict_single_country(date: Any, country: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    if country not in COUNTRIES:
        raise ValueError(f"Unsupported country '{country}'. Valid countries are: {COUNTRIES}")
    chosen_model = model_name or _get_best_model_for_country(country)
    model_path = MODELS_DIR / _model_signature(country, chosen_model)
    if not model_path.exists():
        train_model(country=country)
    model = joblib.load(model_path)
    prediction = float(model.predict(_prediction_row(date, country))[0])
    return {"date": pd.to_datetime(date).strftime("%Y-%m-%d"), "country": country, "model": chosen_model, "prediction": round(prediction, 2)}


def predict_revenue(date: Any, country: str = "all", model_name: Optional[str] = None) -> Dict[str, Any]:
    date_value = pd.to_datetime(date).normalize()
    if country == "all":
        country_predictions = {country_name: _predict_single_country(date_value, country_name, model_name)["prediction"] for country_name in COUNTRIES}
        total_prediction = round(float(sum(country_predictions.values())), 2)
        result = {
            "date": date_value.strftime("%Y-%m-%d"),
            "country": "all",
            "model": model_name or "ensemble_best",
            "prediction": country_predictions,
            "total_prediction": total_prediction,
        }
        return result

    return _predict_single_country(date_value, country, model_name)


__all__ = [
    "COUNTRIES",
    "ingest_data",
    "train_model",
    "predict_revenue",
    "MODELS_DIR",
    "DATA_DIR",
]
