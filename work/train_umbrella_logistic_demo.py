import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "rain_probability",
    "humidity",
    "cloudiness",
    "wind_speed",
    "is_rainy_season",
]


def clip(values, low, high):
    return np.minimum(np.maximum(values, low), high)


def make_dataset(rows=800, seed=7):
    rng = np.random.default_rng(seed)

    rain_probability = clip(rng.beta(2.0, 2.0, rows), 0, 1)
    humidity = clip(0.35 + rain_probability * 0.45 + rng.normal(0, 0.12, rows), 0, 1)
    cloudiness = clip(0.25 + rain_probability * 0.55 + rng.normal(0, 0.14, rows), 0, 1)
    wind_speed = clip(rng.normal(18, 8, rows), 0, 55)
    is_rainy_season = rng.binomial(1, clip(0.25 + rain_probability * 0.5, 0.05, 0.9))

    signal = (
        5.0 * rain_probability
        + 0.8 * humidity
        + 0.6 * cloudiness
        + 0.015 * wind_speed
        + 0.5 * is_rainy_season
        + rng.normal(0, 0.45, rows)
        - 2.7
    )
    umbrella_probability = 1 / (1 + np.exp(-signal))
    take_umbrella = rng.binomial(1, umbrella_probability)

    return pd.DataFrame(
        {
            "rain_probability": np.round(rain_probability, 3),
            "humidity": np.round(humidity, 3),
            "cloudiness": np.round(cloudiness, 3),
            "wind_speed": np.round(wind_speed, 1),
            "is_rainy_season": is_rainy_season,
            "take_umbrella": take_umbrella,
        }
    )


def main():
    df = make_dataset()
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(df[FEATURES], df["take_umbrella"])

    coefficients = pd.DataFrame(
        {
            "feature": FEATURES,
            "coefficient": model.named_steps["logistic"].coef_[0],
        }
    ).sort_values("coefficient", key=lambda values: values.abs(), ascending=False)

    same_day_low_rain = pd.DataFrame(
        [
            {
                "rain_probability": 0.15,
                "humidity": 0.72,
                "cloudiness": 0.75,
                "wind_speed": 18,
                "is_rainy_season": 1,
            }
        ]
    )
    same_day_high_rain = same_day_low_rain.copy()
    same_day_high_rain["rain_probability"] = 0.85

    low_probability = model.predict_proba(same_day_low_rain)[0][1]
    high_probability = model.predict_proba(same_day_high_rain)[0][1]

    print("Coeficientes aprendidos:")
    print(coefficients.to_string(index=False))
    print()
    print("Mismo dia, solo cambia rain_probability:")
    print("rain_probability = 0.15 ->", round(float(low_probability), 4))
    print("rain_probability = 0.85 ->", round(float(high_probability), 4))


if __name__ == "__main__":
    main()
