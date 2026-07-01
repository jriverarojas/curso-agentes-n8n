# Clase 9: Explicabilidad bancaria con notebooks y endpoints

| | |
|---|---|
| **Clase** | 9 de 11 |
| **Duración** | 3 horas |
| **Controlador** | `Clase09Controller` |
| **Endpoints** | `GET /modulo1/clase09/explanations/umbrella`, `GET /modulo1/clase09/explanations/risk/global`, `GET /modulo1/clase09/explanations/amount/global`, `GET /modulo1/clase09/explanations/compare`, `GET /modulo1/clase09/applications/:applicationId/explanation` |

## Objetivos

Al terminar esta sesión podrás:

- Llevar la idea de explicabilidad del paraguas al caso bancario.
- Explicar la diferencia entre explicación global y explicación local.
- Usar un notebook de SageMaker para calcular explicaciones del modelo de riesgo.
- Usar un notebook de SageMaker para calcular explicaciones del modelo de monto.
- Generar JSONs de explicación y subirlos a S3.
- Crear endpoints NestJS que lean explicaciones desde S3.
- Traducir explicaciones técnicas a frases de negocio.

---

## Parte teórica

### 1. De paraguas a crédito

En Clase 8 explicamos una predicción simple:

```txt
¿Llevar paraguas?
```

Ahora explicaremos predicciones bancarias:

```txt
¿Por qué el riesgo es HIGH?
¿Por qué el monto recomendado es menor que el solicitado?
```

La lógica es la misma:

```txt
predicción base + aportes de variables = predicción explicada
```

### 2. Qué queremos explicar

Modelo de riesgo:

```txt
Salida: defaultProbability
```

Preguntas:

```txt
¿Qué variables suben el riesgo?
¿Qué variables bajan el riesgo?
```

Modelo de monto:

```txt
Salida: recommendedAmount
```

Preguntas:

```txt
¿Qué variables suben el monto recomendado?
¿Qué variables bajan el monto recomendado?
```

### 3. Explicación global

Explicación global responde:

```txt
En general, ¿qué variables usa más el modelo?
```

Ejemplo:

| Feature | Importancia |
|---------|------------:|
| `total_obligations_to_income_ratio` | 0.31 |
| `debt_to_income_ratio` | 0.24 |
| `credit_history_score` | 0.18 |
| `loan_to_value_ratio` | 0.12 |

Traducción:

```txt
En general, el modelo de riesgo mira mucho la carga total del cliente, su deuda actual y su historial crediticio.
```

### 4. Explicación local

Explicación local responde:

```txt
Para este applicant, ¿qué variables empujaron esta predicción?
```

Ejemplo:

| Feature | Valor | Contribución |
|---------|------:|-------------:|
| `total_obligations_to_income_ratio` | 1.45 | +0.18 |
| `debt_to_income_ratio` | 0.52 | +0.12 |
| `credit_history_score` | 40 | +0.09 |
| `employment_stability_score` | 90 | -0.04 |

Traducción:

```txt
Este caso sale riesgoso principalmente porque la carga total y la deuda mensual son altas. La buena estabilidad laboral ayuda, pero no compensa todo.
```

### 5. Ejercicio manual: explicación bancaria

Trabajen en Excel, Google Sheets o una hoja.

Caso:

```txt
Probabilidad base de default: 0.35
```

Tabla:

| Feature | Valor | Aporte |
|---------|------:|-------:|
| Base | | 0.35 |
| `debt_to_income_ratio` | 0.52 | +0.12 |
| `total_obligations_to_income_ratio` | 1.45 | +0.18 |
| `credit_history_score` | 40 | +0.09 |
| `employment_stability_score` | 90 | -0.04 |
| Total | | 0.70 |

Preguntas:

1. ¿Qué variable empujó más el riesgo?
2. ¿Qué variable ayudó al applicant?
3. ¿Cómo lo explicarías en lenguaje de negocio?
4. ¿Qué documento o dato revisarías primero?

Respuesta esperada:

```txt
El riesgo aumenta por la carga total y la deuda actual. La estabilidad laboral ayuda, pero el historial y la carga financiera siguen siendo señales fuertes de riesgo.
```

---

## Parte práctica A: notebook de explicabilidad para riesgo

Usaremos el mismo dataset sintético y el mismo modelo de riesgo de Clase 6.

### 1. Instalar dependencias

```python
%pip install --quiet shap scikit-learn pandas numpy boto3
```

### 2. Leer dataset desde S3

```python
import io
import json

import boto3
import numpy as np
import pandas as pd
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BUCKET = "docente-980921750553-us-east-1-an"
CSV_KEY = "synthetic_mortgage_dataset.csv"
RISK_EXPLANATION_KEY = "ml/explanations/risk_global_explanation.json"
APPLICATION_EXPLANATIONS_PREFIX = "ml/explanations/applications"

RISK_FEATURES = [
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "payment_to_income_ratio",
    "expense_to_income_ratio",
    "total_obligations_to_income_ratio",
    "employment_stability_score",
    "banking_capacity_score",
    "credit_history_score",
]

s3 = boto3.client("s3")
response = s3.get_object(Bucket=BUCKET, Key=CSV_KEY)
df = pd.read_csv(io.BytesIO(response["Body"].read()))
df.head()
```

### 3. Entrenar modelo de riesgo para explicación

```python
X = df[RISK_FEATURES]
y = df["default_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

risk_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(class_weight="balanced", max_iter=1000)),
])

risk_model.fit(X_train, y_train)
```

### 4. Calcular SHAP global

```python
background = X_train.sample(100, random_state=42)
sample = X_test.sample(200, random_state=42)

explainer = shap.Explainer(risk_model.predict_proba, background)
shap_values = explainer(sample)

risk_class_values = shap_values[:, :, 1].values
global_importance = np.abs(risk_class_values).mean(axis=0)

risk_global = {
    "model": "risk_logistic_regression",
    "explanation_type": "global",
    "target": "default_probability",
    "top_features": [
        {
            "feature": feature,
            "importance": round(float(importance), 6),
        }
        for feature, importance in sorted(
            zip(RISK_FEATURES, global_importance),
            key=lambda item: item[1],
            reverse=True,
        )
    ],
}

print(json.dumps(risk_global, indent=2))
```

### 5. Guardar explicación global en S3

```python
s3.put_object(
    Bucket=BUCKET,
    Key=RISK_EXPLANATION_KEY,
    Body=json.dumps(risk_global, indent=2).encode("utf-8"),
    ContentType="application/json",
)

print(f"Uploaded s3://{BUCKET}/{RISK_EXPLANATION_KEY}")
```

### 6. Explicar un caso local

```python
case = X_test.iloc[[0]]
case_application_id = df.loc[case.index[0], "application_id"]
case_probability = risk_model.predict_proba(case)[0][1]
case_shap = explainer(case)[:, :, 1]

risk_local = {
    "application_id": str(case_application_id),
    "risk": {
        "default_probability": round(float(case_probability), 4),
        "risk_label": "HIGH" if case_probability >= 0.5 else "LOW",
    },
    "risk_explanation": [
        {
            "feature": feature,
            "value": float(case.iloc[0][feature]),
            "contribution": round(float(contribution), 6),
        }
        for feature, contribution in zip(RISK_FEATURES, case_shap.values[0])
    ],
}

risk_local["risk_explanation"] = sorted(
    risk_local["risk_explanation"],
    key=lambda item: abs(item["contribution"]),
    reverse=True,
)

print(json.dumps(risk_local, indent=2))
```

---

## Parte práctica B: notebook de explicabilidad para monto

### 1. Entrenar XGBoost para explicación

```python
from xgboost import XGBRegressor

AMOUNT_EXPLANATION_KEY = "ml/explanations/amount_global_explanation.json"

AMOUNT_FEATURES = [
    "net_monthly_income",
    "monthly_debt_payment",
    "monthly_expenses",
    "property_value",
    "requested_amount",
    "requested_term_months",
    "estimated_monthly_payment",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "payment_to_income_ratio",
    "expense_to_income_ratio",
    "total_obligations_to_income_ratio",
    "employment_stability_score",
    "banking_capacity_score",
    "credit_history_score",
]

X_amount = df[AMOUNT_FEATURES]
y_amount = df["recommended_amount"]

X_train_amount, X_test_amount, y_train_amount, y_test_amount = train_test_split(
    X_amount,
    y_amount,
    test_size=0.2,
    random_state=42,
)

amount_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=180,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.9,
    random_state=42,
)

amount_model.fit(X_train_amount, y_train_amount)
```

### 2. Calcular SHAP global del modelo de monto

```python
amount_explainer = shap.Explainer(amount_model, X_train_amount.sample(100, random_state=42))
amount_sample = X_test_amount.sample(200, random_state=42)
amount_shap_values = amount_explainer(amount_sample)

amount_importance = np.abs(amount_shap_values.values).mean(axis=0)

amount_global = {
    "model": "amount_xgboost_regressor",
    "explanation_type": "global",
    "target": "recommended_amount",
    "top_features": [
        {
            "feature": feature,
            "importance": round(float(importance), 2),
        }
        for feature, importance in sorted(
            zip(AMOUNT_FEATURES, amount_importance),
            key=lambda item: item[1],
            reverse=True,
        )
    ],
}

print(json.dumps(amount_global, indent=2))
```

### 3. Guardar explicación global de monto en S3

```python
s3.put_object(
    Bucket=BUCKET,
    Key=AMOUNT_EXPLANATION_KEY,
    Body=json.dumps(amount_global, indent=2).encode("utf-8"),
    ContentType="application/json",
)

print(f"Uploaded s3://{BUCKET}/{AMOUNT_EXPLANATION_KEY}")
```

### 4. Crear explicación combinada para una aplicación

```python
amount_case = X_test_amount.loc[[case.index[0]]]
amount_prediction = amount_model.predict(amount_case)[0]
amount_case_shap = amount_explainer(amount_case)

combined_explanation = {
    "application_id": str(case_application_id),
    "risk": risk_local["risk"],
    "risk_explanation": risk_local["risk_explanation"][:5],
    "amount": {
        "recommended_amount": round(float(amount_prediction), 2),
    },
    "amount_explanation": [
        {
            "feature": feature,
            "value": float(amount_case.iloc[0][feature]),
            "contribution": round(float(contribution), 2),
        }
        for feature, contribution in zip(AMOUNT_FEATURES, amount_case_shap.values[0])
    ],
}

combined_explanation["amount_explanation"] = sorted(
    combined_explanation["amount_explanation"],
    key=lambda item: abs(item["contribution"]),
    reverse=True,
)[:5]

application_key = f"{APPLICATION_EXPLANATIONS_PREFIX}/{case_application_id}.json"
s3.put_object(
    Bucket=BUCKET,
    Key=application_key,
    Body=json.dumps(combined_explanation, indent=2).encode("utf-8"),
    ContentType="application/json",
)

print(f"Uploaded s3://{BUCKET}/{application_key}")
```

---

## Parte práctica C: endpoints NestJS

Variables en `.env`:

```env
EXPLAIN_UMBRELLA_KEY=ml/explanations/umbrella_explanation.json
EXPLAIN_RISK_GLOBAL_KEY=ml/explanations/risk_global_explanation.json
EXPLAIN_AMOUNT_GLOBAL_KEY=ml/explanations/amount_global_explanation.json
EXPLAIN_APPLICATIONS_PREFIX=ml/explanations/applications
```

Endpoints:

| Endpoint | Qué devuelve |
|----------|--------------|
| `GET /modulo1/clase09/explanations/umbrella` | explicación del modelo paraguas |
| `GET /modulo1/clase09/explanations/risk/global` | importancia global del modelo de riesgo |
| `GET /modulo1/clase09/explanations/amount/global` | importancia global del modelo de monto |
| `GET /modulo1/clase09/explanations/compare` | comparación de explicaciones globales |
| `GET /modulo1/clase09/applications/:applicationId/explanation` | explicación local combinada |

Prueba:

```bash
curl http://localhost:3000/modulo1/clase09/explanations/compare \
  -H "x-api-key: test1" \
  -H "x-api-secret: pass1"
```

Explicación local:

```bash
curl http://localhost:3000/modulo1/clase09/applications/APP-000123/explanation \
  -H "x-api-key: test1" \
  -H "x-api-secret: pass1"
```

## Entrega

- JSON de explicación global de riesgo en S3.
- JSON de explicación global de monto en S3.
- JSON de explicación local de una aplicación en S3.
- Resultado de `GET /explanations/compare`.
- Traducción en lenguaje de negocio de una explicación local.
