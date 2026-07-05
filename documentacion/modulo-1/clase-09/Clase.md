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
# Instalamos SHAP para calcular explicaciones del modelo.
# Instalamos scikit-learn para entrenar la regresión logística.
# Instalamos pandas y numpy para trabajar con datos tabulares y cálculos.
# Instalamos boto3 para leer y escribir archivos en S3.
%pip install --quiet shap scikit-learn pandas numpy boto3
```

### 2. Leer dataset desde S3

```python
# io nos permite convertir los bytes descargados desde S3 en un archivo en memoria.
import io

# json nos permite convertir diccionarios de Python a texto JSON.
import json

# boto3 nos permite conectarnos a AWS, en este caso a S3.
import boto3

# numpy nos ayuda con cálculos numéricos, promedios y valores absolutos.
import numpy as np

# pandas nos permite manejar datos en tablas llamadas DataFrames.
import pandas as pd

# shap nos permite explicar cuánto aporta cada variable a una predicción.
import shap

# LogisticRegression será el modelo de clasificación de riesgo.
from sklearn.linear_model import LogisticRegression

# train_test_split separa datos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split

# Pipeline une pasos de preparación y modelo en un solo objeto.
from sklearn.pipeline import Pipeline

# StandardScaler normaliza las variables para que estén en escalas comparables.
from sklearn.preprocessing import StandardScaler

# Bucket donde está el dataset sintético y donde guardaremos explicaciones.
BUCKET = "docente-980921750553-us-east-1-an"

# Ruta del CSV dentro del bucket S3.
CSV_KEY = "synthetic_mortgage_dataset.csv"

# Ruta donde guardaremos la explicación global del modelo de riesgo.
RISK_EXPLANATION_KEY = "ml/explanations/risk_global_explanation.json"

# Prefijo que usaremos luego para guardar explicaciones por solicitud.
APPLICATION_EXPLANATIONS_PREFIX = "ml/explanations/applications"

# Lista de variables que el modelo usa para predecir riesgo.
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

# Creamos un cliente de S3 para leer y escribir archivos.
s3 = boto3.client("s3")

# Descargamos el CSV desde S3.
response = s3.get_object(Bucket=BUCKET, Key=CSV_KEY)

# Leemos el contenido descargado como un DataFrame de pandas.
df = pd.read_csv(io.BytesIO(response["Body"].read()))

# Mostramos las primeras filas para confirmar que el dataset cargó bien.
df.head()
```

### 3. Entrenar modelo de riesgo para explicación

```python
# X contiene las variables de entrada que verá el modelo.
X = df[RISK_FEATURES]

# y contiene la etiqueta real: 1 si hubo default, 0 si no hubo default.
y = df["default_flag"]

# Separamos el dataset en entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(
    # Variables de entrada.
    X,

    # Variable objetivo.
    y,

    # Usamos 20% para prueba y 80% para entrenamiento.
    test_size=0.2,

    # Fijamos la aleatoriedad para repetir el mismo resultado.
    random_state=42,

    # Mantenemos proporción parecida de defaults en train y test.
    stratify=y,
)

# Creamos un pipeline con normalización y regresión logística.
risk_model = Pipeline([
    # Primero normalizamos las variables numéricas.
    ("scaler", StandardScaler()),

    # Luego entrenamos una regresión logística.
    # class_weight="balanced" ayuda si hay más casos LOW que HIGH.
    ("logistic", LogisticRegression(class_weight="balanced", max_iter=1000)),
])

# Entrenamos el modelo con los datos de entrenamiento.
risk_model.fit(X_train, y_train)
```

### 4. Calcular SHAP global

```python
# Tomamos 100 filas de entrenamiento como referencia histórica para SHAP.
# Este background sirve para calcular el valor base del modelo.
background = X_train.sample(100, random_state=42)

# Tomamos 200 casos de prueba para calcular importancia global.
# No usamos todo el test para que el notebook corra rápido.
sample = X_test.sample(200, random_state=42)

# Creamos el explicador de SHAP.
# Le pasamos predict_proba porque queremos explicar probabilidades.
# Le pasamos background para que SHAP tenga una referencia histórica.
explainer = shap.Explainer(risk_model.predict_proba, background)

# Calculamos los valores SHAP para los 200 casos de muestra.
shap_values = explainer(sample)

# Extraemos los aportes SHAP de la clase 1.
# Clase 1 significa default o riesgo alto.
risk_class_values = shap_values[:, :, 1].values

# Calculamos la importancia global promedio.
# abs toma el tamaño del aporte sin importar si sube o baja el riesgo.
# mean(axis=0) promedia por variable.
global_importance = np.abs(risk_class_values).mean(axis=0)

# Construimos el JSON de explicación global.
risk_global = {
    # Nombre del modelo explicado.
    "model": "risk_logistic_regression",

    # Tipo de explicación: global significa comportamiento general del modelo.
    "explanation_type": "global",

    # Objetivo explicado: probabilidad de default.
    "target": "default_probability",

    # Lista ordenada de variables más importantes.
    "top_features": [
        {
            # Nombre de la variable.
            "feature": feature,

            # Importancia promedio de esa variable.
            "importance": round(float(importance), 6),
        }

        # Recorremos las variables ordenadas por importancia descendente.
        for feature, importance in sorted(
            # Unimos cada nombre de variable con su importancia.
            zip(RISK_FEATURES, global_importance),

            # Ordenamos usando la importancia.
            key=lambda item: item[1],

            # reverse=True significa de mayor a menor.
            reverse=True,
        )
    ],
}

# Imprimimos el JSON para revisarlo antes de subirlo a S3.
print(json.dumps(risk_global, indent=2))
```

### 5. Guardar explicación global en S3

```python
# Subimos la explicación global al bucket S3.
s3.put_object(
    # Bucket destino.
    Bucket=BUCKET,

    # Ruta del archivo dentro del bucket.
    Key=RISK_EXPLANATION_KEY,

    # Convertimos el diccionario risk_global a JSON y luego a bytes.
    Body=json.dumps(risk_global, indent=2).encode("utf-8"),

    # Indicamos que el archivo es JSON.
    ContentType="application/json",
)

# Mostramos la ruta final del archivo subido.
print(f"Uploaded s3://{BUCKET}/{RISK_EXPLANATION_KEY}")
```

### 6. Explicar un caso local

```python
# Tomamos un caso del conjunto de prueba.
# Usamos doble corchete [[0]] para mantenerlo como DataFrame y no como Series.
case = X_test.iloc[[0]]

# Buscamos el application_id del caso para poder asociar la explicación a una solicitud.
case_application_id = df.loc[case.index[0], "application_id"]

# Aquí se calcula la probabilidad de incumplimiento del caso.
# predict_proba devuelve dos columnas:
# columna 0 -> probabilidad de clase 0, no default
# columna 1 -> probabilidad de clase 1, default
# Por eso usamos [0][1]: primer caso, clase default.
case_probability = risk_model.predict_proba(case)[0][1]

# Aquí calculamos los valores SHAP para el caso local.
# El [:, :, 1] significa que queremos explicar la clase 1: default.
case_shap = explainer(case)[:, :, 1]

# SHAP también nos da un valor base.
# Este es el punto de partida antes de sumar los aportes de las variables.
case_base_value = float(case_shap.base_values[0])

# Sumamos los aportes de todas las variables.
case_contributions_sum = float(case_shap.values[0].sum())

# Verificamos la idea central de SHAP:
# valor base + suma de aportes = predicción explicada.
# En este caso, como estamos explicando predict_proba, el resultado queda en escala de probabilidad.
case_reconstructed_probability = case_base_value + case_contributions_sum

# Construimos el JSON de explicación local para este applicant.
risk_local = {
    # Guardamos el application_id para saber a qué solicitud corresponde.
    "application_id": str(case_application_id),

    # Guardamos la predicción de riesgo del modelo.
    "risk": {
        # Probabilidad de default calculada por el modelo.
        "default_probability": round(float(case_probability), 4),

        # Etiqueta simple basada en threshold 0.5.
        "risk_label": "HIGH" if case_probability >= 0.5 else "LOW",
    },

    # Resumen numérico para verificar base + aportes.
    "shap_summary": {
        # Punto de partida de SHAP antes de mirar este caso.
        "base_value": round(case_base_value, 6),

        # Suma de todos los aportes de las variables.
        "contributions_sum": round(case_contributions_sum, 6),

        # Resultado de base_value + contributions_sum.
        "reconstructed_probability": round(case_reconstructed_probability, 6),

        # Probabilidad directa del modelo para comparar.
        "model_probability": round(float(case_probability), 6),
    },

    # Lista con la explicación por variable.
    "risk_explanation": [
        {
            # Nombre de la variable.
            "feature": feature,

            # Valor de esa variable en este caso.
            "value": float(case.iloc[0][feature]),

            # Aporte SHAP de esa variable.
            # Positivo sube el riesgo; negativo baja el riesgo.
            "contribution": round(float(contribution), 6),
        }

        # Recorremos cada variable junto con su aporte SHAP.
        for feature, contribution in zip(RISK_FEATURES, case_shap.values[0])
    ],
}

# Ordenamos la explicación local desde la variable más influyente.
# abs permite ordenar por impacto total, sea positivo o negativo.
risk_local["risk_explanation"] = sorted(
    # Lista original de explicaciones por variable.
    risk_local["risk_explanation"],

    # Usamos el valor absoluto del aporte como criterio de orden.
    key=lambda item: abs(item["contribution"]),

    # Orden descendente: primero la variable con más impacto.
    reverse=True,
)

# Imprimimos el JSON local para revisar la explicación del caso.
print(json.dumps(risk_local, indent=2))
```

La parte más importante de esta celda es:

```python
case_probability = risk_model.predict_proba(case)[0][1]
```

Eso saca la probabilidad de default directamente del modelo.

Luego SHAP nos permite revisar cómo se forma esa predicción:

```python
case_base_value = float(case_shap.base_values[0])
case_contributions_sum = float(case_shap.values[0].sum())
case_reconstructed_probability = case_base_value + case_contributions_sum
```

Conceptualmente:

```txt
probabilidad base + aportes de variables = probabilidad explicada
```

Ejemplo:

```txt
base_value:                  0.34
suma de aportes SHAP:        0.36
probabilidad reconstruida:   0.70
probabilidad del modelo:     0.70
```

Si hay una diferencia pequeña entre `reconstructed_probability` y `model_probability`, no es un problema para la clase. Puede pasar por redondeos o por el tipo de explicador usado por SHAP.

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
