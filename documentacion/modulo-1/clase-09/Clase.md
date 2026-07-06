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
# Instalamos xgboost porque en la Parte B entrenaremos el modelo de monto.
%pip install --quiet shap scikit-learn pandas numpy boto3 xgboost
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

### 7. Explicar una solicitud real con tus propias features

El caso anterior sale del dataset de prueba. Ahora haremos algo más cercano al sistema real: tomar las features de una solicitud específica y pedirle al notebook que explique ese applicant.

Importante:

```txt
El application_id no existe cuando entrenamos el modelo.
El application_id aparece recién cuando ya se creó una solicitud en el sistema.
```

Por eso hay dos tipos de JSON de explicación:

| Tipo de JSON | Cuándo se genera | Tiene `application_id` |
|--------------|------------------|------------------------|
| Explicación global | Después de entrenar el modelo | No |
| Explicación local | Después de tener una solicitud concreta con features | Sí |

En esta clase usamos un `application_id` de ejemplo porque ya tenemos una solicitud creada en el ambiente del curso.

Los alumnos pueden reemplazar este bloque por las features de su propia aplicación cuando ya tengan:

```txt
1. application_id creado;
2. documentos procesados;
3. features calculadas por Clase 5.
```

Si todavía no tienen un `application_id`, pueden hacer el ejercicio con un identificador temporal, por ejemplo:

```txt
application_id = "demo-application"
```

Pero para que el endpoint de NestJS funcione con una solicitud real, el archivo debe guardarse usando el `application_id` real.

```python
# Creamos un payload parecido al que devuelve nuestro backend.
application_payload = {
    # Dentro de features ponemos los datos ya preparados por Clase 5.
    "features": {
        # Identificador real de la solicitud que queremos explicar.
        "application_id": "dd7b3608-14b3-4426-a927-d92ead8aa9de",

        # Ingreso neto mensual del applicant.
        "net_monthly_income": 10400.0,

        # Pago mensual de deudas existentes.
        "monthly_debt_payment": 2850.0,

        # Gastos mensuales declarados.
        "monthly_expenses": 4500.0,

        # Valor del inmueble.
        "property_value": 700000.0,

        # Monto solicitado.
        "requested_amount": 500000.0,

        # Plazo solicitado en meses.
        "requested_term_months": 240,

        # Cuota estimada del nuevo crédito.
        "estimated_monthly_payment": 2083.33,

        # Relación deuda mensual / ingreso mensual.
        "debt_to_income_ratio": 0.274,

        # Relación monto solicitado / valor inmueble.
        "loan_to_value_ratio": 0.7143,

        # Relación cuota estimada / ingreso mensual.
        "payment_to_income_ratio": 0.2003,

        # Relación gastos mensuales / ingreso mensual.
        "expense_to_income_ratio": 0.4327,

        # Relación deuda + gastos + nueva cuota / ingreso mensual.
        "total_obligations_to_income_ratio": 0.9071,

        # Score de estabilidad laboral.
        "employment_stability_score": 40,

        # Score de capacidad bancaria.
        "banking_capacity_score": 40,

        # Score de historial crediticio.
        "credit_history_score": 76,

        # Etiqueta sintética usada para entrenamiento, no para predecir este caso.
        "synthetic_risk_label": 0,
    }
}

# Sacamos solo el diccionario interno de features.
application_features = application_payload["features"]

# Guardamos el application_id para incluirlo en la explicación final.
real_application_id = application_features["application_id"]

# Creamos un DataFrame con una sola fila.
# Usamos únicamente RISK_FEATURES porque son las variables que espera el modelo de riesgo.
real_case = pd.DataFrame([{
    feature: application_features[feature]
    for feature in RISK_FEATURES
}])

# Calculamos la probabilidad de default para esta solicitud real.
real_probability = risk_model.predict_proba(real_case)[0][1]

# Calculamos los valores SHAP para esta solicitud real.
real_shap = explainer(real_case)[:, :, 1]

# Extraemos el valor base de SHAP.
real_base_value = float(real_shap.base_values[0])

# Sumamos todos los aportes SHAP de las variables.
real_contributions_sum = float(real_shap.values[0].sum())

# Reconstruimos la probabilidad usando base + aportes.
real_reconstructed_probability = real_base_value + real_contributions_sum

# Construimos el JSON de explicación para esta solicitud.
real_risk_explanation = {
    # Guardamos el identificador de la solicitud.
    "application_id": real_application_id,

    # Guardamos la predicción del modelo.
    "risk": {
        # Probabilidad de default calculada por el modelo.
        "default_probability": round(float(real_probability), 4),

        # Etiqueta de riesgo según threshold 0.5.
        "risk_label": "HIGH" if real_probability >= 0.5 else "LOW",
    },

    # Guardamos el resumen SHAP para verificar la suma.
    "shap_summary": {
        # Punto de partida del modelo según SHAP.
        "base_value": round(real_base_value, 6),

        # Suma de aportes de todas las variables.
        "contributions_sum": round(real_contributions_sum, 6),

        # Resultado de base + aportes.
        "reconstructed_probability": round(real_reconstructed_probability, 6),

        # Probabilidad directa calculada por el modelo.
        "model_probability": round(float(real_probability), 6),
    },

    # Lista de variables con sus valores y aportes.
    "risk_explanation": [
        {
            # Nombre de la variable.
            "feature": feature,

            # Valor de esa variable en la solicitud real.
            "value": float(real_case.iloc[0][feature]),

            # Aporte SHAP de esa variable.
            "contribution": round(float(contribution), 6),
        }

        # Recorremos cada feature con su contribución SHAP.
        for feature, contribution in zip(RISK_FEATURES, real_shap.values[0])
    ],
}

# Ordenamos las variables por impacto absoluto.
real_risk_explanation["risk_explanation"] = sorted(
    # Lista original de aportes.
    real_risk_explanation["risk_explanation"],

    # Criterio de ordenamiento: tamaño del aporte.
    key=lambda item: abs(item["contribution"]),

    # Mayor impacto primero.
    reverse=True,
)

# Imprimimos la explicación completa.
print(json.dumps(real_risk_explanation, indent=2))
```

También podemos imprimir una explicación corta en español:

```python
# Tomamos las tres variables que más influyeron en la predicción.
top_risk_reasons = real_risk_explanation["risk_explanation"][:3]

# Imprimimos la predicción principal.
print(
    f'El modelo predice riesgo {real_risk_explanation["risk"]["risk_label"]} '
    f'con probabilidad de default {real_probability:.2%}.'
)

# Imprimimos las razones principales.
print("Principales variables que explican la predicción:")

# Recorremos las tres variables más importantes.
for index, reason in enumerate(top_risk_reasons, start=1):
    # contribution > 0 significa que sube el riesgo.
    direction = "sube el riesgo" if reason["contribution"] > 0 else "baja el riesgo"

    # Imprimimos una línea entendible para el alumno.
    print(
        f'{index}. {reason["feature"]}: valor {reason["value"]}, '
        f'aporte {reason["contribution"]:+.6f} ({direction})'
    )
```

Salida esperada aproximada:

```txt
El modelo predice riesgo LOW con probabilidad de default 38.20%.
Principales variables que explican la predicción:
1. total_obligations_to_income_ratio: valor 0.9071, aporte +0.120000 (sube el riesgo)
2. credit_history_score: valor 76.0, aporte -0.080000 (baja el riesgo)
3. employment_stability_score: valor 40.0, aporte +0.060000 (sube el riesgo)
```

Los números pueden variar según el dataset y el entrenamiento, pero esta celda debe mostrar claramente qué variables suben o bajan el riesgo de la solicitud real.

---

## Parte práctica B: notebook de explicabilidad para monto

### 0. Preparar datos para la Parte B

La Parte B usa el mismo `df` que cargamos en la Parte A.

Si ejecutaste toda la Parte A, esta celda solo confirmará que `df` ya existe. Si entraste directo a la Parte B, esta celda volverá a leer el CSV desde S3.

```python
# Importamos io para leer bytes descargados desde S3 como archivo en memoria.
import io

# Importamos json para imprimir y guardar diccionarios como JSON.
import json

# Importamos boto3 para leer el CSV desde S3.
import boto3

# Importamos numpy para cálculos numéricos.
import numpy as np

# Importamos pandas para trabajar con DataFrames.
import pandas as pd

# Importamos shap para explicar el modelo de monto.
import shap

# Importamos train_test_split por si no se ejecutó la Parte A.
from sklearn.model_selection import train_test_split

# Si BUCKET no existe porque saltamos la Parte A, lo definimos aquí.
if "BUCKET" not in globals():
    BUCKET = "docente-980921750553-us-east-1-an"

# Si CSV_KEY no existe porque saltamos la Parte A, lo definimos aquí.
if "CSV_KEY" not in globals():
    CSV_KEY = "synthetic_mortgage_dataset.csv"

# Si s3 no existe porque saltamos la Parte A, creamos el cliente.
if "s3" not in globals():
    s3 = boto3.client("s3")

# Si df no existe, lo cargamos desde S3.
if "df" not in globals():
    response = s3.get_object(Bucket=BUCKET, Key=CSV_KEY)
    df = pd.read_csv(io.BytesIO(response["Body"].read()))

# Mostramos las primeras filas para confirmar que el dataset está listo.
df.head()
```

### 1. Entrenar XGBoost para explicación

```python
# Importamos XGBRegressor para entrenar el modelo de monto recomendado.
from xgboost import XGBRegressor

# Importamos xgboost completo para usar DMatrix y pred_contribs.
import xgboost as xgb

# Ruta donde guardaremos la explicación global del modelo de monto.
AMOUNT_EXPLANATION_KEY = "ml/explanations/amount_global_explanation.json"

# Variables que usará el modelo de monto.
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

# X_amount contiene las variables de entrada para el modelo de monto.
X_amount = df[AMOUNT_FEATURES]

# y_amount contiene el valor que queremos predecir: monto recomendado.
y_amount = df["recommended_amount"]

# Separamos datos de entrenamiento y prueba.
X_train_amount, X_test_amount, y_train_amount, y_test_amount = train_test_split(
    # Variables de entrada.
    X_amount,

    # Variable objetivo.
    y_amount,

    # Usamos 20% para prueba.
    test_size=0.2,

    # Fijamos la aleatoriedad para repetir resultados.
    random_state=42,
)

# Creamos el modelo XGBoost de regresión.
amount_model = XGBRegressor(
    # Indicamos que queremos resolver un problema de regresión.
    objective="reg:squarederror",

    # Número de árboles que usará el modelo.
    n_estimators=180,

    # Profundidad máxima de cada árbol.
    max_depth=4,

    # Velocidad de aprendizaje.
    learning_rate=0.08,

    # Porcentaje de filas usadas en cada árbol.
    subsample=0.85,

    # Porcentaje de columnas usadas en cada árbol.
    colsample_bytree=0.9,

    # Semilla para repetir resultados.
    random_state=42,
)

# Entrenamos el modelo de monto.
amount_model.fit(X_train_amount, y_train_amount)
```

### 2. Calcular SHAP global del modelo de monto

```python
# Tomamos el booster interno de XGBoost.
# Usaremos su función nativa pred_contribs=True para obtener aportes tipo SHAP.
amount_booster = amount_model.get_booster()

# Tomamos 200 casos para calcular importancia global.
amount_sample = X_test_amount.sample(200, random_state=42)

# Convertimos la muestra a DMatrix, que es el formato interno de XGBoost.
amount_sample_matrix = xgb.DMatrix(amount_sample, feature_names=AMOUNT_FEATURES)

# Calculamos contribuciones tipo SHAP usando XGBoost directamente.
# El resultado trae una columna por feature y una última columna llamada bias o base value.
amount_contribs = amount_booster.predict(amount_sample_matrix, pred_contribs=True)

# Separamos los aportes de variables.
amount_feature_contribs = amount_contribs[:, :-1]

# Calculamos importancia promedio por variable.
amount_importance = np.abs(amount_feature_contribs).mean(axis=0)

# Construimos el JSON de explicación global del modelo de monto.
amount_global = {
    # Nombre del modelo.
    "model": "amount_xgboost_regressor",

    # Explicación global significa importancia general de variables.
    "explanation_type": "global",

    # Objetivo que explica este modelo.
    "target": "recommended_amount",

    # Variables más importantes ordenadas.
    "top_features": [
        {
            # Nombre de la variable.
            "feature": feature,

            # Importancia promedio de esa variable.
            "importance": round(float(importance), 2),
        }

        # Ordenamos las variables por importancia descendente.
        for feature, importance in sorted(
            zip(AMOUNT_FEATURES, amount_importance),
            key=lambda item: item[1],
            reverse=True,
        )
    ],
}

# Imprimimos el JSON global.
print(json.dumps(amount_global, indent=2))
```

### 3. Guardar explicación global de monto en S3

```python
# Subimos la explicación global del modelo de monto a S3.
s3.put_object(
    # Bucket destino.
    Bucket=BUCKET,

    # Ruta del JSON dentro de S3.
    Key=AMOUNT_EXPLANATION_KEY,

    # Convertimos el diccionario a JSON y luego a bytes.
    Body=json.dumps(amount_global, indent=2).encode("utf-8"),

    # Indicamos que es un archivo JSON.
    ContentType="application/json",
)

# Mostramos la ruta final.
print(f"Uploaded s3://{BUCKET}/{AMOUNT_EXPLANATION_KEY}")
```

### 4. Explicar un caso cualquiera del modelo de monto

Primero explicaremos un caso del conjunto de prueba, igual que hicimos con riesgo.

```python
# Tomamos un caso cualquiera del conjunto de prueba.
amount_case = X_test_amount.iloc[[0]]

# Calculamos el monto recomendado por el modelo.
amount_prediction = amount_model.predict(amount_case)[0]

# Convertimos el caso a DMatrix para XGBoost.
amount_case_matrix = xgb.DMatrix(amount_case, feature_names=AMOUNT_FEATURES)

# Calculamos contribuciones tipo SHAP para este caso.
# La última columna es el valor base; las anteriores son aportes por feature.
amount_case_contribs = amount_booster.predict(amount_case_matrix, pred_contribs=True)[0]

# Extraemos el valor base.
amount_base_value = float(amount_case_contribs[-1])

# Extraemos los aportes de las variables.
amount_case_feature_contribs = amount_case_contribs[:-1]

# Sumamos los aportes de todas las variables.
amount_contributions_sum = float(amount_case_feature_contribs.sum())

# Reconstruimos el monto recomendado con base + aportes.
amount_reconstructed_prediction = amount_base_value + amount_contributions_sum

# Creamos el JSON de explicación local de monto.
amount_local = {
    # Predicción del modelo.
    "amount": {
        "recommended_amount": round(float(amount_prediction), 2),
    },

    # Resumen SHAP para comprobar la suma.
    "shap_summary": {
        "base_value": round(amount_base_value, 2),
        "contributions_sum": round(amount_contributions_sum, 2),
        "reconstructed_amount": round(amount_reconstructed_prediction, 2),
        "model_prediction": round(float(amount_prediction), 2),
    },

    # Aporte de cada variable.
    "amount_explanation": [
        {
            "feature": feature,
            "value": float(amount_case.iloc[0][feature]),
            "contribution": round(float(contribution), 2),
        }
        for feature, contribution in zip(AMOUNT_FEATURES, amount_case_feature_contribs)
    ],
}

# Ordenamos las variables por impacto absoluto.
amount_local["amount_explanation"] = sorted(
    amount_local["amount_explanation"],
    key=lambda item: abs(item["contribution"]),
    reverse=True,
)

# Imprimimos la explicación.
print(json.dumps(amount_local, indent=2))
```

La idea es la misma que en riesgo:

```txt
valor base + aportes de variables = monto recomendado explicado
```

Pero ahora la salida no es una probabilidad. La salida es dinero:

```txt
base_value:              380000
suma de aportes SHAP:     42000
monto reconstruido:      422000
predicción del modelo:   422000
```

### 5. Explicar monto usando un JSON de solicitud

Ahora haremos la explicación de monto con un JSON propio.

La práctica de monto debe poder ejecutarse de forma independiente de la práctica de riesgo. Por eso volvemos a declarar los datos de la solicitud.

```python
# Creamos un payload independiente para la práctica de monto.
amount_application_payload = {
    "features": {
        "application_id": "dd7b3608-14b3-4426-a927-d92ead8aa9de",
        "net_monthly_income": 10400.0,
        "monthly_debt_payment": 2850.0,
        "monthly_expenses": 4500.0,
        "property_value": 700000.0,
        "requested_amount": 500000.0,
        "requested_term_months": 240,
        "estimated_monthly_payment": 2083.33,
        "debt_to_income_ratio": 0.274,
        "loan_to_value_ratio": 0.7143,
        "payment_to_income_ratio": 0.2003,
        "expense_to_income_ratio": 0.4327,
        "total_obligations_to_income_ratio": 0.9071,
        "employment_stability_score": 40,
        "banking_capacity_score": 40,
        "credit_history_score": 76,
        "synthetic_risk_label": 0,
    }
}

# Extraemos el diccionario de features.
amount_application_features = amount_application_payload["features"]

# Guardamos el application_id para identificar la explicación.
amount_application_id = amount_application_features["application_id"]

# Creamos un DataFrame de una sola fila usando las variables del modelo de monto.
real_amount_case = pd.DataFrame([{
    feature: amount_application_features[feature]
    for feature in AMOUNT_FEATURES
}])

# Calculamos el monto recomendado para la solicitud real.
real_amount_prediction = amount_model.predict(real_amount_case)[0]

# Convertimos el caso real a DMatrix.
real_amount_matrix = xgb.DMatrix(real_amount_case, feature_names=AMOUNT_FEATURES)

# Calculamos contribuciones tipo SHAP para el caso real.
real_amount_contribs = amount_booster.predict(real_amount_matrix, pred_contribs=True)[0]

# Extraemos el valor base.
real_amount_base_value = float(real_amount_contribs[-1])

# Extraemos aportes por variable.
real_amount_feature_contribs = real_amount_contribs[:-1]

# Sumamos los aportes de todas las variables.
real_amount_contributions_sum = float(real_amount_feature_contribs.sum())

# Reconstruimos el monto con base + aportes.
real_amount_reconstructed = real_amount_base_value + real_amount_contributions_sum

# Creamos la explicación local de monto para la solicitud real.
real_amount_explanation = {
    # Identificador de la solicitud.
    "application_id": amount_application_id,

    # Predicción del modelo de monto.
    "amount": {
        "requested_amount": float(amount_application_features["requested_amount"]),
        "recommended_amount": round(float(real_amount_prediction), 2),
    },

    # Resumen SHAP.
    "shap_summary": {
        "base_value": round(real_amount_base_value, 2),
        "contributions_sum": round(real_amount_contributions_sum, 2),
        "reconstructed_amount": round(real_amount_reconstructed, 2),
        "model_prediction": round(float(real_amount_prediction), 2),
    },

    # Aportes por variable.
    "amount_explanation": [
        {
            "feature": feature,
            "value": float(real_amount_case.iloc[0][feature]),
            "contribution": round(float(contribution), 2),
        }
        for feature, contribution in zip(AMOUNT_FEATURES, real_amount_feature_contribs)
    ],
}

# Ordenamos por impacto absoluto.
real_amount_explanation["amount_explanation"] = sorted(
    real_amount_explanation["amount_explanation"],
    key=lambda item: abs(item["contribution"]),
    reverse=True,
)

# Imprimimos la explicación de monto.
print(json.dumps(real_amount_explanation, indent=2))
```

También imprimimos una explicación simple en español:

```python
# Tomamos las tres variables con más impacto sobre el monto.
top_amount_reasons = real_amount_explanation["amount_explanation"][:3]

# Imprimimos el resumen de la recomendación.
print(
    f'El modelo recomienda un monto de {real_amount_prediction:,.2f}. '
    f'El monto solicitado era {amount_application_features["requested_amount"]:,.2f}.'
)

# Imprimimos las variables principales.
print("Principales variables que explican el monto recomendado:")

# Recorremos las razones principales.
for index, reason in enumerate(top_amount_reasons, start=1):
    # En regresión, aporte positivo sube el monto recomendado.
    direction = "sube el monto recomendado" if reason["contribution"] > 0 else "baja el monto recomendado"

    # Imprimimos una línea legible.
    print(
        f'{index}. {reason["feature"]}: valor {reason["value"]}, '
        f'aporte {reason["contribution"]:+,.2f} ({direction})'
    )
```

Salida esperada aproximada:

```txt
El modelo recomienda un monto de 390,000.00. El monto solicitado era 500,000.00.
Principales variables que explican el monto recomendado:
1. net_monthly_income: valor 10400.0, aporte -45000.00 (baja el monto recomendado)
2. total_obligations_to_income_ratio: valor 0.9071, aporte -38000.00 (baja el monto recomendado)
3. property_value: valor 700000.0, aporte +25000.00 (sube el monto recomendado)
```

Los números exactos pueden variar según el entrenamiento, pero la lectura debe ser clara: algunas variables empujan el monto recomendado hacia arriba y otras hacia abajo.

---

## Parte práctica C: endpoints NestJS

Variables en `.env`:

```env
EXPLAIN_UMBRELLA_KEY=ml/explanations/umbrella_explanation.json
EXPLAIN_RISK_GLOBAL_KEY=ml/explanations/risk_global_explanation.json
EXPLAIN_AMOUNT_GLOBAL_KEY=ml/explanations/amount_global_explanation.json
EXPLAIN_APPLICATIONS_PREFIX=ml/explanations/applications
PYTHON_EXPLAINER_BIN=python
PYTHON_EXPLAINER_SCRIPT=python-explainer/generate_explanation.py
```

### 1. Qué vamos a construir

En esta versión NestJS usará un pequeño proyecto Python para generar explicaciones.

El flujo será:

```txt
NestJS recibe request
-> NestJS busca features de la aplicación
-> NestJS llama script Python
-> Python lee modelos JSON desde S3
-> Python genera explicación
-> NestJS guarda la explicación local en S3
-> NestJS devuelve el JSON
```

Hay una diferencia clave entre explicación global y local:

```txt
Explicación global:
se genera una vez para el modelo completo.
No necesita application_id.

Explicación local:
se genera para una solicitud específica usando Python.
Sí necesita application_id porque NestJS debe buscar las features de esa solicitud.
```

Entonces, si todavía no existe una solicitud, solo puedes consultar explicaciones globales.

La explicación local se genera después de que el sistema ya creó la solicitud y ya calculó sus features.

Archivos que crearemos o modificaremos:

| Archivo | Qué haremos |
|---------|-------------|
| `python-explainer/requirements.txt` | Librerías Python necesarias |
| `python-explainer/generate_explanation.py` | Script Python que genera explicaciones |
| `src/modulo1/clase09/clase09.controller.ts` | Crear endpoints HTTP para consultar explicaciones |
| `src/modulo1/clase09/clase09.service.ts` | Llamar el script Python y guardar resultados |
| `src/modulo1/modulo1.module.ts` | Registrar controller y service |
| `.env` | Configurar rutas S3 y Python |

### 2. Crear proyecto Python de explicabilidad

Crea esta carpeta dentro del proyecto NestJS:

```txt
python-explainer
```

Dentro crea:

```txt
python-explainer/requirements.txt
python-explainer/generate_explanation.py
```

`requirements.txt`:

```txt
boto3==1.40.0
numpy==2.2.6
```

Este `requirements.txt` es solo para el script que llama NestJS.

No incluye `xgboost` porque ese script no entrena el modelo; solo lee el JSON de árboles generado en Clase 7 y lo recorre manualmente.

En cambio, el notebook sí instala `xgboost` porque ahí entrenamos el modelo de monto para calcular explicabilidad.

Para mantener la clase simple, este script no recalcula SHAP real para XGBoost. Hace dos cosas:

```txt
Riesgo:
usa coeficientes de la regresión logística para explicar aportes locales.

Monto:
usa los caminos de los árboles XGBoost para aproximar qué variables empujan el monto.
```

Esto es suficiente para clase porque permite ver:

```txt
modelo JSON + features -> explicación local -> JSON final
```

### 3. Instalar Python en macOS

Desde la carpeta del proyecto:

```bash
cd solucion
python3 -m venv .venv-explainer
source .venv-explainer/bin/activate
pip install -r python-explainer/requirements.txt
```

Luego en `.env` usa:

```env
PYTHON_EXPLAINER_BIN=.venv-explainer/bin/python
PYTHON_EXPLAINER_SCRIPT=python-explainer/generate_explanation.py
```

Prueba rápida:

```bash
.venv-explainer/bin/python python-explainer/generate_explanation.py --help
```

### 4. Instalar Python en Windows

Desde PowerShell, en la carpeta del proyecto:

```powershell
cd solucion
py -m venv .venv-explainer
.\.venv-explainer\Scripts\Activate.ps1
pip install -r python-explainer\requirements.txt
```

Luego en `.env` usa:

```env
PYTHON_EXPLAINER_BIN=.venv-explainer\Scripts\python.exe
PYTHON_EXPLAINER_SCRIPT=python-explainer\generate_explanation.py
```

Prueba rápida:

```powershell
.\.venv-explainer\Scripts\python.exe python-explainer\generate_explanation.py --help
```

Si PowerShell bloquea el entorno virtual, ejecuta:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 5. Crear el controller de Clase 9

Crea el archivo:

```txt
src/modulo1/clase09/clase09.controller.ts
```

Código:

```ts
import { Controller, Get, Param, Post, UseGuards } from '@nestjs/common';
import { ApiKeyGuard } from '../../auth/guards/api-key.guard';
import { Clase09Service } from './clase09.service';

@Controller('modulo1/clase09')
@UseGuards(ApiKeyGuard)
export class Clase09Controller {
  constructor(private readonly clase09: Clase09Service) {}

  @Get('explanations/umbrella')
  async getUmbrellaExplanation() {
    return await this.clase09.getUmbrellaExplanation();
  }

  @Get('explanations/risk/global')
  async getRiskGlobalExplanation() {
    return await this.clase09.getRiskGlobalExplanation();
  }

  @Get('explanations/amount/global')
  async getAmountGlobalExplanation() {
    return await this.clase09.getAmountGlobalExplanation();
  }

  @Get('explanations/compare')
  async compareExplanations() {
    return await this.clase09.compareExplanations();
  }

  @Get('applications/:applicationId/explanation')
  async getApplicationExplanation(@Param('applicationId') applicationId: string) {
    return await this.clase09.getApplicationExplanation(applicationId);
  }

  @Post('applications/:applicationId/explanation/generate')
  async generateApplicationExplanation(
    @Param('applicationId') applicationId: string,
  ) {
    return await this.clase09.generateApplicationExplanation(applicationId);
  }
}
```

Qué hace cada endpoint:

| Endpoint | Qué devuelve |
|----------|--------------|
| `GET /modulo1/clase09/explanations/umbrella` | explicación del modelo paraguas |
| `GET /modulo1/clase09/explanations/risk/global` | importancia global del modelo de riesgo |
| `GET /modulo1/clase09/explanations/amount/global` | importancia global del modelo de monto |
| `GET /modulo1/clase09/explanations/compare` | comparación de explicaciones globales |
| `GET /modulo1/clase09/applications/:applicationId/explanation` | explicación local combinada |
| `POST /modulo1/clase09/applications/:applicationId/explanation/generate` | genera explicación local llamando Python y la guarda en S3 |

### 6. Crear el service de Clase 9

Crea el archivo:

```txt
src/modulo1/clase09/clase09.service.ts
```

Código:

```ts
import { GetObjectCommand, PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { InjectRepository } from '@nestjs/typeorm';
import { execFile } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { Repository } from 'typeorm';
import { CreditFeatureSet } from '../../entities/credit-feature-set.entity';

const execFileAsync = promisify(execFile);

@Injectable()
export class Clase09Service {
  private readonly s3: S3Client;

  constructor(
    private readonly config: ConfigService,
    @InjectRepository(CreditFeatureSet)
    private readonly featureSets: Repository<CreditFeatureSet>,
  ) {
    this.s3 = new S3Client({
      region: this.config.getOrThrow<string>('AWS_REGION'),
    });
  }

  async getUmbrellaExplanation() {
    return await this.readGlobalWithPython(
      this.config.getOrThrow<string>('EXPLAIN_UMBRELLA_KEY'),
    );
  }

  async getRiskGlobalExplanation() {
    return await this.readGlobalWithPython(
      this.config.getOrThrow<string>('EXPLAIN_RISK_GLOBAL_KEY'),
    );
  }

  async getAmountGlobalExplanation() {
    return await this.readGlobalWithPython(
      this.config.getOrThrow<string>('EXPLAIN_AMOUNT_GLOBAL_KEY'),
    );
  }

  async compareExplanations() {
    return {
      riskModel: await this.getRiskGlobalExplanation(),
      amountModel: await this.getAmountGlobalExplanation(),
    };
  }

  async getApplicationExplanation(applicationId: string) {
    const prefix = this.config.getOrThrow<string>(
      'EXPLAIN_APPLICATIONS_PREFIX',
    );

    return await this.readJson(`${prefix}/${applicationId}.json`);
  }

  async generateApplicationExplanation(applicationId: string) {
    const featureSet = await this.featureSets.findOne({
      where: { applicationId },
    });

    if (!featureSet) {
      throw new NotFoundException('Feature set not found for this application');
    }

    const features = this.buildExplanationFeatures(featureSet);
    const result = await this.runPythonLocalExplanation(applicationId, features);

    const prefix = this.config.getOrThrow<string>(
      'EXPLAIN_APPLICATIONS_PREFIX',
    );
    const key = `${prefix}/${applicationId}.json`;

    await this.s3.send(
      new PutObjectCommand({
        Bucket: this.config.getOrThrow<string>('SAGEMAKER_BUCKET'),
        Key: key,
        Body: JSON.stringify(result, null, 2),
        ContentType: 'application/json',
      }),
    );

    return {
      ...result,
      s3Key: key,
    };
  }

  private async readGlobalWithPython(key: string) {
    return await this.runPython([
      '--mode',
      'read-global',
      '--bucket',
      this.config.getOrThrow<string>('SAGEMAKER_BUCKET'),
      '--key',
      key,
    ]);
  }

  private async runPythonLocalExplanation(
    applicationId: string,
    features: Record<string, number>,
  ) {
    const dir = await mkdtemp(join(tmpdir(), 'clase09-explainer-'));
    const inputPath = join(dir, 'features.json');

    try {
      await writeFile(
        inputPath,
        JSON.stringify({ application_id: applicationId, features }, null, 2),
        'utf-8',
      );

      return await this.runPython([
        '--mode',
        'local',
        '--bucket',
        this.config.getOrThrow<string>('SAGEMAKER_BUCKET'),
        '--input',
        inputPath,
        '--risk-model-key',
        this.config.getOrThrow<string>('SAGEMAKER_RISK_MODEL_PARAMS_KEY'),
        '--amount-model-key',
        this.config.getOrThrow<string>('SAGEMAKER_AMOUNT_MODEL_KEY'),
      ]);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }

  private async runPython(args: string[]) {
    const pythonBin = this.config.get<string>('PYTHON_EXPLAINER_BIN') || 'python';
    const scriptPath =
      this.config.get<string>('PYTHON_EXPLAINER_SCRIPT') ||
      join(process.cwd(), 'python-explainer', 'generate_explanation.py');

    const { stdout } = await execFileAsync(pythonBin, [scriptPath, ...args], {
      maxBuffer: 1024 * 1024 * 10,
      env: process.env,
    });

    return JSON.parse(stdout);
  }

  private buildExplanationFeatures(featureSet: CreditFeatureSet) {
    const payload = featureSet.featuresPayload ?? {};
    return this.pickNumericFeatures({
      ...payload,
      debt_to_income_ratio: featureSet.debtToIncomeRatio,
      loan_to_value_ratio: featureSet.loanToValueRatio,
      payment_to_income_ratio: featureSet.paymentToIncomeRatio,
      expense_to_income_ratio: featureSet.expenseToIncomeRatio,
      total_obligations_to_income_ratio:
        featureSet.totalObligationsToIncomeRatio,
      employment_stability_score: featureSet.employmentStabilityScore,
      banking_capacity_score: featureSet.bankingCapacityScore,
      credit_history_score: featureSet.creditHistoryScore,
    });
  }

  private pickNumericFeatures(source: Record<string, unknown>) {
    const names = [
      'net_monthly_income',
      'monthly_debt_payment',
      'monthly_expenses',
      'property_value',
      'requested_amount',
      'requested_term_months',
      'estimated_monthly_payment',
      'debt_to_income_ratio',
      'loan_to_value_ratio',
      'payment_to_income_ratio',
      'expense_to_income_ratio',
      'total_obligations_to_income_ratio',
      'employment_stability_score',
      'banking_capacity_score',
      'credit_history_score',
    ];

    return Object.fromEntries(
      names.map((name) => [name, this.toNumber(source[name])]),
    );
  }

  private async readJson(key: string) {
    const response = await this.s3.send(
      new GetObjectCommand({
        Bucket: this.config.getOrThrow<string>('SAGEMAKER_BUCKET'),
        Key: key,
      }),
    );

    return JSON.parse(await response.Body!.transformToString());
  }

  private toNumber(value: unknown) {
    const numberValue = Number(value);
    if (!Number.isFinite(numberValue)) {
      throw new NotFoundException('Application has incomplete explanation features');
    }
    return numberValue;
  }
}
```

La idea principal está aquí:

```ts
const result = await this.runPythonLocalExplanation(applicationId, features);
```

NestJS no recalcula la explicación en TypeScript. Le entrega las features al script Python y Python devuelve el JSON.

### 7. Registrar Clase 9 en el módulo

Abre:

```txt
src/modulo1/modulo1.module.ts
```

Agrega imports:

```ts
import { Clase09Controller } from './clase09/clase09.controller';
import { Clase09Service } from './clase09/clase09.service';
```

Agrega el controller:

```ts
controllers: [
  Clase09Controller,
]
```

Agrega el service:

```ts
providers: [
  Clase09Service,
]
```

En el archivo real habrá más clases registradas. No borres las anteriores. Solo agrega Clase 9.

### 8. Probar endpoints

Comparar explicaciones globales:

```bash
curl http://localhost:3000/modulo1/clase09/explanations/compare \
  -H "x-api-key: test1" \
  -H "x-api-secret: pass1"
```

Resultado esperado:

```json
{
  "riskModel": {
    "model": "risk_logistic_regression",
    "explanation_type": "global",
    "target": "default_probability",
    "top_features": []
  },
  "amountModel": {
    "model": "amount_xgboost_regressor",
    "explanation_type": "global",
    "target": "recommended_amount",
    "top_features": []
  }
}
```

Generar explicación local:

```bash
curl -X POST http://localhost:3000/modulo1/clase09/applications/dd7b3608-14b3-4426-a927-d92ead8aa9de/explanation/generate \
  -H "x-api-key: test1" \
  -H "x-api-secret: pass1"
```

Resultado esperado:

```json
{
  "application_id": "dd7b3608-14b3-4426-a927-d92ead8aa9de",
  "risk": {
    "default_probability": 0.382,
    "risk_label": "LOW",
    "threshold": 0.5
  },
  "risk_explanation": [],
  "amount": {
    "requested_amount": 500000,
    "recommended_amount": 390000
  },
  "amount_explanation": [],
  "s3Key": "ml/explanations/applications/dd7b3608-14b3-4426-a927-d92ead8aa9de.json"
}
```

Consultar explicación local:

```bash
curl http://localhost:3000/modulo1/clase09/applications/dd7b3608-14b3-4426-a927-d92ead8aa9de/explanation \
  -H "x-api-key: test1" \
  -H "x-api-secret: pass1"
```

Resultado esperado:

```json
{
  "application_id": "dd7b3608-14b3-4426-a927-d92ead8aa9de",
  "risk": {
    "default_probability": 0.382,
    "risk_label": "LOW"
  },
  "risk_explanation": [],
  "amount": {
    "requested_amount": 500000,
    "recommended_amount": 390000
  },
  "amount_explanation": []
}
```

Si aparece `NoSuchKey`, significa que el JSON local todavía no fue subido a esta ruta:

```txt
ml/explanations/applications/dd7b3608-14b3-4426-a927-d92ead8aa9de.json
```

## Entrega

- JSON de explicación global de riesgo en S3.
- JSON de explicación global de monto en S3.
- JSON de explicación local de una aplicación en S3.
- Resultado de `GET /explanations/compare`.
- Traducción en lenguaje de negocio de una explicación local.
