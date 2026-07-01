# Clase 8: Integración de modelos y primera explicabilidad

| | |
|---|---|
| **Clase** | 8 de 11 |
| **Duración** | 3 horas |
| **Controlador** | `Clase08Controller` |
| **Endpoints** | `POST /modulo1/clase08/credit-files/:applicationId/evaluate`, `POST /modulo1/clase08/models/risk`, `POST /modulo1/clase08/models/amount` |

## Objetivos

Al terminar esta sesión podrás:

- Integrar el modelo de riesgo de Clase 6 con el modelo de monto de Clase 7.
- Mantener el estándar del curso: modelos entrenados en notebook, JSONs en S3 y NestJS usando esos JSONs.
- Crear una evaluación final del expediente crediticio.
- Entender qué significa explicar una predicción.
- Diferenciar predicción, decisión y explicación.
- Hacer un ejercicio manual de explicabilidad con números simples.
- Entrenar y explicar el modelo del paraguas en un notebook de SageMaker.

---

## Parte teórica

### 1. Qué cambia en esta clase

Hasta ahora usamos los modelos por separado.

Clase 6:

```txt
features -> modelo de riesgo -> probabilidad de default
```

Clase 7:

```txt
features -> modelo de monto -> monto recomendado
```

En Clase 8 vamos a unirlos:

```txt
features -> modelo de riesgo
         -> modelo de monto
         -> decision final
```

La idea importante:

```txt
Un modelo responde una pregunta.
Una evaluacion combina varias respuestas.
```

### 2. Predicción no es decisión

El modelo de riesgo predice:

```json
{
  "defaultProbability": 0.72,
  "riskLabel": "HIGH"
}
```

El modelo de monto predice:

```json
{
  "recommendedAmount": 360000
}
```

Pero la decisión final es una regla de negocio que usa ambas salidas.

Ejemplo:

```txt
Si riesgo alto -> revision manual.
Si monto solicitado supera mucho al recomendado -> revisar monto.
Si riesgo bajo y monto razonable -> preaprobacion para revision.
```

En esta clase usaremos decisiones didácticas:

| Decisión | Significado |
|----------|-------------|
| `PRE_APPROVE_FOR_REVIEW` | Caso razonable para continuar revisión |
| `REVIEW_AMOUNT` | El monto solicitado parece alto contra el recomendado |
| `REJECT_OR_MANUAL_REVIEW` | Riesgo alto, requiere revisión estricta |

No estamos creando una política bancaria real. Estamos creando una forma de integrar señales de modelos.

### 3. Qué devuelve el endpoint integrado

Entrada:

```txt
applicationId
```

El backend busca las features en `credit_feature_sets`, llama los modelos ya entrenados y devuelve una evaluación completa:

```json
{
  "applicationId": "APP_ID",
  "risk": {
    "defaultProbability": 0.72,
    "riskLabel": "HIGH"
  },
  "amount": {
    "requestedAmount": 500000,
    "recommendedAmount": 360000
  },
  "decision": "REJECT_OR_MANUAL_REVIEW",
  "reasons": [
    "El modelo de riesgo clasifico la solicitud como HIGH.",
    "La probabilidad de incumplimiento supera el 60%.",
    "El monto solicitado supera en mas de 10% al monto recomendado."
  ]
}
```

### 4. Por qué aparece la explicabilidad

Una vez que el sistema devuelve una decisión, aparece una pregunta natural:

```txt
Por que dijo eso?
```

No basta con responder:

```txt
Porque lo dijo el modelo.
```

Necesitamos explicar qué variables influyeron.

Ejemplo:

```txt
Riesgo alto porque:
- la carga total sobre ingreso es alta;
- el historial crediticio es bajo;
- el pago mensual de deudas es alto.
```

Eso es explicabilidad.

### 5. Explicabilidad global y local

Hay dos formas útiles de explicar modelos:

| Tipo | Pregunta |
|------|----------|
| Explicabilidad global | ¿Qué variables importan más en general? |
| Explicabilidad local | ¿Qué variables influyeron en este caso específico? |

Ejemplo global:

```txt
En general, el modelo de riesgo usa mucho:
1. total_obligations_to_income_ratio
2. debt_to_income_ratio
3. credit_history_score
```

Ejemplo local:

```txt
Para esta solicitud, el riesgo subio principalmente por:
1. deuda alta;
2. carga total alta;
3. historial bajo.
```

### 6. SHAP explicado con una idea simple

SHAP es una técnica para repartir una predicción entre sus variables.

Idea simple:

```txt
Prediccion base + aportes de variables = prediccion final
```

Ejemplo con paraguas:

```txt
Probabilidad base: 0.30
+ lluvia alta:     0.35
+ humedad alta:    0.10
+ nubosidad alta:  0.08
- viento bajo:     0.03
= resultado:       0.80
```

#### ¿De dónde sale la probabilidad base?

La probabilidad base es el punto de partida antes de mirar el caso específico.

En un ejemplo real, esa base normalmente sale del promedio de muchos casos históricos. Por ejemplo:

```txt
En los datos históricos, de cada 100 días parecidos al contexto general,
30 terminaron siendo días donde convenía llevar paraguas.

Entonces usamos:
probabilidad base = 0.30
```

Es decir, `0.30` no significa que ya miramos la humedad, la nubosidad o el viento de este día. Significa:

```txt
Si no supiera nada especial de este día, partiría desde 30%.
```

En el ejercicio manual usamos `0.30` como valor didáctico para que el cálculo sea fácil de seguir. En el notebook, SHAP calcula ese valor base usando datos de referencia del modelo.

#### ¿Cómo definimos si una variable está alta o baja?

Para el ejercicio manual necesitamos reglas simples. Estas reglas no son leyes meteorológicas universales; son rangos didácticos para que todos trabajemos con el mismo criterio.

| Variable | Baja | Media | Alta |
|----------|------|-------|------|
| `rain_probability` | menor a `0.30` | `0.30` a `0.59` | `0.60` o más |
| `humidity` | menor a `0.45` | `0.45` a `0.69` | `0.70` o más |
| `cloudiness` | menor a `0.40` | `0.40` a `0.69` | `0.70` o más |
| `wind_speed` | menor a `10` km/h | `10` a `24` km/h | `25` km/h o más |

Por eso, en el ejemplo:

```txt
rain_probability = 0.85 -> lluvia alta
humidity = 0.78         -> humedad alta
cloudiness = 0.90       -> nubosidad alta
wind_speed = 5          -> viento bajo
```

#### ¿Quién decide el aporte de cada variable?

En el ejercicio manual, nosotros asignamos aportes sencillos para entender la idea:

| Condición | Aporte didáctico |
|-----------|-----------------:|
| `rain_probability` alta | `+0.35` |
| `rain_probability` media | `+0.15` |
| `rain_probability` baja | `-0.10` |
| `humidity` alta | `+0.10` |
| `humidity` media | `+0.04` |
| `humidity` baja | `-0.02` |
| `cloudiness` alta | `+0.08` |
| `cloudiness` media | `+0.03` |
| `cloudiness` baja | `-0.02` |
| `wind_speed` bajo | `-0.03` |
| `wind_speed` medio | `+0.02` |
| `wind_speed` alto | `+0.05` |

Estos aportes son una simulación para aprender. Más adelante, cuando usemos SHAP en el notebook, los aportes no los inventamos nosotros: salen del modelo entrenado y de los datos de referencia.

No necesitamos memorizar la matemática. La idea es:

```txt
Cada variable empuja la prediccion hacia arriba o hacia abajo.
```

---

## Ejercicio manual: explicabilidad en tabla

Trabajen en parejas. Pueden hacerlo en Excel, Google Sheets o una hoja.

Queremos explicar esta predicción:

```txt
Probabilidad de llevar paraguas = 0.80
```

Usaremos una explicación simplificada:

```txt
probabilidad final = probabilidad base + suma de aportes
```

Para este ejercicio asumimos:

```txt
probabilidad base = 0.30
```

Esto representa el promedio histórico inicial: antes de mirar las variables del día, el modelo parte de un 30% de probabilidad de llevar paraguas.

Usaremos estas reglas para clasificar los valores:

| Variable | Baja | Media | Alta |
|----------|------|-------|------|
| `rain_probability` | menor a `0.30` | `0.30` a `0.59` | `0.60` o más |
| `humidity` | menor a `0.45` | `0.45` a `0.69` | `0.70` o más |
| `cloudiness` | menor a `0.40` | `0.40` a `0.69` | `0.70` o más |
| `wind_speed` | menor a `10` km/h | `10` a `24` km/h | `25` km/h o más |

Y estos aportes didácticos:

| Condición | Aporte |
|-----------|-------:|
| Lluvia alta | `+0.35` |
| Humedad alta | `+0.10` |
| Nubosidad alta | `+0.08` |
| Viento bajo | `-0.03` |

Tabla:

| Variable | Valor | Aporte |
|----------|------:|-------:|
| Base | | 0.30 |
| `rain_probability` | 0.85 | +0.35 |
| `humidity` | 0.78 | +0.10 |
| `cloudiness` | 0.90 | +0.08 |
| `wind_speed` | 5 | -0.03 |
| Total | | 0.80 |

Preguntas:

1. ¿Qué variable empujó más la predicción?
2. ¿Qué variable redujo la predicción?
3. Si `rain_probability` bajara de `0.85` a `0.20`, ¿qué esperarías que pase?
4. ¿Cómo explicarías el resultado sin usar la palabra SHAP?

Una buena explicación de negocio sería:

```txt
El modelo recomienda llevar paraguas principalmente porque la probabilidad de lluvia es alta, y esa señal se refuerza con humedad y nubosidad elevadas.
```

---

## Parte práctica A: endpoints integrados en NestJS

El flujo sigue el estándar del curso:

```txt
Notebook entrena -> JSON en S3 -> NestJS lee JSON -> predice
```

Clase 8 no entrena modelos. Solo integra los modelos de Clase 6 y 7.

### 1. Qué vamos a construir

En esta práctica agregaremos una nueva capa de integración.

No vamos a volver a escribir la lógica completa de riesgo ni de monto. Ya existe:

```txt
Clase06Service -> sabe predecir riesgo
Clase07Service -> sabe recomendar monto
```

Clase 8 hará esto:

```txt
Clase08Controller -> recibe requests HTTP
Clase08Service -> llama Clase06Service y Clase07Service
Clase08Service -> combina resultados y devuelve una decisión final
```

Archivos que crearemos o modificaremos:

| Archivo | Qué haremos |
|---------|-------------|
| `src/modulo1/clase08/clase08.controller.ts` | Crear endpoints de clase 8 |
| `src/modulo1/clase08/clase08.service.ts` | Integrar riesgo + monto |
| `src/modulo1/modulo1.module.ts` | Registrar controller y service |
| `.env` | Verificar keys de modelos JSON en S3 |

### 2. Verificar variables de entorno

Clase 8 depende de los JSON creados en las clases anteriores.

En `.env` deben existir estas variables:

```env
SAGEMAKER_RISK_MODEL_PARAMS_KEY=ml/models/risk/risk_model_params.json
SAGEMAKER_AMOUNT_MODEL_KEY=ml/models/amount/amount_model.json
```

También pueden existir estas keys para consultar métricas:

```env
SAGEMAKER_RISK_METRICS_KEY=ml/metrics/risk_metrics.json
SAGEMAKER_AMOUNT_METRICS_KEY=ml/metrics/amount_metrics.json
```

Y para la explicación del paraguas que subiremos al final:

```env
EXPLAIN_UMBRELLA_KEY=ml/explanations/umbrella_explanation.json
```

Si los archivos no existen en S3, los endpoints fallarán con un error parecido a:

```txt
NoSuchKey
```

Eso significa:

```txt
NestJS está buscando el JSON en S3, pero el archivo todavía no fue subido
o la ruta configurada no coincide.
```

### 3. Crear el controller de Clase 8

Crea el archivo:

```txt
src/modulo1/clase08/clase08.controller.ts
```

Código:

```ts
import { Body, Controller, Param, Post, UseGuards } from '@nestjs/common';
import { ApiKeyGuard } from '../../auth/guards/api-key.guard';
import { Clase08Service } from './clase08.service';

@Controller('modulo1/clase08')
@UseGuards(ApiKeyGuard)
export class Clase08Controller {
  constructor(private readonly clase08: Clase08Service) {}

  @Post('credit-files/:applicationId/evaluate')
  async evaluateCreditFile(@Param('applicationId') applicationId: string) {
    return await this.clase08.evaluateCreditFile(applicationId);
  }

  @Post('models/risk')
  async predictRisk(@Body() body: { features: Record<string, number> }) {
    return await this.clase08.predictRisk(body.features);
  }

  @Post('models/amount')
  async predictAmount(@Body() body: { features: Record<string, number> }) {
    return await this.clase08.predictAmount(body.features);
  }
}
```

Qué hace cada endpoint:

| Endpoint | Qué hace |
|----------|----------|
| `POST /credit-files/:applicationId/evaluate` | Busca features por `applicationId`, calcula riesgo, calcula monto y devuelve una decisión |
| `POST /models/risk` | Permite probar el modelo de riesgo enviando features manualmente |
| `POST /models/amount` | Permite probar el modelo de monto enviando features manualmente |

### 4. Crear el service de Clase 8

Crea el archivo:

```txt
src/modulo1/clase08/clase08.service.ts
```

Código:

```ts
import { Injectable } from '@nestjs/common';
import { Clase06Service } from '../clase06/clase06.service';
import { Clase07Service } from '../clase07/clase07.service';

type RiskPrediction = {
  defaultProbability: number;
  riskLabel: 'HIGH' | 'LOW';
  threshold: number;
  modelType: string;
  features: Record<string, number>;
};

type AmountPrediction = {
  recommendedAmount: number;
  modelType: string;
  features: Record<string, number>;
};

@Injectable()
export class Clase08Service {
  constructor(
    private readonly clase06: Clase06Service,
    private readonly clase07: Clase07Service,
  ) {}

  async evaluateCreditFile(applicationId: string) {
    const risk = (await this.clase06.predictApplicationRisk(
      applicationId,
    )) as RiskPrediction;

    const amount = (await this.clase07.recommendApplicationAmount(
      applicationId,
    )) as AmountPrediction;

    const requestedAmount = Number(amount.features.requested_amount ?? 0);

    const decision = this.makeDecision(
      risk.defaultProbability,
      requestedAmount,
      amount.recommendedAmount,
    );

    return {
      applicationId,
      risk: {
        defaultProbability: risk.defaultProbability,
        threshold: risk.threshold,
        riskLabel: risk.riskLabel,
        modelType: risk.modelType,
      },
      amount: {
        requestedAmount,
        recommendedAmount: amount.recommendedAmount,
        modelType: amount.modelType,
      },
      decision,
      reasons: this.buildReasons(
        risk.defaultProbability,
        risk.riskLabel,
        requestedAmount,
        amount.recommendedAmount,
      ),
    };
  }

  async predictRisk(features: Record<string, number>) {
    return await this.clase06.predictRisk({
      debt_to_income_ratio: Number(features.debt_to_income_ratio),
      loan_to_value_ratio: Number(features.loan_to_value_ratio),
      payment_to_income_ratio: Number(features.payment_to_income_ratio),
      expense_to_income_ratio: Number(features.expense_to_income_ratio),
      total_obligations_to_income_ratio: Number(
        features.total_obligations_to_income_ratio,
      ),
      employment_stability_score: Number(features.employment_stability_score),
      banking_capacity_score: Number(features.banking_capacity_score),
      credit_history_score: Number(features.credit_history_score),
    });
  }

  async predictAmount(features: Record<string, number>) {
    return await this.clase07.predictAmount(features);
  }

  private makeDecision(
    defaultProbability: number,
    requestedAmount: number,
    recommendedAmount: number,
  ) {
    if (defaultProbability >= 0.6) {
      return 'REJECT_OR_MANUAL_REVIEW';
    }

    if (requestedAmount > recommendedAmount * 1.1) {
      return 'REVIEW_AMOUNT';
    }

    return 'PRE_APPROVE_FOR_REVIEW';
  }

  private buildReasons(
    defaultProbability: number,
    riskLabel: string,
    requestedAmount: number,
    recommendedAmount: number,
  ) {
    const reasons: string[] = [];

    if (riskLabel === 'HIGH') {
      reasons.push('El modelo de riesgo clasifico la solicitud como HIGH.');
    }

    if (defaultProbability >= 0.6) {
      reasons.push('La probabilidad de incumplimiento supera el 60%.');
    }

    if (requestedAmount > recommendedAmount * 1.1) {
      reasons.push('El monto solicitado supera en mas de 10% al monto recomendado.');
    }

    if (reasons.length === 0) {
      reasons.push('Riesgo y monto recomendado se mantienen dentro de rangos de revision.');
    }

    return reasons;
  }
}
```

La idea más importante está aquí:

```ts
const risk = await this.clase06.predictApplicationRisk(applicationId);
const amount = await this.clase07.recommendApplicationAmount(applicationId);
```

Clase 8 no calcula todo desde cero. Reutiliza las clases anteriores.

Luego toma ambas respuestas y aplica una regla:

```ts
if (defaultProbability >= 0.6) {
  return 'REJECT_OR_MANUAL_REVIEW';
}

if (requestedAmount > recommendedAmount * 1.1) {
  return 'REVIEW_AMOUNT';
}

return 'PRE_APPROVE_FOR_REVIEW';
```

Esta regla es didáctica:

| Regla | Decisión |
|-------|----------|
| Probabilidad de incumplimiento mayor o igual a 60% | `REJECT_OR_MANUAL_REVIEW` |
| Monto solicitado mayor al recomendado por más de 10% | `REVIEW_AMOUNT` |
| Caso dentro de rangos razonables | `PRE_APPROVE_FOR_REVIEW` |

### 5. Registrar Clase 8 en el módulo

Abre:

```txt
src/modulo1/modulo1.module.ts
```

Agrega los imports:

```ts
import { Clase08Controller } from './clase08/clase08.controller';
import { Clase08Service } from './clase08/clase08.service';
```

Agrega el controller:

```ts
controllers: [
  Clase08Controller,
]
```

Agrega el service:

```ts
providers: [
  Clase08Service,
]
```

En el archivo real habrá más controllers y providers de clases anteriores. No los borres. Solo agrega los de Clase 8.

### 6. Endpoint final de evaluación

```txt
POST /modulo1/clase08/credit-files/:applicationId/evaluate
```

Este endpoint:

1. Busca features de Clase 5.
2. Usa el modelo de riesgo de Clase 6.
3. Usa el modelo de monto de Clase 7.
4. Combina resultados.
5. Devuelve decisión y razones.

Prueba:

```bash
curl -X POST http://localhost:3000/modulo1/clase08/credit-files/APPLICATION_ID/evaluate \
  -H "x-api-key: test1" \
  -H "x-api-secret: pass1"
```

Resultado esperado:

```json
{
  "applicationId": "APPLICATION_ID",
  "risk": {
    "defaultProbability": 0.27,
    "threshold": 0.5,
    "riskLabel": "LOW",
    "modelType": "logistic_regression_classifier"
  },
  "amount": {
    "requestedAmount": 500000,
    "recommendedAmount": 430000,
    "modelType": "xgboost_regressor"
  },
  "decision": "REVIEW_AMOUNT",
  "reasons": [
    "El monto solicitado supera en mas de 10% al monto recomendado."
  ]
}
```

Los números pueden cambiar según tus features y tus modelos JSON. Lo importante es que la respuesta tenga:

```txt
risk
amount
decision
reasons
```

### 7. Probar solo riesgo desde Clase 8

```txt
POST /modulo1/clase08/models/risk
```

Ejemplo:

```bash
curl -X POST http://localhost:3000/modulo1/clase08/models/risk \
  -H "Content-Type: application/json" \
  -H "x-api-key: test1" \
  -H "x-api-secret: pass1" \
  -d '{
    "features": {
      "debt_to_income_ratio": 0.62,
      "loan_to_value_ratio": 0.95,
      "payment_to_income_ratio": 0.55,
      "expense_to_income_ratio": 0.65,
      "total_obligations_to_income_ratio": 1.82,
      "employment_stability_score": 35,
      "banking_capacity_score": 30,
      "credit_history_score": 40
    }
  }'
```

Resultado esperado:

```json
{
  "defaultProbability": 0.83,
  "threshold": 0.5,
  "riskLabel": "HIGH",
  "modelType": "logistic_regression_classifier",
  "features": {
    "debt_to_income_ratio": 0.62,
    "loan_to_value_ratio": 0.95,
    "payment_to_income_ratio": 0.55
  }
}
```

Los valores exactos pueden cambiar. En este ejemplo esperamos riesgo alto porque enviamos ratios altos y scores bajos.

### 8. Probar solo monto desde Clase 8

```txt
POST /modulo1/clase08/models/amount
```

Ejemplo:

```bash
curl -X POST http://localhost:3000/modulo1/clase08/models/amount \
  -H "Content-Type: application/json" \
  -H "x-api-key: test1" \
  -H "x-api-secret: pass1" \
  -d '{
    "features": {
      "net_monthly_income": 15000,
      "monthly_debt_payment": 1200,
      "monthly_expenses": 4500,
      "property_value": 900000,
      "requested_amount": 500000,
      "requested_term_months": 240,
      "estimated_monthly_payment": 2083.33,
      "debt_to_income_ratio": 0.08,
      "loan_to_value_ratio": 0.56,
      "payment_to_income_ratio": 0.14,
      "expense_to_income_ratio": 0.30,
      "total_obligations_to_income_ratio": 0.52,
      "employment_stability_score": 100,
      "banking_capacity_score": 90,
      "credit_history_score": 95
    }
  }'
```

Resultado esperado:

```json
{
  "recommendedAmount": 420000,
  "modelType": "xgboost_regressor",
  "features": {
    "net_monthly_income": 15000,
    "requested_amount": 500000,
    "property_value": 900000
  }
}
```

El monto exacto puede cambiar según el modelo entrenado. Lo importante es que NestJS ya no entrena: solo lee el JSON de S3 y calcula la predicción.

### 9. Resultado final esperado de la parte NestJS

Al terminar esta parte deberías tener:

```txt
1. Clase08Controller creado.
2. Clase08Service creado.
3. Clase08Controller registrado en Modulo1Module.
4. Clase08Service registrado en Modulo1Module.
5. Endpoint integrado funcionando.
6. Endpoint de riesgo funcionando desde Clase 8.
7. Endpoint de monto funcionando desde Clase 8.
```

Y deberías poder explicar esta frase:

```txt
Clase 8 no crea modelos nuevos.
Clase 8 combina modelos ya entrenados para producir una evaluación final.
```

---

## Parte práctica B: explicabilidad del paraguas en SageMaker

Usaremos el ejemplo del paraguas porque es simple y fácil de explicar.

### 1. Instala dependencias

```python
# Instalamos SHAP para explicar el modelo.
# También instalamos scikit-learn, pandas, numpy y matplotlib porque los usaremos en el notebook.
%pip install --quiet shap scikit-learn pandas numpy matplotlib
```

### 2. Crea dataset y entrena el modelo

```python
# json nos permite convertir diccionarios de Python a texto JSON.
import json

# boto3 nos permite conectarnos a servicios de AWS, por ejemplo S3.
import boto3

# numpy nos ayuda a crear datos numéricos y trabajar con cálculos matemáticos.
import numpy as np

# pandas nos permite crear tablas tipo Excel dentro de Python.
import pandas as pd

# shap es la librería que usaremos para explicar la predicción del modelo.
import shap

# LogisticRegression es el modelo de clasificación que aprenderá si conviene llevar paraguas.
from sklearn.linear_model import LogisticRegression

# Pipeline permite unir pasos: primero preparar datos, luego entrenar el modelo.
from sklearn.pipeline import Pipeline

# StandardScaler normaliza las variables para que estén en una escala comparable.
from sklearn.preprocessing import StandardScaler

# Lista de columnas que usará el modelo como variables de entrada.
FEATURES = [
    "rain_probability",
    "humidity",
    "cloudiness",
    "wind_speed",
    "is_rainy_season",
]


# Esta función evita que un valor se salga de un rango.
# Por ejemplo, una probabilidad no debería ser menor que 0 ni mayor que 1.
def clip(values, low, high):
    return np.minimum(np.maximum(values, low), high)


# Esta función crea datos históricos sintéticos para entrenar el modelo.
# rows indica cuántos registros queremos crear.
# seed fija la aleatoriedad para que todos obtengan resultados parecidos.
def make_umbrella_dataset(rows=800, seed=7):
    # Creamos un generador de números aleatorios controlado por la semilla.
    rng = np.random.default_rng(seed)

    # Creamos una probabilidad de lluvia entre 0 y 1.
    rain_probability = clip(rng.beta(2.0, 2.0, rows), 0, 1)

    # Creamos humedad correlacionada con lluvia: si sube la lluvia, suele subir la humedad.
    humidity = clip(0.35 + rain_probability * 0.45 + rng.normal(0, 0.12, rows), 0, 1)

    # Creamos nubosidad correlacionada con lluvia: si sube la lluvia, suele subir la nubosidad.
    cloudiness = clip(0.25 + rain_probability * 0.55 + rng.normal(0, 0.14, rows), 0, 1)

    # Creamos velocidad de viento en km/h con valores realistas para el ejemplo.
    wind_speed = clip(rng.normal(18, 8, rows), 0, 55)

    # Creamos una variable 0/1 que indica si estamos en temporada de lluvia.
    is_rainy_season = rng.binomial(1, clip(0.25 + rain_probability * 0.5, 0.05, 0.9))

    # Creamos una señal interna: mientras más alta la lluvia, más aumenta la decisión de llevar paraguas.
    # Le damos más peso a rain_probability para que el modelo aprenda que es la variable más importante.
    signal = (
        5.0 * rain_probability
        + 0.8 * humidity
        + 0.6 * cloudiness
        + 0.015 * wind_speed
        + 0.5 * is_rainy_season
        + rng.normal(0, 0.45, rows)
        - 2.7
    )

    # Convertimos la señal en una probabilidad entre 0 y 1 usando la función logística.
    umbrella_probability = 1 / (1 + np.exp(-signal))

    # Generamos la etiqueta real: 1 significa llevar paraguas, 0 significa no llevar paraguas.
    take_umbrella = rng.binomial(1, umbrella_probability)

    # Devolvemos todos los datos como una tabla de pandas.
    return pd.DataFrame({
        "rain_probability": np.round(rain_probability, 3),
        "humidity": np.round(humidity, 3),
        "cloudiness": np.round(cloudiness, 3),
        "wind_speed": np.round(wind_speed, 1),
        "is_rainy_season": is_rainy_season,
        "take_umbrella": take_umbrella,
    })


# Creamos el dataset sintético con 800 filas.
df = make_umbrella_dataset()

# Creamos un pipeline con dos pasos:
# 1. scaler: normaliza las variables.
# 2. logistic: entrena la regresión logística.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000)),
])

# Entrenamos el modelo usando las variables FEATURES para predecir take_umbrella.
model.fit(df[FEATURES], df["take_umbrella"])

# Mostramos las primeras filas para revisar que la tabla se creó correctamente.
df.head()
```

### 3. Probar una predicción

```python
# Creamos un caso nuevo que queremos evaluar.
case = pd.DataFrame([{
    "rain_probability": 0.85,
    "humidity": 0.78,
    "cloudiness": 0.90,
    "wind_speed": 5,
    "is_rainy_season": 1,
}])

# predict_proba devuelve dos probabilidades:
# posición 0 -> probabilidad de clase 0, no llevar paraguas.
# posición 1 -> probabilidad de clase 1, llevar paraguas.
probability = model.predict_proba(case)[0][1]

# Imprimimos la probabilidad de llevar paraguas con 4 decimales.
print("Umbrella probability:", round(float(probability), 4))
```

### 4. Explicar con SHAP

```python
# Tomamos 100 filas históricas como datos de referencia.
# SHAP usa este background para calcular cuál es la predicción base del modelo.
background = df[FEATURES].sample(100, random_state=42)

# Creamos el explicador de SHAP.
# Le pasamos model.predict_proba porque queremos explicar probabilidades.
# Le pasamos background para que SHAP tenga un punto de comparación.
explainer = shap.Explainer(model.predict_proba, background)

# Calculamos los valores SHAP del caso específico que queremos explicar.
# Aquí SHAP responde: cuánto aportó cada variable a esta predicción.
shap_values = explainer(case)

# Dibujamos el gráfico tipo waterfall.
# shap_values[0, :, 1] significa:
# 0 -> primer caso analizado.
# : -> todas las variables.
# 1 -> clase positiva, es decir, llevar paraguas.
shap.plots.waterfall(shap_values[0, :, 1])
```

### 5. Crear JSON de explicación para S3

```python
# Extraemos los aportes SHAP de cada variable para la clase positiva.
# Cada valor indica cuánto empujó esa variable la predicción hacia arriba o hacia abajo.
values = shap_values[0, :, 1].values

# Extraemos el valor base calculado por SHAP.
# Este es el punto de partida antes de sumar los aportes del caso actual.
base_value = shap_values[0, :, 1].base_values

# Construimos un diccionario que luego guardaremos como JSON.
explanation = {
    # Nombre descriptivo del modelo que estamos explicando.
    "model": "umbrella_logistic_regression",

    # Guardamos la predicción principal del caso.
    "prediction": {
        "take_umbrella_probability": round(float(probability), 4)
    },

    # Guardamos el valor base de SHAP.
    "base_value": round(float(base_value), 4),

    # Creamos una lista con el aporte de cada variable.
    "feature_contributions": [
        {
            # Nombre de la variable.
            "feature": feature,

            # Valor real que tenía esa variable en el caso evaluado.
            "value": float(case.iloc[0][feature]),

            # Aporte SHAP de esa variable.
            # Positivo significa que empujó hacia llevar paraguas.
            # Negativo significa que empujó hacia no llevar paraguas.
            "contribution": round(float(contribution), 4),
        }

        # Recorremos al mismo tiempo la lista de variables y sus aportes SHAP.
        for feature, contribution in zip(FEATURES, values)
    ],
}

# Ordenamos las variables desde la más influyente hasta la menos influyente.
# Usamos abs porque nos importa el tamaño del impacto, sea positivo o negativo.
explanation["feature_contributions"] = sorted(
    explanation["feature_contributions"],
    key=lambda item: abs(item["contribution"]),
    reverse=True,
)

# Imprimimos el JSON ordenado para revisarlo antes de subirlo a S3.
print(json.dumps(explanation, indent=2))
```

### 6. Generar una explicación final en español

Ahora convertiremos los aportes técnicos de SHAP en una explicación que una persona pueda entender.

```python
# Nombres en español para mostrar una explicación más amigable.
FEATURE_NAMES_ES = {
    "rain_probability": "probabilidad de lluvia",
    "humidity": "humedad",
    "cloudiness": "nubosidad",
    "wind_speed": "velocidad del viento",
    "is_rainy_season": "temporada de lluvia",
}


# Esta función convierte el valor numérico de cada variable en una etiqueta simple.
# Por ejemplo: 0.85 de rain_probability se mostrará como "alta".
def describe_feature_level(feature, value):
    if feature == "rain_probability":
        if value >= 0.60:
            return "alta"
        if value >= 0.30:
            return "media"
        return "baja"

    if feature == "humidity":
        if value >= 0.70:
            return "alta"
        if value >= 0.45:
            return "media"
        return "baja"

    if feature == "cloudiness":
        if value >= 0.70:
            return "alta"
        if value >= 0.40:
            return "media"
        return "baja"

    if feature == "wind_speed":
        if value >= 25:
            return "alta"
        if value >= 10:
            return "media"
        return "baja"

    if feature == "is_rainy_season":
        return "sí" if value == 1 else "no"

    return "sin clasificar"


# Filtramos solo las variables que empujaron la predicción hacia llevar paraguas.
positive_contributions = [
    item
    for item in explanation["feature_contributions"]
    if item["contribution"] > 0
]

# Tomamos las 3 variables positivas más importantes.
top_reasons = positive_contributions[:3]

# Creamos una etiqueta final según la probabilidad calculada por el modelo.
prediction_label = "llevar paraguas" if probability >= 0.5 else "no llevar paraguas"

# Imprimimos una explicación final en español.
print(f'El modelo predice "{prediction_label}" con probabilidad {probability:.2%}.')
print("Las principales razones son:")

for index, reason in enumerate(top_reasons, start=1):
    feature = reason["feature"]
    value = reason["value"]
    contribution = reason["contribution"]
    feature_name_es = FEATURE_NAMES_ES.get(feature, feature)
    level = describe_feature_level(feature, value)

    print(
        f"{index}. {feature_name_es} {level} "
        f"(valor: {value}, aporte SHAP: {contribution:+.4f})"
    )

# Guardamos también esta explicación dentro del JSON.
# Así el archivo de S3 no solo tiene números, también tiene una explicación lista para mostrar.
explanation["natural_language_explanation"] = {
    "prediction_label": prediction_label,
    "summary": (
        f'El modelo predice "{prediction_label}" con probabilidad {probability:.2%}.'
    ),
    "main_reasons": [
        {
            "feature": reason["feature"],
            "feature_name_es": FEATURE_NAMES_ES.get(reason["feature"], reason["feature"]),
            "value": reason["value"],
            "level": describe_feature_level(reason["feature"], reason["value"]),
            "contribution": reason["contribution"],
        }
        for reason in top_reasons
    ],
}
```

Salida esperada aproximada:

```txt
El modelo predice "llevar paraguas" con probabilidad 85.20%.
Las principales razones son:
1. probabilidad de lluvia alta (valor: 0.85, aporte SHAP: +0.3120)
2. humedad alta (valor: 0.78, aporte SHAP: +0.0710)
3. nubosidad alta (valor: 0.9, aporte SHAP: +0.0520)
```

Los números exactos pueden cambiar un poco, pero la idea debería mantenerse: la probabilidad de lluvia debería aparecer como una de las variables más importantes.

### 7. Subir explicación a S3

```python
# Nombre del bucket S3 donde guardaremos el archivo.
# Cambia este valor por el bucket de tu cuenta.
BUCKET = "docente-980921750553-us-east-1-an"

# Ruta interna del archivo dentro del bucket.
EXPLAIN_UMBRELLA_KEY = "ml/explanations/umbrella_explanation.json"

# Creamos el cliente de S3 usando boto3.
s3 = boto3.client("s3")

# Subimos el JSON de explicación a S3.
s3.put_object(
    # Bucket destino.
    Bucket=BUCKET,

    # Key o ruta destino dentro del bucket.
    Key=EXPLAIN_UMBRELLA_KEY,

    # Convertimos el diccionario explanation a texto JSON y luego a bytes.
    Body=json.dumps(explanation, indent=2).encode("utf-8"),

    # Indicamos que el archivo es JSON.
    ContentType="application/json",
)

# Mostramos la ruta completa para confirmar dónde quedó guardado.
print(f"Uploaded s3://{BUCKET}/{EXPLAIN_UMBRELLA_KEY}")
```

Este JSON será usado en Clase 9 para consultar explicaciones desde NestJS.

### 8. Resultado esperado de la parte SageMaker

Al terminar el notebook deberías tener:

```txt
1. Dataset sintético de paraguas creado.
2. Modelo de regresión logística entrenado.
3. Predicción para un caso nuevo.
4. Gráfico waterfall de SHAP.
5. JSON de explicación creado.
6. Explicación final en español impresa.
7. Archivo umbrella_explanation.json subido a S3.
```

El JSON generado debería verse parecido a esto:

```json
{
  "model": "umbrella_logistic_regression",
  "prediction": {
    "take_umbrella_probability": 0.852
  },
  "base_value": 0.31,
  "feature_contributions": [
    {
      "feature": "rain_probability",
      "value": 0.85,
      "contribution": 0.312
    },
    {
      "feature": "humidity",
      "value": 0.78,
      "contribution": 0.071
    },
    {
      "feature": "cloudiness",
      "value": 0.9,
      "contribution": 0.052
    }
  ],
  "natural_language_explanation": {
    "prediction_label": "llevar paraguas",
    "summary": "El modelo predice \"llevar paraguas\" con probabilidad 85.20%.",
    "main_reasons": [
      {
        "feature": "rain_probability",
        "feature_name_es": "probabilidad de lluvia",
        "value": 0.85,
        "level": "alta",
        "contribution": 0.312
      }
    ]
  }
}
```

Los valores exactos pueden variar, pero deberías revisar tres cosas:

```txt
1. Que rain_probability aparezca entre las variables más importantes.
2. Que base_value exista.
3. Que natural_language_explanation exista.
```

## Resultado final esperado de la clase

Al terminar Clase 8 debes poder mostrar:

| Parte | Resultado esperado |
|-------|--------------------|
| NestJS | Endpoint integrado `evaluate` funcionando |
| NestJS | Endpoint manual de riesgo funcionando |
| NestJS | Endpoint manual de monto funcionando |
| SageMaker | Modelo del paraguas entrenado |
| SageMaker | Waterfall plot de SHAP generado |
| SageMaker | JSON `umbrella_explanation.json` subido a S3 |
| Conceptual | Diferencia clara entre predicción, decisión y explicación |

La frase de cierre de la clase es:

```txt
Un modelo predice.
Una regla de negocio decide.
La explicabilidad ayuda a entender por qué el modelo predijo eso.
```

## Entrega

- Resultado del endpoint integrado `evaluate`.
- Una tabla manual de explicabilidad del paraguas.
- Captura del waterfall plot de SHAP.
- `umbrella_explanation.json` subido a S3.
- Explicación corta: "por qué el modelo recomendó llevar paraguas".
