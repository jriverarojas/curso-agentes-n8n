# Clase 10: Artefactos de explicabilidad e integración final

| | |
|---|---|
| **Clase** | 10 de 11 |
| **Duración** | 3 horas |
| **Controlador** | `Clase10Controller` |
| **Pantalla** | `frontend` en `http://localhost:5173` |
| **Objetivo** | Preparar modelos para explicabilidad y unir el flujo final |

## Objetivos

Al terminar esta sesión podrás:

- Entender por qué un modelo no debe guardarse solo para predecir.
- Regenerar artefactos de riesgo necesarios para explicabilidad.
- Regenerar artefactos de monto necesarios para explicabilidad.
- Actualizar el script Python para generar explicaciones locales desde NestJS.
- Crear una solicitud de crédito desde una interfaz.
- Subir documentos desde el navegador para guardarlos en S3.
- Procesar documentos con el flujo ya creado en clases anteriores.
- Generar features.
- Ejecutar análisis de riesgo.
- Ejecutar análisis de monto recomendado.
- Consultar explicaciones de riesgo y monto.
- Entender cómo una interfaz conversa con una API NestJS.

Esta clase es práctica. Empezaremos preparando artefactos de explicabilidad y luego usaremos esos artefactos en el flujo final.

---

## Parte A: preparar modelos para explicabilidad

En clases anteriores guardamos modelos pensando principalmente en predicción.

Ahora necesitamos guardar también artefactos para explicar.

La idea central:

```txt
Un modelo no debería guardarse solo para predecir.
También debemos guardar lo necesario para explicar sus resultados.
```

### 1. Qué nos faltaba

Para riesgo teníamos:

```txt
risk_model_params.json
```

Ese archivo sirve para:

```txt
predicción desde NestJS
explicación por coeficientes
```

Pero para usar SHAP desde Python necesitamos además un conjunto de referencia:

```txt
risk_background.json
```

Para monto teníamos:

```txt
amount_model.json
```

Ese archivo sirve para que NestJS recorra árboles y prediga.

Pero para usar contribuciones nativas de XGBoost necesitamos también:

```txt
amount_xgboost_model.json
```

Resumen:

| Modelo | Ya teníamos | Agregaremos en Clase 10 | Para qué sirve |
|--------|-------------|--------------------------|----------------|
| Riesgo | `risk_model_params.json` | `risk_background.json` | SHAP necesita referencia histórica |
| Monto | `amount_model.json` | `amount_xgboost_model.json` | XGBoost puede calcular `pred_contribs=True` |

---

## Parte práctica A: regenerar artefactos en notebook

Abre tu notebook de SageMaker o Jupyter local y ejecuta estas celdas.

### 1. Instalar dependencias

```python
%pip install --quiet scikit-learn pandas numpy boto3 xgboost
```

### 2. Cargar dataset desde S3

```python
import io
import json

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BUCKET = "docente-980921750553-us-east-1-an"
CSV_KEY = "synthetic_mortgage_dataset.csv"

RISK_MODEL_PARAMS_KEY = "ml/models/risk/risk_model_params.json"
RISK_BACKGROUND_KEY = "ml/explanations/risk_background.json"

AMOUNT_NATIVE_MODEL_KEY = "ml/models/amount/amount_xgboost_model.json"

s3 = boto3.client("s3")
response = s3.get_object(Bucket=BUCKET, Key=CSV_KEY)
df = pd.read_csv(io.BytesIO(response["Body"].read()))
df.head()
```

### 3. Regenerar artefactos de riesgo

```python
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

X_risk = df[RISK_FEATURES]
y_risk = df["default_flag"]

X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
    X_risk,
    y_risk,
    test_size=0.2,
    random_state=42,
    stratify=y_risk,
)

risk_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(class_weight="balanced", max_iter=1000)),
])

risk_model.fit(X_train_risk, y_train_risk)

scaler = risk_model.named_steps["scaler"]
logistic = risk_model.named_steps["logistic"]

risk_model_params = {
    "model_type": "logistic_regression_classifier",
    "target": "default_flag",
    "features": RISK_FEATURES,
    "threshold": 0.5,
    "scaler": {
        "mean": dict(zip(RISK_FEATURES, scaler.mean_.tolist())),
        "scale": dict(zip(RISK_FEATURES, scaler.scale_.tolist())),
    },
    "coefficients": dict(zip(RISK_FEATURES, logistic.coef_[0].tolist())),
    "intercept": float(logistic.intercept_[0]),
}

risk_background = (
    X_train_risk
    .sample(100, random_state=42)
    .round(6)
    .to_dict(orient="records")
)

s3.put_object(
    Bucket=BUCKET,
    Key=RISK_MODEL_PARAMS_KEY,
    Body=json.dumps(risk_model_params, indent=2).encode("utf-8"),
    ContentType="application/json",
)

s3.put_object(
    Bucket=BUCKET,
    Key=RISK_BACKGROUND_KEY,
    Body=json.dumps(risk_background, indent=2).encode("utf-8"),
    ContentType="application/json",
)

print(f"Uploaded s3://{BUCKET}/{RISK_MODEL_PARAMS_KEY}")
print(f"Uploaded s3://{BUCKET}/{RISK_BACKGROUND_KEY}")
```

Qué hicimos:

```txt
risk_model_params.json -> permite predecir riesgo
risk_background.json   -> permite explicar riesgo con SHAP desde Python
```

### 4. Regenerar artefacto nativo de XGBoost para monto

```python
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

local_native_model_path = "amount_xgboost_model.json"
amount_model.save_model(local_native_model_path)

s3.upload_file(
    local_native_model_path,
    BUCKET,
    AMOUNT_NATIVE_MODEL_KEY,
)

print(f"Uploaded s3://{BUCKET}/{AMOUNT_NATIVE_MODEL_KEY}")
```

Qué hicimos:

```txt
amount_xgboost_model.json -> permite usar XGBoost nativo desde Python
```

Con ese archivo, nuestro script Python puede hacer:

```python
booster.predict(dmatrix, pred_contribs=True)
```

Eso nos da contribuciones tipo SHAP para el monto recomendado.

---

## Parte práctica B: actualizar Python explainer

En `esqueleto/python-explainer/requirements.txt` dejaremos:

```txt
boto3==1.40.0
numpy==2.2.6
pandas==2.3.0
shap==0.48.0
xgboost==3.0.2
```

Instala dependencias:

macOS:

```bash
cd esqueleto
python3 -m venv .venv-explainer
source .venv-explainer/bin/activate
pip install -r python-explainer/requirements.txt
```

Windows:

```powershell
cd esqueleto
py -m venv .venv-explainer
.\.venv-explainer\Scripts\Activate.ps1
pip install -r python-explainer\requirements.txt
```

El script ahora puede trabajar en dos modos:

| Si existe el artefacto | Método usado |
|------------------------|--------------|
| `EXPLAIN_RISK_BACKGROUND_KEY` | SHAP con background para riesgo |
| `SAGEMAKER_AMOUNT_NATIVE_MODEL_KEY` | `pred_contribs=True` nativo de XGBoost para monto |
| Si no existen | fallback a explicación aproximada |

Configura `.env`:

```env
EXPLAIN_RISK_BACKGROUND_KEY=ml/explanations/risk_background.json
SAGEMAKER_AMOUNT_NATIVE_MODEL_KEY=ml/models/amount/amount_xgboost_model.json
PYTHON_EXPLAINER_BIN=.venv-explainer/bin/python
PYTHON_EXPLAINER_SCRIPT=python-explainer/generate_explanation.py
```

En Windows:

```env
PYTHON_EXPLAINER_BIN=.venv-explainer\Scripts\python.exe
PYTHON_EXPLAINER_SCRIPT=python-explainer\generate_explanation.py
```

---

## Flujo final de interfaz

La interfaz seguirá este orden:

```txt
1. Crear nueva credit application
2. Subir documentos a S3
3. Procesar documentos y features
4. Analizar riesgo
5. Analizar monto
6. Pedir explicacion de riesgo
7. Pedir explicacion de monto
```

Visualmente será algo así:

```txt
[1] Aplicacion  [2] Documentos  [3] Features  [4] Riesgo  [5] Monto  [6] Exp. riesgo  [7] Exp. monto
```

Cada botón llama a un endpoint distinto.

---

## Antes de empezar

Esta clase asume que ya existen y funcionan:

| Clase | Qué necesitamos |
|-------|-----------------|
| Clase 3 | Procesamiento Textract |
| Clase 4 | Limpieza con Glue |
| Clase 5 | Feature engineering |
| Clase 6 | Modelo de riesgo desde JSON en S3 |
| Clase 7 | Modelo de monto desde JSON en S3 |
| Clase 8 | Evaluación integrada |
| Clase 9 | Explicaciones en JSON desde S3 |

En esta clase no volvemos a explicar esos modelos. Los usamos.

---

## Endpoints que construiremos

Todos estarán bajo:

```txt
/modulo1/clase10
```

| Método | Endpoint | Qué hace |
|--------|----------|----------|
| `POST` | `/applications` | Crea una solicitud nueva |
| `GET` | `/applications` | Lista solicitudes con `applicationId`, nombre, estado actual y metadatos |
| `GET` | `/document-types` | Lista documentos activos configurados en la base de datos |
| `GET` | `/applications/:applicationId` | Consulta solicitud, documentos y features |
| `POST` | `/applications/:applicationId/documents` | Sube un documento a S3 y lo registra |
| `POST` | `/applications/:applicationId/process-documents` | Ejecuta Textract |
| `POST` | `/applications/:applicationId/clean` | Inicia limpieza |
| `GET` | `/applications/:applicationId/clean-status` | Consulta estado de limpieza |
| `POST` | `/applications/:applicationId/features` | Inicia feature engineering |
| `GET` | `/applications/:applicationId/features-status` | Consulta estado de features |
| `GET` | `/applications/:applicationId/features` | Lee features generadas |
| `POST` | `/applications/:applicationId/risk` | Analiza riesgo |
| `POST` | `/applications/:applicationId/amount` | Analiza monto |
| `POST` | `/applications/:applicationId/evaluate` | Evalúa riesgo + monto |
| `GET` | `/applications/:applicationId/risk-explanation` | Lee explicación de riesgo |
| `GET` | `/applications/:applicationId/amount-explanation` | Lee explicación de monto |
| `GET` | `/applications/:applicationId/explanation` | Lee explicación completa |

---

## Paso 1: crear `Clase10Service`

Archivo:

```txt
src/modulo1/clase10/clase10.service.ts
```

Este servicio será el orquestador.

No queremos reescribir toda la lógica de las clases anteriores. Queremos reutilizarla:

```txt
Clase10Service
├── usa Clase03Service para Textract
├── usa Clase04Service para limpieza
├── usa Clase05Service para features
├── usa Clase06Service para riesgo
├── usa Clase07Service para monto
├── usa Clase08Service para evaluación integrada
└── usa Clase09Service para explicaciones
```

Importaciones principales:

```typescript
import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { InjectRepository } from '@nestjs/typeorm';
import { In, Repository } from 'typeorm';
```

Para qué sirve cada una:

| Import | Para qué sirve |
|--------|----------------|
| `PutObjectCommand` | subir archivos a S3 |
| `S3Client` | cliente AWS S3 |
| `BadRequestException` | responder errores de datos inválidos |
| `NotFoundException` | responder cuando no existe una solicitud |
| `ConfigService` | leer `.env` |
| `InjectRepository` | usar repositorios TypeORM |
| `Repository` | leer/escribir entidades en BD |

### Estados de la solicitud

Cada solicitud vive en `credit_applications` y tiene un campo `status`.

Ese campo permite que el listado muestre exactamente en qué parte del flujo está cada aplicación:

| Estado técnico | Texto para el usuario | Cuándo ocurre |
|----------------|-----------------------|---------------|
| `DRAFT` | Borrador | La solicitud fue creada, pero todavía no tiene documentos |
| `DOCUMENTS_UPLOADED` | Archivos cargados | Ya se subió al menos un documento a S3 |
| `DOCUMENTS_PROCESSED` | Documentos procesados | Textract ya leyó los documentos |
| `CLEANING_STARTED` | Limpieza iniciada | Se lanzó el job de limpieza en Glue |
| `CLEAN_COMPLETED` | Información limpia | Ya existe el perfil limpio del cliente |
| `FEATURES_STARTED` | Features en proceso | Se lanzó el job para crear variables |
| `FEATURES_COMPLETED` | Variables listas | Las features ya están listas para los modelos |
| `RISK_ANALYZED` | Riesgo analizado | Ya se ejecutó el modelo de riesgo |
| `AMOUNT_ANALYZED` | Monto analizado | Ya se ejecutó el modelo de monto |
| `EVALUATED` | Evaluación integrada | Ya se combinó riesgo + monto |
| `EXPLANATION_GENERATED` | Explicación generada | Ya se consultó/generó la explicación |

Esto es importante porque el frontend no debe mostrar un estado genérico como “completado”.

Debe mostrar el paso real del proceso.

### Crear solicitud

Método:

```typescript
async createApplication(body: {
  applicantExternalId?: string;
  applicantName?: string;
}) {
  const application = await this.applications.save(
    this.applications.create({
      applicantExternalId: body.applicantExternalId,
      applicantName: body.applicantName,
      status: 'DRAFT',
    }),
  );

  return {
    applicationId: application.id,
    status: application.status,
    statusLabel: this.statusMeta(application.status).label,
    application,
  };
}
```

Qué hace:

1. Crea un registro en `credit_applications`.
2. Guarda nombre e ID externo opcional.
3. Devuelve el `applicationId`.

Este `applicationId` será usado por todos los pasos siguientes.

---

### Listar solicitudes

Endpoint:

```txt
GET /modulo1/clase10/applications
```

Este endpoint es el que usa la primera pantalla del frontend.

Debe devolver:

| Campo | Para qué sirve |
|-------|----------------|
| `applicationId` | ID real que usaremos en todos los endpoints |
| `applicantName` | nombre visible del solicitante |
| `applicantExternalId` | código externo opcional |
| `status` | estado técnico guardado en BD |
| `statusLabel` | texto entendible para el usuario |
| `statusDescription` | explicación corta del estado |
| `documentsCount` | cantidad de documentos subidos |
| `hasFeatures` | indica si ya existen variables para el modelo |
| `createdAt` | fecha de creación |
| `updatedAt` | última actualización |

Ejemplo de respuesta:

```json
{
  "total": 1,
  "items": [
    {
      "applicationId": "dd7b3608-14b3-4426-a927-d92ead8aa9de",
      "applicantName": "Cliente demo",
      "applicantExternalId": "SOL-DEMO-001",
      "status": "FEATURES_COMPLETED",
      "statusLabel": "Variables listas",
      "statusDescription": "Las features ya están guardadas y listas para los modelos.",
      "documentsCount": 5,
      "hasFeatures": true,
      "createdAt": "2026-07-08T10:00:00.000Z",
      "updatedAt": "2026-07-08T10:20:00.000Z"
    }
  ]
}
```

En el service lo construimos leyendo:

```txt
credit_applications
application_documents
credit_feature_sets
```

Así el listado no depende de datos inventados del frontend.

Código del service:

```typescript
async listApplications() {
  const [applications, total] = await this.applications.findAndCount({
    order: { updatedAt: 'DESC' },
    take: 100,
  });

  const applicationIds = applications.map((application) => application.id);
  const [documents, features] = applicationIds.length
    ? await Promise.all([
        this.documents.find({ where: { applicationId: In(applicationIds) } }),
        this.featureSets.find({ where: { applicationId: In(applicationIds) } }),
      ])
    : [[], []];

  const documentCountByApplication = new Map<string, number>();
  for (const document of documents) {
    documentCountByApplication.set(
      document.applicationId,
      (documentCountByApplication.get(document.applicationId) ?? 0) + 1,
    );
  }

  const applicationsWithFeatures = new Set(
    features.map((featureSet) => featureSet.applicationId),
  );

  return {
    total,
    items: applications.map((application) => {
      const status = this.statusMeta(application.status);

      return {
        applicationId: application.id,
        applicantExternalId: application.applicantExternalId,
        applicantName: application.applicantName,
        status: application.status,
        statusLabel: status.label,
        statusDescription: status.description,
        statusOrder: status.order,
        documentsCount: documentCountByApplication.get(application.id) ?? 0,
        hasFeatures: applicationsWithFeatures.has(application.id),
        createdAt: application.createdAt,
        updatedAt: application.updatedAt,
      };
    }),
  };
}
```

Código del controller:

```typescript
@Get('applications')
@UseGuards(ApiKeyGuard)
async listApplications() {
  return await this.clase10.listApplications();
}
```

Este endpoint debe ir antes de:

```typescript
@Get('applications/:applicationId')
```

Si lo ponemos después, NestJS podría interpretar la palabra `applications` o una ruta parecida como si fuera un `applicationId`.

---

## Paso 2: subir documentos a S3

Antes de mostrar inputs en el frontend necesitamos saber qué documentos acepta el sistema.

Ese catálogo vive en la tabla:

```txt
document_types
```

En este proyecto tenemos:

| Código | Documento | Categoría |
|--------|-----------|-----------|
| `CARNET_IDENTIDAD_BOLIVIANO` | Carné de identidad boliviano | `PERSONAL` |
| `CERTIFICADO_TRABAJO` | Certificado de trabajo | `EMPLOYMENT` |
| `BOLETA_PAGO` | Boleta de pago | `INCOME` |
| `EXTRACTO_BANCARIO` | Extracto bancario | `BANKING` |
| `FORMULARIO_SOLICITUD_CREDITO` | Formulario de solicitud de crédito | `LOAN_REQUEST` |
| `REPORTE_CREDITICIO_SIMULADO` | Reporte crediticio simulado | `CREDIT_HISTORY` |

Para no hardcodear estos documentos en el frontend, creamos:

```txt
GET /modulo1/clase10/document-types
```

Código del service:

```typescript
async listDocumentTypes() {
  const documentTypes = await this.documentTypes.find({
    where: { isActive: true },
    order: { category: 'ASC', name: 'ASC' },
  });

  return {
    total: documentTypes.length,
    items: documentTypes.map((documentType) => ({
      code: documentType.code,
      name: documentType.name,
      category: documentType.category,
    })),
  };
}
```

Código del controller:

```typescript
@Get('document-types')
@UseGuards(ApiKeyGuard)
async listDocumentTypes() {
  return await this.clase10.listDocumentTypes();
}
```

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/documents
```

Body esperado:

```json
{
  "documentType": "FORMULARIO_SOLICITUD_CREDITO",
  "fileName": "solicitud.pdf",
  "contentType": "application/pdf",
  "contentBase64": "data:application/pdf;base64,..."
}
```

Por qué usamos `base64`:

```txt
Para que el navegador pueda leer el archivo y enviarlo en JSON al backend.
```

No es la forma más eficiente para producción, pero es muy clara para aprender.

El servicio debe:

1. Validar que existe la aplicación.
2. Validar que el tipo de documento existe.
3. Decodificar `contentBase64`.
4. Subir el archivo a S3.
5. Registrar el documento en `application_documents`.

Fragmento clave:

```typescript
await this.s3.send(
  new PutObjectCommand({
    Bucket: this.config.getOrThrow<string>('AWS_S3_BUCKET'),
    Key: key,
    Body: content,
    ContentType: body.contentType ?? 'application/pdf',
  }),
);
```

Qué significa:

| Campo | Significado |
|-------|-------------|
| `Bucket` | bucket del curso |
| `Key` | ruta del archivo dentro de S3 |
| `Body` | contenido del archivo |
| `ContentType` | tipo de archivo |

Ejemplo de key:

```txt
credit-files/APPLICATION_ID/FORMULARIO_SOLICITUD_CREDITO/1710000000000-solicitud.pdf
```

---

## Paso 3: procesar documentos

Después de subir los archivos a S3, el sistema todavía no sabe qué dicen esos documentos.

Por eso la siguiente etapa es procesarlos con Textract.

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/process-documents
```

En la interfaz tendremos un botón:

```txt
Process Documents
```

Ese botón solo debe habilitarse cuando la solicitud seleccionada ya tiene documentos.

Este endpoint reutiliza Clase 3:

```typescript
async processDocuments(applicationId: string) {
  const result = await this.clase03.processCreditFile(applicationId);
  await this.applications.update(applicationId, {
    status: 'DOCUMENTS_PROCESSED',
  });
  return result;
}
```

Qué hace Clase 3:

```txt
S3 PDF/imagen -> Textract Queries -> respuestas -> datos extraídos
```

Qué hace el backend:

1. Busca la solicitud en `credit_applications`.
2. Busca sus documentos en `application_documents`.
3. Para cada documento lee su `document_type_code`.
4. Busca las preguntas configuradas en `document_types.queries`.
5. Envía el archivo de S3 a Textract.
6. Guarda el resultado en `textract_results`.
7. Guarda cada respuesta en `textract_query_answers`.
8. Reconstruye el perfil agrupado en `application_extracted_data`.
9. Cambia el estado de la solicitud a `DOCUMENTS_PROCESSED`.

Después de procesar, el frontend vuelve a consultar:

```txt
GET /modulo1/clase10/applications/:applicationId
```

Y muestra:

| Bloque | De dónde sale |
|--------|---------------|
| Datos personales | `application_extracted_data.personal_data` |
| Datos laborales | `application_extracted_data.employment_data` |
| Ingresos | `application_extracted_data.income_data` |
| Datos bancarios | `application_extracted_data.banking_data` |
| Solicitud de crédito | `application_extracted_data.loan_request_data` |
| Historial crediticio | `application_extracted_data.credit_history_data` |
| Confianza | `application_extracted_data.confidence_summary` |
| Respuestas Textract | `textract_query_answers` |

Código del frontend:

```javascript
async function processSelectedDocuments() {
  await apiRequest(`/applications/${selectedApplicationId}/process-documents`, {
    method: "POST",
  });

  selectedApplicationDetail = await apiRequest(
    `/applications/${selectedApplicationId}`,
  );

  await loadApplications();
  renderDocumentUploadPanel();
  renderExtractedInformation();
}
```

Qué hace:

| Línea | Para qué sirve |
|-------|----------------|
| `POST /process-documents` | ejecuta Textract sobre los archivos subidos |
| `GET /applications/:id` | trae documentos, features y datos extraídos actualizados |
| `loadApplications()` | refresca el listado y el estado de la solicitud |
| `renderDocumentUploadPanel()` | actualiza los documentos subidos/procesados |
| `renderExtractedInformation()` | muestra la información extraída en tarjetas |

---

## Paso 4: primer job Glue, limpieza de datos

Después de procesar documentos ya tenemos datos extraídos, pero todavía pueden venir como texto desordenado.

Ejemplos:

```txt
"Bs. 10.400,00"
"10 años y 3 meses"
"NO registra mora"
```

El primer job de Glue limpia y normaliza esos datos para que el sistema pueda usarlos de forma consistente.

La limpieza se inicia con este endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/clean
```

Internamente llama:

```typescript
return await this.clase04.cleanCreditFile({ applicationId });
```

Al iniciarlo, la solicitud cambia a:

```txt
CLEANING_STARTED
```

Como Glue es asíncrono, necesitamos consultar estado:

```txt
GET /modulo1/clase10/applications/:applicationId/clean-status
```

Cuando el job responde `SUCCEEDED`, el backend importa el archivo limpio desde S3 y guarda el perfil en:

```txt
clean_credit_profiles
```

La solicitud cambia a:

```txt
CLEAN_COMPLETED
```

En la interfaz tendremos dos botones:

```txt
Start Clean Job
Check Status
```

Y mostraremos:

| Bloque | Campos ejemplo |
|--------|----------------|
| Applicant | nombre, documento, fecha de nacimiento |
| Employment | empresa, cargo, antigüedad |
| Income and banking | ingresos, saldo promedio, gastos |
| Loan request | monto solicitado, plazo, valor inmueble |
| Credit history | deuda total, cuota mensual, mora |
| Quality report | advertencias o campos faltantes |

Código del frontend:

```javascript
async function startCleanJob() {
  await apiRequest(`/applications/${selectedApplicationId}/clean`, {
    method: "POST",
  });

  await refreshSelectedApplication();
}

async function checkCleanStatus() {
  await apiRequest(`/applications/${selectedApplicationId}/clean-status`);
  await refreshSelectedApplication();
}
```

`refreshSelectedApplication()` vuelve a leer:

```txt
GET /modulo1/clase10/applications/:applicationId
```

Así el frontend puede mostrar `cleanProfile` apenas esté listo.

---

## Paso 5: segundo job Glue, features

Cuando la limpieza termina, ya tenemos un perfil limpio.

Pero los modelos no trabajan directamente con “nombre”, “empresa” o “monto en texto”.

Los modelos necesitan variables numéricas listas, por ejemplo:

```json
{
  "debt_to_income_ratio": 0.274,
  "loan_to_value_ratio": 0.7143,
  "payment_to_income_ratio": 0.2003,
  "credit_history_score": 76
}
```

Para eso usamos el segundo job Glue: feature engineering.

Lo iniciamos con:

```txt
POST /modulo1/clase10/applications/:applicationId/features
```

Y luego consultamos:

```txt
GET /modulo1/clase10/applications/:applicationId/features-status
GET /modulo1/clase10/applications/:applicationId/features
```

Al iniciarlo, la solicitud cambia a:

```txt
FEATURES_STARTED
```

Cuando el job responde `SUCCEEDED`, el backend importa el archivo de features desde S3 y guarda el resultado en:

```txt
credit_feature_sets
```

La solicitud cambia a:

```txt
FEATURES_COMPLETED
```

Explicación simple:

```txt
Clean convierte texto/documentos en perfil limpio.
Features convierte perfil limpio en variables numéricas para modelos.
```

En la interfaz tendremos dos botones:

```txt
Start Features Job
Check Status
```

Y mostraremos:

| Bloque | Qué contiene |
|--------|--------------|
| Model ratios | ratios como deuda/ingreso, cuota/ingreso, préstamo/valor |
| Model scores | scores como estabilidad laboral, capacidad bancaria e historial |
| Features payload | JSON completo generado por el job |
| Schema payload | variables esperadas por los modelos |

Código del frontend:

```javascript
async function startFeaturesJob() {
  await apiRequest(`/applications/${selectedApplicationId}/features`, {
    method: "POST",
  });

  await refreshSelectedApplication();
}

async function checkFeaturesStatus() {
  await apiRequest(`/applications/${selectedApplicationId}/features-status`);
  await refreshSelectedApplication();
}
```

---

## Paso 6: analizar riesgo

Cuando `FEATURES_COMPLETED` está listo, ya podemos usar el primer modelo.

Antes de ejecutar predicciones desde la interfaz, vamos a crear una tabla para guardar los resultados.

Así los resultados no se pierden al refrescar el navegador.

### Tabla de predicciones

Crearemos:

```txt
application_model_predictions
```

Campos:

| Campo | Para qué sirve |
|-------|----------------|
| `id` | identificador de la predicción |
| `application_id` | solicitud analizada |
| `prediction_type` | `RISK` o `AMOUNT` |
| `model_type` | tipo de modelo usado |
| `result_payload` | respuesta completa del modelo |
| `features_payload` | variables usadas para predecir |
| `created_at` | fecha de ejecución |

Migración:

```typescript
import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreateApplicationModelPredictions1782100000000
  implements MigrationInterface
{
  name = 'CreateApplicationModelPredictions1782100000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;

    await queryRunner.query(`
      CREATE TABLE ${q}."application_model_predictions" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        "application_id" uuid NOT NULL,
        "prediction_type" text NOT NULL,
        "model_type" text NOT NULL,
        "result_payload" jsonb NOT NULL DEFAULT '{}'::jsonb,
        "features_payload" jsonb NOT NULL DEFAULT '{}'::jsonb,
        "created_at" timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT "FK_application_model_predictions_application"
          FOREIGN KEY ("application_id") REFERENCES ${q}."credit_applications"("id")
      )
    `);

    await queryRunner.query(`
      CREATE INDEX "IDX_application_model_predictions_application_type_created"
        ON ${q}."application_model_predictions" ("application_id", "prediction_type", "created_at" DESC)
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;

    await queryRunner.query(
      `DROP INDEX IF EXISTS ${q}."IDX_application_model_predictions_application_type_created"`,
    );
    await queryRunner.query(
      `DROP TABLE IF EXISTS ${q}."application_model_predictions"`,
    );
  }
}
```

Entidad:

```typescript
@Entity({ name: 'application_model_predictions' })
export class ApplicationModelPrediction {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ name: 'application_id', type: 'uuid' })
  applicationId: string;

  @Column({ name: 'prediction_type', type: 'text' })
  predictionType: string;

  @Column({ name: 'model_type', type: 'text' })
  modelType: string;

  @Column({ name: 'result_payload', type: 'jsonb', default: {} })
  resultPayload: Record<string, unknown>;

  @Column({ name: 'features_payload', type: 'jsonb', default: {} })
  featuresPayload: Record<string, unknown>;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt: Date;
}
```

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/risk
```

Internamente llama al modelo de Clase 6:

```typescript
async analyzeRisk(applicationId: string) {
  const result = await this.clase06.predictApplicationRisk(applicationId);
  await this.savePrediction(applicationId, 'RISK', result);
  await this.applications.update(applicationId, {
    status: 'RISK_ANALYZED',
  });
  return result;
}
```

Este endpoint no entrena nada.

```txt
Lee features -> lee risk_model_params.json de S3 -> calcula riesgo
```

En la interfaz tendremos el botón:

```txt
Analyze Risk
```

Respuesta esperada:

```json
{
  "defaultProbability": 0.32,
  "threshold": 0.5,
  "riskLabel": "LOW",
  "modelType": "logistic_regression_classifier"
}
```

Qué mostramos:

| Campo | Significado |
|-------|-------------|
| `defaultProbability` | probabilidad estimada de incumplimiento |
| `threshold` | punto de corte para decidir bajo/alto riesgo |
| `riskLabel` | etiqueta final: `LOW` o `HIGH` |
| `features` | variables usadas por el modelo |

---

## Paso 7: analizar monto

Después del riesgo usamos el modelo de monto.

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/amount
```

Internamente llama al modelo de Clase 7:

```typescript
async analyzeAmount(applicationId: string) {
  const result = await this.clase07.recommendApplicationAmount(applicationId);
  await this.savePrediction(applicationId, 'AMOUNT', result);
  await this.applications.update(applicationId, {
    status: 'AMOUNT_ANALYZED',
  });
  return result;
}
```

Este endpoint:

```txt
Lee features -> lee amount_model.json de S3 -> calcula monto recomendado
```

En la interfaz tendremos el botón:

```txt
Recommend Amount
```

Respuesta esperada:

```json
{
  "recommendedAmount": 420000,
  "modelType": "xgboost_regressor_tree_dump"
}
```

Qué mostramos:

| Campo | Significado |
|-------|-------------|
| `recommendedAmount` | monto sugerido por el modelo |
| `modelType` | tipo de modelo usado |
| `features` | variables usadas para calcular el monto |

Código frontend:

```javascript
async function analyzeRisk() {
  riskPrediction = await apiRequest(`/applications/${selectedApplicationId}/risk`, {
    method: "POST",
  });

  await refreshSelectedApplication();
  renderPredictions();
}

async function recommendAmount() {
  amountPrediction = await apiRequest(`/applications/${selectedApplicationId}/amount`, {
    method: "POST",
  });

  await refreshSelectedApplication();
  renderPredictions();
}
```

El método común para guardar:

```typescript
private async savePrediction(
  applicationId: string,
  predictionType: 'RISK' | 'AMOUNT',
  result: Record<string, unknown>,
) {
  await this.modelPredictions.save(
    this.modelPredictions.create({
      applicationId,
      predictionType,
      modelType: String(result.modelType ?? predictionType),
      resultPayload: result,
      featuresPayload:
        typeof result.features === 'object' && result.features !== null
          ? (result.features as Record<string, unknown>)
          : {},
    }),
  );
}
```

Cuando consultamos la solicitud con:

```txt
GET /modulo1/clase10/applications/:applicationId
```

también devolvemos:

```txt
predictions
latestPredictions
```

Así el frontend puede mostrar las predicciones guardadas incluso si se recarga la página.

---

## Paso 8: evaluación integrada

Igual que predicciones y explicaciones, la evaluación integrada debe quedar guardada.

Crearemos:

```txt
application_evaluations
```

Campos:

| Campo | Para qué sirve |
|-------|----------------|
| `id` | identificador de la evaluación |
| `application_id` | solicitud evaluada |
| `decision` | decisión final |
| `evaluation_payload` | respuesta completa |
| `created_at` | fecha de evaluación |

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/evaluate
```

Internamente reutiliza Clase 8:

```typescript
async evaluate(applicationId: string) {
  const result = await this.clase08.evaluateCreditFile(applicationId);
  await this.saveEvaluation(applicationId, result);
  await this.applications.update(applicationId, {
    status: 'EVALUATED',
  });
  return result;
}
```

Devuelve algo como:

```json
{
  "applicationId": "...",
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
    "El monto solicitado supera en mas de 10% al monto recomendado."
  ]
}
```

Cuando consultamos la solicitud con:

```txt
GET /modulo1/clase10/applications/:applicationId
```

también devolvemos:

```txt
evaluations
latestEvaluation
```

Así la interfaz puede mostrar la evaluación sin volver a calcularla.

---

## Paso 9: explicaciones

Después de riesgo y monto, generamos explicaciones para ambos modelos.

Clase 9 ya sabe generar la explicación local usando el script Python.

En Clase 10 haremos tres cosas:

1. Llamar a Clase 9 para generar la explicación.
2. Guardar el JSON en S3.
3. Guardar el resultado en base de datos para mostrarlo desde la interfaz.

### Tabla de explicaciones

Crearemos:

```txt
application_model_explanations
```

Campos:

| Campo | Para qué sirve |
|-------|----------------|
| `id` | identificador de la explicación |
| `application_id` | solicitud explicada |
| `explanation_type` | `FULL`, `RISK` o `AMOUNT` |
| `model_type` | método o tipo de explicación |
| `explanation_payload` | explicación completa |
| `s3_key` | ruta del JSON generado en S3 |
| `created_at` | fecha de generación |

Migración:

```typescript
import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreateApplicationModelExplanations1782200000000
  implements MigrationInterface
{
  name = 'CreateApplicationModelExplanations1782200000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;

    await queryRunner.query(`
      CREATE TABLE ${q}."application_model_explanations" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        "application_id" uuid NOT NULL,
        "explanation_type" text NOT NULL,
        "model_type" text NOT NULL,
        "explanation_payload" jsonb NOT NULL DEFAULT '{}'::jsonb,
        "s3_key" text,
        "created_at" timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT "FK_application_model_explanations_application"
          FOREIGN KEY ("application_id") REFERENCES ${q}."credit_applications"("id")
      )
    `);

    await queryRunner.query(`
      CREATE INDEX "IDX_application_model_explanations_application_type_created"
        ON ${q}."application_model_explanations" ("application_id", "explanation_type", "created_at" DESC)
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;

    await queryRunner.query(
      `DROP INDEX IF EXISTS ${q}."IDX_application_model_explanations_application_type_created"`,
    );
    await queryRunner.query(
      `DROP TABLE IF EXISTS ${q}."application_model_explanations"`,
    );
  }
}
```

Entidad:

```typescript
@Entity({ name: 'application_model_explanations' })
export class ApplicationModelExplanation {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ name: 'application_id', type: 'uuid' })
  applicationId: string;

  @Column({ name: 'explanation_type', type: 'text' })
  explanationType: string;

  @Column({ name: 'model_type', type: 'text' })
  modelType: string;

  @Column({ name: 'explanation_payload', type: 'jsonb', default: {} })
  explanationPayload: Record<string, unknown>;

  @Column({ name: 's3_key', type: 'text', nullable: true })
  s3Key?: string;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt: Date;
}
```

Endpoints:

```txt
POST /modulo1/clase10/applications/:applicationId/explanation/generate
GET /modulo1/clase10/applications/:applicationId/risk-explanation
GET /modulo1/clase10/applications/:applicationId/amount-explanation
GET /modulo1/clase10/applications/:applicationId/explanation
```

Método principal del service:

```typescript
async generateExplanation(applicationId: string) {
  const result =
    await this.clase09.generateApplicationExplanation(applicationId);

  await this.saveExplanationBundle(applicationId, result);
  await this.applications.update(applicationId, {
    status: 'EXPLANATION_GENERATED',
  });

  return result;
}
```

Guardamos tres registros:

| `explanation_type` | Qué guarda |
|--------------------|------------|
| `FULL` | JSON completo de explicación |
| `RISK` | predicción y explicación de riesgo |
| `AMOUNT` | predicción y explicación de monto |

Cuando consultamos la solicitud con:

```txt
GET /modulo1/clase10/applications/:applicationId
```

también devolvemos:

```txt
explanations
latestExplanations
```

Así la interfaz muestra las explicaciones desde la base de datos.

En el frontend tendremos:

```txt
Generate Explanations
```

Ese botón llama:

```javascript
await apiRequest(`/applications/${selectedApplicationId}/explanation/generate`, {
  method: "POST",
});
```

Y luego muestra:

| Bloque | Qué muestra |
|--------|-------------|
| Risk explanation | resultado de riesgo y factores principales |
| Amount explanation | monto recomendado y factores principales |

---

## Paso 10: crear `Clase10Controller`

Archivo:

```txt
src/modulo1/clase10/clase10.controller.ts
```

El controller define rutas HTTP y llama al service.

Ejemplo:

```typescript
@Post('applications')
@UseGuards(ApiKeyGuard)
async createApplication(
  @Body() body: { applicantExternalId?: string; applicantName?: string },
) {
  return await this.clase10.createApplication(body);
}
```

Qué hace cada parte:

| Parte | Para qué sirve |
|-------|----------------|
| `@Post('applications')` | define endpoint HTTP |
| `@UseGuards(ApiKeyGuard)` | protege con API key/secret |
| `@Body()` | lee JSON enviado por frontend |
| `this.clase10.createApplication` | llama la lógica del service |

---

## Paso 11: crear la interfaz

El frontend de esta clase vive en una carpeta separada:

```txt
frontend/
├── index.html
├── styles.css
├── app.js
├── package.json
└── server.js
```

No tendrá login ni menú porque es un demo de integración.

La primera pantalla debe mostrar:

| Columna | Qué muestra |
|---------|-------------|
| `Application ID` | ID real de la solicitud |
| `Applicant Name` | nombre del solicitante |
| `Status` | estado actual del flujo |
| `Updated` | última actualización |
| `Action` | acción rápida, por ejemplo copiar el ID |

Para ejecutar:

```bash
cd frontend
npm run dev
```

Abrimos:

```txt
http://localhost:5173
```

Como el frontend corre en `localhost:5173` y NestJS en `localhost:3000`, el backend debe habilitar CORS:

```typescript
app.enableCors({
  origin: ['http://localhost:5173', 'http://127.0.0.1:5173'],
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'x-api-key', 'x-api-secret'],
});
```

Qué significa:

| Línea | Para qué sirve |
|-------|----------------|
| `origin` | permite llamadas desde el frontend local |
| `methods` | habilita los métodos HTTP que usaremos |
| `allowedHeaders` | permite enviar API key y secret |

---

## Paso 12: llamadas desde el frontend

En `frontend/app.js` tendremos una configuración base:

```javascript
const API_BASE_URL = "http://localhost:3000/modulo1/clase10";
const API_HEADERS = {
  "Content-Type": "application/json",
  "x-api-key": "test1",
  "x-api-secret": "pass1",
};
```

La función central del frontend será:

```javascript
async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      ...API_HEADERS,
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  return await response.json();
}
```

Qué hace:

1. Llama un endpoint.
2. Agrega `x-api-key` y `x-api-secret`.
3. Convierte la respuesta a JSON.
4. Lanza un error si el backend responde mal.

Para cargar el listado:

```javascript
async function loadApplications() {
  const response = await apiRequest("/applications");
  applications = response.items ?? [];
  renderApplications();
}
```

Esta llamada usa:

```txt
GET /modulo1/clase10/applications
```

Para crear solicitud:

```javascript
async function createApplication() {
  await apiRequest("/applications", {
    method: "POST",
    body: JSON.stringify({
      applicantName: document.getElementById("applicantName").value,
      applicantExternalId: document.getElementById("externalId").value
    })
  });

  await loadApplications();
}
```

Para subir archivo:

```javascript
const contentBase64 = await fileToBase64(file);
await apiRequest(`/applications/${applicationId}/documents`, {
  method: "POST",
  body: JSON.stringify({
    documentType,
    fileName: file.name,
    contentType: file.type,
    contentBase64
  })
});
```

Para cargar los tipos de documento:

```javascript
const documentTypesResponse = await apiRequest("/document-types");
documentTypes = documentTypesResponse.items ?? [];
```

Para convertir un archivo seleccionado a base64:

```javascript
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
```

Qué hace cada línea:

| Línea | Para qué sirve |
|-------|----------------|
| `new FileReader()` | crea un lector de archivos del navegador |
| `reader.onload` | se ejecuta cuando el archivo ya fue leído |
| `reader.onerror` | captura errores de lectura |
| `readAsDataURL(file)` | convierte el archivo a `data:...;base64,...` |

Flujo del frontend para documentos:

1. El usuario crea o selecciona una solicitud.
2. El frontend consulta `/document-types`.
3. Por cada tipo de documento muestra una tarjeta con input de archivo.
4. El usuario selecciona PDF o imagen.
5. El navegador convierte el archivo a base64.
6. El frontend llama `POST /applications/:applicationId/documents`.
7. El backend guarda el archivo en S3.
8. El backend registra el documento en `application_documents`.
9. La solicitud cambia a `DOCUMENTS_UPLOADED`.

---

## Paso 13: registrar en `Modulo1Module`

Agregar imports:

```typescript
import { Clase10Controller } from './clase10/clase10.controller';
import { Clase10Service } from './clase10/clase10.service';
```

Agregar controller:

```typescript
controllers: [
  Clase10Controller,
]
```

Agregar provider:

```typescript
providers: [
  Clase10Service,
]
```

---

## Paso 14: aumentar límite de JSON

Como subimos archivos en base64, necesitamos aceptar bodies más grandes.

Archivo:

```txt
src/main.ts
```

Agregar:

```typescript
import { json } from 'express';
```

Y dentro de `bootstrap`:

```typescript
app.use(json({ limit: '20mb' }));
```

Esto evita errores cuando subimos PDFs pequeños desde el frontend.

---

## Paso 15: probar

Antes de levantar el backend, correr migraciones:

```bash
npm run migration:run
```

Esto crea las tablas nuevas de la clase, especialmente:

```txt
application_model_predictions
application_model_explanations
application_evaluations
```

La primera guarda los resultados de riesgo y monto.

La segunda guarda las explicaciones de riesgo y monto.

La tercera guarda la evaluación integrada.

Así la interfaz puede mostrar predicciones y explicaciones desde la base de datos.

Luego levanta el backend:

```bash
npm run start:dev
```

En otra terminal levanta el frontend:

```bash
cd ../frontend
npm run dev
```

Abre:

```txt
http://localhost:5173
```

Orden recomendado:

1. Crear aplicación.
2. Subir documentos.
3. Procesar documentos.
4. Iniciar limpieza.
5. Consultar estado de limpieza hasta `SUCCEEDED`.
6. Generar features.
7. Consultar estado de features hasta `SUCCEEDED`.
8. Ver features.
9. Analizar riesgo.
10. Analizar monto.
11. Evaluación integrada.
12. Generar explicaciones.
13. Revisar explicaciones de riesgo y monto en la interfaz.

---

## Entrega

Cada estudiante debe entregar:

- captura de la aplicación creada;
- captura de documentos subidos;
- captura de features generadas;
- resultado de riesgo;
- resultado de monto;
- evaluación integrada;
- explicación de riesgo;
- explicación de monto;
- breve conclusión sobre el expediente.

Esta clase prepara directamente la presentación de Clase 11.
