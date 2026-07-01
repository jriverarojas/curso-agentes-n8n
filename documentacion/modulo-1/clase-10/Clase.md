# Clase 10: Interfaz completa del flujo crediticio

| | |
|---|---|
| **Clase** | 10 de 11 |
| **Duración** | 3 horas |
| **Controlador** | `Clase10Controller` |
| **Pantalla** | `GET /modulo1/clase10/ui` |
| **Objetivo** | Unir todo el pipeline en una interfaz práctica |

## Objetivos

Al terminar esta sesión podrás:

- Crear una solicitud de crédito desde una interfaz.
- Subir documentos desde el navegador para guardarlos en S3.
- Procesar documentos con el flujo ya creado en clases anteriores.
- Generar features.
- Ejecutar análisis de riesgo.
- Ejecutar análisis de monto recomendado.
- Consultar explicaciones de riesgo y monto.
- Entender cómo una interfaz conversa con una API NestJS.

Esta clase es netamente práctica. La idea es construir una pantalla sencilla, no una aplicación de producción.

---

## Flujo final

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
| `GET` | `/ui` | Muestra la interfaz |
| `POST` | `/applications` | Crea una solicitud nueva |
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
      status: 'CREATED_FROM_UI',
    }),
  );

  return {
    applicationId: application.id,
    status: application.status,
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

## Paso 2: subir documentos a S3

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/documents
```

Tipos de documento disponibles:

| Código | Documento |
|--------|-----------|
| `CERTIFICADO_TRABAJO` | certificado laboral |
| `BOLETA_PAGO` | boleta de pago |
| `EXTRACTO_BANCARIO` | extracto bancario |
| `FORMULARIO_SOLICITUD_CREDITO` | solicitud de crédito |
| `REPORTE_CREDITICIO_SIMULADO` | reporte crediticio |

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

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/process-documents
```

Este endpoint reutiliza Clase 3:

```typescript
async processDocuments(applicationId: string) {
  const result = await this.clase03.processCreditFile(applicationId);
  await this.applications.update(applicationId, {
    status: 'TEXTRACT_COMPLETED',
  });
  return result;
}
```

Qué hace Clase 3:

```txt
S3 PDF -> Textract Queries -> respuestas -> tablas de extracción
```

En la interfaz, este paso se ejecuta después de subir documentos.

---

## Paso 4: limpieza y features

La limpieza se inicia con:

```txt
POST /modulo1/clase10/applications/:applicationId/clean
```

Internamente llama:

```typescript
return await this.clase04.cleanCreditFile({ applicationId });
```

Como Glue es asíncrono, necesitamos consultar estado:

```txt
GET /modulo1/clase10/applications/:applicationId/clean-status
```

Cuando la limpieza termina, generamos features:

```txt
POST /modulo1/clase10/applications/:applicationId/features
```

Y luego consultamos:

```txt
GET /modulo1/clase10/applications/:applicationId/features-status
GET /modulo1/clase10/applications/:applicationId/features
```

Explicación simple:

```txt
Clean convierte texto/documentos en perfil limpio.
Features convierte perfil limpio en variables numéricas para modelos.
```

---

## Paso 5: analizar riesgo

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/risk
```

Internamente llama al modelo de Clase 6:

```typescript
async analyzeRisk(applicationId: string) {
  return await this.clase06.predictApplicationRisk(applicationId);
}
```

Este endpoint no entrena nada.

```txt
Lee features -> lee risk_model_params.json de S3 -> calcula riesgo
```

---

## Paso 6: analizar monto

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/amount
```

Internamente llama al modelo de Clase 7:

```typescript
async analyzeAmount(applicationId: string) {
  return await this.clase07.recommendApplicationAmount(applicationId);
}
```

Este endpoint:

```txt
Lee features -> lee amount_model.json de S3 -> calcula monto recomendado
```

---

## Paso 7: evaluación integrada

Endpoint:

```txt
POST /modulo1/clase10/applications/:applicationId/evaluate
```

Internamente reutiliza Clase 8:

```typescript
async evaluate(applicationId: string) {
  return await this.clase08.evaluateCreditFile(applicationId);
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

---

## Paso 8: explicaciones

Clase 9 generó explicaciones en JSON dentro de S3.

Clase 10 solo las consulta.

Endpoints:

```txt
GET /modulo1/clase10/applications/:applicationId/risk-explanation
GET /modulo1/clase10/applications/:applicationId/amount-explanation
GET /modulo1/clase10/applications/:applicationId/explanation
```

Ejemplo de respuesta de riesgo:

```json
{
  "applicationId": "...",
  "risk": {
    "default_probability": 0.7,
    "risk_label": "HIGH"
  },
  "riskExplanation": [
    {
      "feature": "total_obligations_to_income_ratio",
      "value": 1.45,
      "contribution": 0.18
    }
  ]
}
```

---

## Paso 9: crear `Clase10Controller`

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

## Paso 10: crear la interfaz

Para no instalar React, Vite ni paquetes extra, serviremos una página HTML desde NestJS:

```txt
GET /modulo1/clase10/ui
```

En el controller:

```typescript
@Get('ui')
@Header('Content-Type', 'text/html; charset=utf-8')
getUi() {
  return this.renderUi();
}
```

Qué significa:

| Línea | Para qué sirve |
|-------|----------------|
| `@Get('ui')` | crea una ruta que se abre en navegador |
| `@Header(...)` | indica que la respuesta es HTML |
| `renderUi()` | devuelve el HTML como string |

La UI tendrá secciones:

```txt
Credenciales API
Progreso
Crear solicitud
Subir documentos
Procesar documentos/features
Analizar modelos
Explicaciones
Resultado JSON
```

---

## Paso 11: llamadas desde el frontend

La función central del frontend será:

```javascript
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...headers(), ...(options.headers || {}) }
  });
  const data = await response.json();
  if (!response.ok) throw data;
  setOutput(data);
  return data;
}
```

Qué hace:

1. Llama un endpoint.
2. Agrega `x-api-key` y `x-api-secret`.
3. Convierte la respuesta a JSON.
4. Muestra el resultado en pantalla.

Para crear solicitud:

```javascript
async function createApplication() {
  const data = await api("/modulo1/clase10/applications", {
    method: "POST",
    body: JSON.stringify({
      applicantName: document.getElementById("applicantName").value,
      applicantExternalId: document.getElementById("externalId").value
    })
  });
  currentApplicationId = data.applicationId;
}
```

Para subir archivo:

```javascript
const contentBase64 = await fileToBase64(file);
await api("/modulo1/clase10/applications/" + applicationId + "/documents", {
  method: "POST",
  body: JSON.stringify({
    documentType,
    fileName: file.name,
    contentType: file.type,
    contentBase64
  })
});
```

---

## Paso 12: registrar en `Modulo1Module`

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

## Paso 13: aumentar límite de JSON

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

## Paso 14: probar

Levanta el backend:

```bash
npm run start:dev
```

Abre:

```txt
http://localhost:3000/modulo1/clase10/ui
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
12. Explicaciones.

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
