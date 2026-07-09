import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import {
  BadRequestException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { InjectRepository } from '@nestjs/typeorm';
import { In, Repository } from 'typeorm';
import { ApplicationDocument } from '../../entities/application-document.entity';
import { ApplicationEvaluation } from '../../entities/application-evaluation.entity';
import { ApplicationExtractedData } from '../../entities/application-extracted-data.entity';
import { ApplicationModelExplanation } from '../../entities/application-model-explanation.entity';
import { ApplicationModelPrediction } from '../../entities/application-model-prediction.entity';
import { CleanCreditProfile } from '../../entities/clean-credit-profile.entity';
import { CreditApplication } from '../../entities/credit-application.entity';
import { CreditFeatureSet } from '../../entities/credit-feature-set.entity';
import { DocumentType } from '../../entities/document-type.entity';
import { GlueJobRunEntity } from '../../entities/glue-job-run.entity';
import { TextractQueryAnswer } from '../../entities/textract-query-answer.entity';
import { TextractResult } from '../../entities/textract-result.entity';
import { Clase03Service } from '../clase03/clase03.service';
import { Clase04Service } from '../clase04/clase04.service';
import { Clase05Service } from '../clase05/clase05.service';
import { Clase06Service } from '../clase06/clase06.service';
import { Clase07Service } from '../clase07/clase07.service';
import { Clase08Service } from '../clase08/clase08.service';
import { Clase09Service } from '../clase09/clase09.service';

type UploadDocumentBody = {
  documentType: string;
  fileName: string;
  contentType?: string;
  contentBase64: string;
};

const APPLICATION_STATUSES = {
  DRAFT: {
    label: 'Borrador',
    description: 'La solicitud existe, pero todavia no tiene documentos.',
    order: 1,
  },
  DOCUMENTS_UPLOADED: {
    label: 'Archivos cargados',
    description: 'Ya se subieron uno o mas documentos a S3.',
    order: 2,
  },
  DOCUMENTS_PROCESSED: {
    label: 'Documentos procesados',
    description: 'Textract ya leyo los documentos y guardo los datos extraidos.',
    order: 3,
  },
  CLEANING_STARTED: {
    label: 'Limpieza iniciada',
    description: 'El job de limpieza de datos fue enviado a Glue.',
    order: 4,
  },
  CLEAN_COMPLETED: {
    label: 'Informacion limpia',
    description: 'Los datos extraidos ya fueron normalizados.',
    order: 5,
  },
  FEATURES_STARTED: {
    label: 'Features en proceso',
    description: 'El job que crea las variables del modelo fue enviado a Glue.',
    order: 6,
  },
  FEATURES_COMPLETED: {
    label: 'Variables listas',
    description: 'Las features ya estan guardadas y listas para los modelos.',
    order: 7,
  },
  RISK_ANALYZED: {
    label: 'Riesgo analizado',
    description: 'El modelo de riesgo ya devolvio una prediccion.',
    order: 8,
  },
  AMOUNT_ANALYZED: {
    label: 'Monto analizado',
    description: 'El modelo de monto ya devolvio una recomendacion.',
    order: 9,
  },
  EVALUATED: {
    label: 'Evaluacion integrada',
    description: 'La respuesta final combina riesgo y monto recomendado.',
    order: 10,
  },
  EXPLANATION_GENERATED: {
    label: 'Explicacion generada',
    description: 'Ya se genero la explicacion para los resultados del modelo.',
    order: 11,
  },
} as const;

type ApplicationStatus = keyof typeof APPLICATION_STATUSES;

const LEGACY_STATUS_MAP: Record<string, ApplicationStatus> = {
  CREATED_FROM_UI: 'DRAFT',
  DOCUMENTS_REGISTERED: 'DOCUMENTS_UPLOADED',
  TEXTRACT_COMPLETED: 'DOCUMENTS_PROCESSED',
};

@Injectable()
export class Clase10Service {
  private readonly s3: S3Client;

  constructor(
    private readonly config: ConfigService,
    private readonly clase03: Clase03Service,
    private readonly clase04: Clase04Service,
    private readonly clase05: Clase05Service,
    private readonly clase06: Clase06Service,
    private readonly clase07: Clase07Service,
    private readonly clase08: Clase08Service,
    private readonly clase09: Clase09Service,
    @InjectRepository(CreditApplication)
    private readonly applications: Repository<CreditApplication>,
    @InjectRepository(ApplicationDocument)
    private readonly documents: Repository<ApplicationDocument>,
    @InjectRepository(ApplicationEvaluation)
    private readonly evaluations: Repository<ApplicationEvaluation>,
    @InjectRepository(DocumentType)
    private readonly documentTypes: Repository<DocumentType>,
    @InjectRepository(CreditFeatureSet)
    private readonly featureSets: Repository<CreditFeatureSet>,
    @InjectRepository(ApplicationExtractedData)
    private readonly extractedData: Repository<ApplicationExtractedData>,
    @InjectRepository(TextractQueryAnswer)
    private readonly queryAnswers: Repository<TextractQueryAnswer>,
    @InjectRepository(TextractResult)
    private readonly textractResults: Repository<TextractResult>,
    @InjectRepository(CleanCreditProfile)
    private readonly cleanProfiles: Repository<CleanCreditProfile>,
    @InjectRepository(GlueJobRunEntity)
    private readonly glueRuns: Repository<GlueJobRunEntity>,
    @InjectRepository(ApplicationModelPrediction)
    private readonly modelPredictions: Repository<ApplicationModelPrediction>,
    @InjectRepository(ApplicationModelExplanation)
    private readonly modelExplanations: Repository<ApplicationModelExplanation>,
  ) {
    this.s3 = new S3Client({
      region: this.config.getOrThrow<string>('AWS_REGION'),
      credentials: {
        accessKeyId: this.config.getOrThrow<string>('AWS_ACCESS_KEY_ID'),
        secretAccessKey: this.config.getOrThrow<string>('AWS_SECRET_ACCESS_KEY'),
      },
    });
  }

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
          documentsCount:
            documentCountByApplication.get(application.id) ?? 0,
          hasFeatures: applicationsWithFeatures.has(application.id),
          createdAt: application.createdAt,
          updatedAt: application.updatedAt,
        };
      }),
      statusCatalog: Object.entries(APPLICATION_STATUSES).map(
        ([status, metadata]) => ({
          status,
          ...metadata,
        }),
      ),
    };
  }

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

  async getApplication(applicationId: string) {
    const application = await this.getApplicationOrThrow(applicationId);
    const documents = await this.documents.find({ where: { applicationId } });
    const features = await this.featureSets.findOne({
      where: { applicationId },
    });
    const extractedData = await this.extractedData.findOne({
      where: { applicationId },
    });
    const queryAnswers = await this.queryAnswers.find({
      where: { applicationId },
      order: { documentTypeCode: 'ASC', alias: 'ASC' },
    });
    const textractResults = await this.textractResults.find({
      where: { applicationId },
      order: { createdAt: 'DESC' },
    });
    const cleanProfile = await this.cleanProfiles.findOne({
      where: { applicationId },
    });
    const cleanJob = await this.glueRuns.findOne({
      where: { applicationId, jobType: 'CLEAN_CREDIT_FILE' },
      order: { createdAt: 'DESC' },
    });
    const featureJob = await this.glueRuns.findOne({
      where: { applicationId, jobType: 'FEATURE_ENGINEERING' },
      order: { createdAt: 'DESC' },
    });
    const predictions = await this.modelPredictions.find({
      where: { applicationId },
      order: { createdAt: 'DESC' },
    });
    const explanations = await this.modelExplanations.find({
      where: { applicationId },
      order: { createdAt: 'DESC' },
    });
    const evaluations = await this.evaluations.find({
      where: { applicationId },
      order: { createdAt: 'DESC' },
    });

    return {
      application,
      documents,
      features,
      extractedData,
      cleanProfile,
      cleanJob,
      featureJob,
      predictions,
      latestPredictions: this.latestPredictionsByType(predictions),
      explanations,
      latestExplanations: this.latestExplanationsByType(explanations),
      evaluations,
      latestEvaluation: evaluations[0] ?? null,
      queryAnswers,
      textractResults: textractResults.map((result) => ({
        id: result.id,
        documentId: result.documentId,
        documentTypeCode: result.documentTypeCode,
        status: result.status,
        summary: result.summary,
        createdAt: result.createdAt,
      })),
    };
  }

  

  async uploadDocument(applicationId: string, body: UploadDocumentBody) {
    await this.getApplicationOrThrow(applicationId);

    if (!body.documentType || !body.fileName || !body.contentBase64) {
      throw new BadRequestException(
        'documentType, fileName and contentBase64 are required',
      );
    }

    const documentType = body.documentType.toUpperCase();
    await this.validateDocumentType(documentType);

    const key = this.documentKey(applicationId, documentType, body.fileName);
    const content = this.decodeBase64(body.contentBase64);

    await this.s3.send(
      new PutObjectCommand({
        Bucket: this.config.getOrThrow<string>('AWS_S3_BUCKET'),
        Key: key,
        Body: content,
        ContentType: body.contentType ?? 'application/pdf',
      }),
    );

    const document = await this.documents.save(
      this.documents.create({
        applicationId,
        documentTypeCode: documentType,
        fileName: body.fileName,
        s3Key: key,
        status: 'UPLOADED',
      }),
    );

    await this.applications.update(applicationId, {
      status: 'DOCUMENTS_UPLOADED',
    });

    return {
      applicationId,
      document,
      bucket: this.config.getOrThrow<string>('AWS_S3_BUCKET'),
      s3Key: key,
    };
  }

  private async getApplicationOrThrow(applicationId: string) {
    const application = await this.applications.findOne({
      where: { id: applicationId },
    });

    if (!application) {
      throw new NotFoundException(`Application not found: ${applicationId}`);
    }

    return application;
  }

  private async validateDocumentType(documentType: string) {
    const existing = await this.documentTypes.find({
      where: { code: In([documentType]), isActive: true },
    });

    if (!existing.length) {
      throw new BadRequestException(`Unknown documentType: ${documentType}`);
    }
  }

  private documentKey(
    applicationId: string,
    documentType: string,
    fileName: string,
  ) {
    const safeName = fileName.replace(/[^a-zA-Z0-9._-]/g, '_');
    return `credit-files/${applicationId}/${documentType}/${Date.now()}-${safeName}`;
  }

  private decodeBase64(value: string) {
    const base64 = value.includes(',') ? value.split(',').pop()! : value;
    return Buffer.from(base64, 'base64');
  }

  private statusMeta(status?: string) {
    const normalizedStatus =
      status && LEGACY_STATUS_MAP[status] ? LEGACY_STATUS_MAP[status] : status;

    return (
      APPLICATION_STATUSES[normalizedStatus as ApplicationStatus] ?? {
        label: status ?? 'Sin estado',
        description: 'Estado no catalogado todavia.',
        order: 0,
      }
    );
  }

  private isSucceededStatus(status?: string) {
    return ['SUCCEEDED', 'COMPLETED', 'SUCCESS'].includes(
      String(status ?? '').toUpperCase(),
    );
  }

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

  private async saveEvaluation(
    applicationId: string,
    result: Record<string, unknown>,
  ) {
    await this.evaluations.save(
      this.evaluations.create({
        applicationId,
        decision: String(result.decision ?? 'UNKNOWN'),
        evaluationPayload: result,
      }),
    );
  }

  private latestPredictionsByType(predictions: ApplicationModelPrediction[]) {
    const latest: Record<string, ApplicationModelPrediction> = {};

    for (const prediction of predictions) {
      if (!latest[prediction.predictionType]) {
        latest[prediction.predictionType] = prediction;
      }
    }

    return latest;
  }

  private async saveExplanationBundle(
    applicationId: string,
    result: Record<string, unknown>,
  ) {
    const s3Key = typeof result.s3Key === 'string' ? result.s3Key : undefined;

    await Promise.all([
      this.saveExplanation(applicationId, 'FULL', result, s3Key),
      this.saveExplanation(
        applicationId,
        'RISK',
        this.pickExplanationPayload(
          result,
          'risk',
          'risk_explanation_summary',
          'risk_explanation',
        ),
        s3Key,
      ),
      this.saveExplanation(
        applicationId,
        'AMOUNT',
        this.pickExplanationPayload(
          result,
          'amount',
          'amount_explanation_summary',
          'amount_explanation',
        ),
        s3Key,
      ),
    ]);
  }

  private pickExplanationPayload(
    result: Record<string, unknown>,
    predictionKey: string,
    summaryKey: string,
    contributionsKey: string,
  ) {
    const summary =
      typeof result[summaryKey] === 'object' && result[summaryKey] !== null
        ? (result[summaryKey] as Record<string, unknown>)
        : {};

    return {
      prediction: result[predictionKey],
      explanation: {
        ...summary,
        contributions: result[contributionsKey],
      },
    };
  }

  private async saveExplanation(
    applicationId: string,
    explanationType: 'FULL' | 'RISK' | 'AMOUNT',
    payload: Record<string, unknown>,
    s3Key?: string,
  ) {
    await this.modelExplanations.save(
      this.modelExplanations.create({
        applicationId,
        explanationType,
        modelType: this.explanationModelType(explanationType, payload),
        explanationPayload: payload,
        s3Key,
      }),
    );
  }

  private explanationModelType(
    explanationType: 'FULL' | 'RISK' | 'AMOUNT',
    payload: Record<string, unknown>,
  ) {
    const explanation = payload.explanation;
    if (
      typeof explanation === 'object' &&
      explanation !== null &&
      'method' in explanation
    ) {
      return String((explanation as Record<string, unknown>).method);
    }

    return explanationType;
  }

  private latestExplanationsByType(
    explanations: ApplicationModelExplanation[],
  ) {
    const latest: Record<string, ApplicationModelExplanation> = {};

    for (const explanation of explanations) {
      if (!latest[explanation.explanationType]) {
        latest[explanation.explanationType] = explanation;
      }
    }

    return latest;
  }

}