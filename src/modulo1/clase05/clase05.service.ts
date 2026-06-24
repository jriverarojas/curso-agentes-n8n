import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { GetObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { CleanCreditProfile } from '../../entities/clean-credit-profile.entity';
import { CreditFeatureSet } from '../../entities/credit-feature-set.entity';
import { GlueJobRunEntity } from '../../entities/glue-job-run.entity';
import { GlueService } from '../clase04/glue.service';

@Injectable()
export class Clase05Service {
  private readonly s3: S3Client;

  constructor(
    private readonly config: ConfigService,
    private readonly glue: GlueService,
    @InjectRepository(CleanCreditProfile)
    private readonly cleanProfiles: Repository<CleanCreditProfile>,
    @InjectRepository(CreditFeatureSet)
    private readonly featureSets: Repository<CreditFeatureSet>,
    @InjectRepository(GlueJobRunEntity)
    private readonly glueRuns: Repository<GlueJobRunEntity>,
  ) {
    this.s3 = new S3Client({
      region: this.config.getOrThrow<string>('AWS_REGION'),
    });
  }

  async generateFeatures(body: { applicationId: string }) {
    const profile = await this.cleanProfiles.findOne({
      where: { applicationId: body.applicationId },
    });

    if (!profile) {
      throw new BadRequestException('Run Clase 4 before generating features');
    }

    const cleanPrefix = this.config.getOrThrow<string>('AWS_S3_CLEAN_PREFIX');
    const featuresPrefix = this.config.getOrThrow<string>('AWS_S3_FEATURES_PREFIX');
    const inputKey = `${cleanPrefix}/${body.applicationId}/clean-profile.json`;
    const outputKey = `${featuresPrefix}/${body.applicationId}/features.json`;

    const job = await this.glue.startFeaturesJob({
      applicationId: body.applicationId,
      inputKey,
      outputKey,
    });

    const run = await this.glueRuns.save(
      this.glueRuns.create({
        applicationId: body.applicationId,
        jobName: job.jobName,
        jobRunId: job.jobRunId,
        jobType: 'FEATURE_ENGINEERING',
        status: 'STARTING',
        inputPath: inputKey,
        outputPath: outputKey,
      }),
    );

    return {
      applicationId: body.applicationId,
      jobRunId: run.jobRunId,
      status: run.status,
      outputKey,
    };
  }

  async getFeaturesStatus(applicationId: string) {
    const run = await this.glueRuns.findOne({
      where: { applicationId, jobType: 'FEATURE_ENGINEERING' },
      order: { createdAt: 'DESC' },
    });

    if (!run) {
      throw new NotFoundException('No features job found for this application');
    }

    const status = await this.glue.getJobStatus(run.jobName, run.jobRunId);
    await this.glueRuns.update(run.id, { status });

    if (status === 'SUCCEEDED') {
      await this.importFeatures(applicationId, run.outputPath!);
    }

    return {
      applicationId,
      jobRunId: run.jobRunId,
      status,
    };
  }

  async getFeatures(applicationId: string) {
    const features = await this.featureSets.findOne({ where: { applicationId } });
    if (!features) {
      throw new NotFoundException('Features not found for this application');
    }
    return features;
  }

  private async importFeatures(applicationId: string, key: string) {
    const response = await this.s3.send(
      new GetObjectCommand({
        Bucket: this.config.getOrThrow<string>('AWS_S3_BUCKET'),
        Key: key,
      }),
    );

    const payload = JSON.parse(await response.Body!.transformToString());
    const features = payload.features;
    const existing = await this.featureSets.findOne({ where: { applicationId } });

    await this.featureSets.save(
      this.featureSets.create({
        ...(existing ?? {}),
        applicationId,
        debtToIncomeRatio: features.debt_to_income_ratio,
        loanToValueRatio: features.loan_to_value_ratio,
        paymentToIncomeRatio: features.payment_to_income_ratio,
        expenseToIncomeRatio: features.expense_to_income_ratio,
        totalObligationsToIncomeRatio: features.total_obligations_to_income_ratio,
        employmentStabilityScore: features.employment_stability_score,
        bankingCapacityScore: features.banking_capacity_score,
        creditHistoryScore: features.credit_history_score,
        syntheticRiskLabel: features.synthetic_risk_label,
        featuresPayload: features,
        schemaPayload: payload.schema,
      }),
    );
  }
}
