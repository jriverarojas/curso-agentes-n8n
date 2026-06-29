import { GetObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { CreditFeatureSet } from '../../entities/credit-feature-set.entity';

type RiskFeatures = {
  debt_to_income_ratio: number;
  loan_to_value_ratio: number;
  payment_to_income_ratio: number;
  expense_to_income_ratio: number;
  total_obligations_to_income_ratio: number;
  employment_stability_score: number;
  banking_capacity_score: number;
  credit_history_score: number;
};

type RiskModelParams = {
  features: (keyof RiskFeatures)[];
  threshold: number;
  scaler: {
    mean: Record<string, number>;
    scale: Record<string, number>;
  };
  coefficients: Record<string, number>;
  intercept: number;
};

@Injectable()
export class Clase06Service {
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

  async getRiskMetrics() {
    return await this.readJson(
      this.config.getOrThrow<string>('SAGEMAKER_RISK_METRICS_KEY'),
    );
  }

  async predictApplicationRisk(applicationId: string) {
    const featureSet = await this.featureSets.findOne({
      where: { applicationId },
    });

    if (!featureSet) {
      throw new NotFoundException('Feature set not found for this application');
    }

    const features: RiskFeatures = {
      debt_to_income_ratio: this.toNumber(featureSet.debtToIncomeRatio),
      loan_to_value_ratio: this.toNumber(featureSet.loanToValueRatio),
      payment_to_income_ratio: this.toNumber(featureSet.paymentToIncomeRatio),
      expense_to_income_ratio: this.toNumber(featureSet.expenseToIncomeRatio),
      total_obligations_to_income_ratio: this.toNumber(
        featureSet.totalObligationsToIncomeRatio,
      ),
      employment_stability_score: this.toNumber(
        featureSet.employmentStabilityScore,
      ),
      banking_capacity_score: this.toNumber(featureSet.bankingCapacityScore),
      credit_history_score: this.toNumber(featureSet.creditHistoryScore),
    };

    return await this.predictRisk(features);
  }

  async predictRisk(features: RiskFeatures) {
    const model = (await this.readJson(
      this.config.getOrThrow<string>('SAGEMAKER_RISK_MODEL_PARAMS_KEY'),
    )) as RiskModelParams;

    let score = model.intercept;

    for (const featureName of model.features) {
      const rawValue = features[featureName];
      const mean = model.scaler.mean[featureName];
      const scale = model.scaler.scale[featureName] || 1;
      const coefficient = model.coefficients[featureName];
      const standardizedValue = (rawValue - mean) / scale;
      score += standardizedValue * coefficient;
    }

    const defaultProbability = 1 / (1 + Math.exp(-score));
    const threshold = model.threshold ?? 0.5;

    return {
      defaultProbability: Number(defaultProbability.toFixed(4)),
      threshold,
      riskLabel: defaultProbability >= threshold ? 'HIGH' : 'LOW',
      modelType: 'logistic_regression_classifier',
      features,
    };
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

  private toNumber(value?: number) {
    const numberValue = Number(value);
    if (!Number.isFinite(numberValue)) {
      throw new NotFoundException('Application has incomplete risk features');
    }
    return numberValue;
  }
}
