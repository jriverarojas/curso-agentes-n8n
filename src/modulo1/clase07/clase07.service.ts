import { GetObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { CreditFeatureSet } from '../../entities/credit-feature-set.entity';

type AmountFeatures = Record<string, number>;

type XGBoostTreeNode = {
  nodeid: number;
  split?: string;
  split_condition?: number;
  yes?: number;
  no?: number;
  missing?: number;
  children?: XGBoostTreeNode[];
  leaf?: number;
};

type AmountModelJson = {
  model_type: 'xgboost_regressor_tree_dump';
  target: 'recommended_amount';
  features: string[];
  base_score: number;
  trees: XGBoostTreeNode[];
  prediction_clip?: {
    min?: number;
    max?: number;
  };
};

@Injectable()
export class Clase07Service {
  private readonly s3: S3Client;

  constructor(
    private readonly config: ConfigService,
    @InjectRepository(CreditFeatureSet)
    private readonly featureSets: Repository<CreditFeatureSet>,
  ) {
    this.s3 = new S3Client({
      region: this.config.getOrThrow<string>('AWS_REGION'),
      credentials: {
        accessKeyId: this.config.getOrThrow<string>('AWS_ACCESS_KEY_ID'),
        secretAccessKey: this.config.getOrThrow<string>('AWS_SECRET_ACCESS_KEY'),
      },
    });
  }

  async getAmountMetrics() {
    return await this.readJson(
      this.config.getOrThrow<string>('SAGEMAKER_AMOUNT_METRICS_KEY'),
    );
  }

  async compareModels() {
    return {
      riskModel: await this.readJson(
        this.config.getOrThrow<string>('SAGEMAKER_RISK_METRICS_KEY'),
      ),
      amountModel: await this.getAmountMetrics(),
    };
  }

  async recommendApplicationAmount(applicationId: string) {
    const featureSet = await this.featureSets.findOne({
      where: { applicationId },
    });

    if (!featureSet) {
      throw new NotFoundException('Feature set not found for this application');
    }

    const payload = featureSet.featuresPayload ?? {};
    const features = this.pickAmountFeatures({
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

    return await this.predictAmount(features);
  }

  async predictAmount(features: AmountFeatures) {
    const model = (await this.readJson(
      this.config.getOrThrow<string>('SAGEMAKER_AMOUNT_MODEL_KEY'),
    )) as AmountModelJson;

    let prediction = model.base_score;
    for (const tree of model.trees) {
      prediction += this.evaluateTree(tree, features);
    }

    const min = model.prediction_clip?.min;
    const max = model.prediction_clip?.max;
    if (typeof min === 'number') {
      prediction = Math.max(min, prediction);
    }
    if (typeof max === 'number') {
      prediction = Math.min(max, prediction);
    }

    return {
      recommendedAmount: Math.round(prediction),
      modelType: model.model_type,
      features,
    };
  }

  private pickAmountFeatures(source: Record<string, unknown>) {
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

  private evaluateTree(node: XGBoostTreeNode, features: AmountFeatures): number {
    if (typeof node.leaf === 'number') {
      return node.leaf;
    }

    if (!node.children || !node.split) {
      throw new NotFoundException('Invalid amount model tree');
    }

    const rawValue = features[node.split];
    const nextNodeId =
      !Number.isFinite(rawValue) && node.missing !== undefined
        ? node.missing
        : rawValue < (node.split_condition ?? 0)
          ? node.yes
          : node.no;

    const child = node.children.find(
      (candidate) => candidate.nodeid === nextNodeId,
    );
    if (!child) {
      throw new NotFoundException('Invalid amount model tree path');
    }

    return this.evaluateTree(child, features);
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
      throw new NotFoundException('Application has incomplete amount features');
    }
    return numberValue;
  }
}