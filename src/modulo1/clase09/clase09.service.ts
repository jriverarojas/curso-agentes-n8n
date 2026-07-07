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