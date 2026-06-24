import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreateCreditFeatureSets1781572215811
  implements MigrationInterface
{
  name = 'CreateCreditFeatureSets1781572215811';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;

    await queryRunner.query(`
      CREATE TABLE ${q}."credit_feature_sets" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        "application_id" uuid NOT NULL UNIQUE,
        "debt_to_income_ratio" numeric(8,4),
        "loan_to_value_ratio" numeric(8,4),
        "payment_to_income_ratio" numeric(8,4),
        "expense_to_income_ratio" numeric(8,4),
        "total_obligations_to_income_ratio" numeric(8,4),
        "employment_stability_score" numeric(8,2),
        "banking_capacity_score" numeric(8,2),
        "credit_history_score" numeric(8,2),
        "synthetic_risk_label" integer,
        "features_payload" jsonb NOT NULL DEFAULT '{}'::jsonb,
        "schema_payload" jsonb NOT NULL DEFAULT '{}'::jsonb,
        "created_at" timestamptz NOT NULL DEFAULT now(),
        "updated_at" timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT "FK_credit_feature_sets_application"
          FOREIGN KEY ("application_id") REFERENCES ${q}."credit_applications"("id")
      )
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;
    await queryRunner.query(`DROP TABLE IF EXISTS ${q}."credit_feature_sets"`);
  }
}
