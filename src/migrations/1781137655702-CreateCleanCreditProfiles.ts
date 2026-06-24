import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreateCleanCreditProfiles1781137655702
  implements MigrationInterface
{
  name = 'CreateCleanCreditProfiles1781137655702';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;

    await queryRunner.query(`
      CREATE TABLE ${q}."clean_credit_profiles" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        "application_id" uuid NOT NULL UNIQUE,
        "applicant_name" text,
        "identity_number" text,
        "birth_date" date,
        "employer_name" text,
        "job_title" text,
        "employment_tenure_months" integer,
        "net_monthly_income" numeric(14,2),
        "gross_monthly_income" numeric(14,2),
        "average_monthly_balance" numeric(14,2),
        "requested_amount" numeric(14,2),
        "requested_term_months" integer,
        "property_value" numeric(14,2),
        "reported_total_debt" numeric(14,2),
        "monthly_debt_payment" numeric(14,2),
        "monthly_expenses" numeric(14,2),
        "active_loan_count" integer,
        "has_late_payments" boolean,
        "estimated_monthly_payment" numeric(14,2),
        "initial_debt_to_income_ratio" numeric(8,4),
        "clean_payload" jsonb NOT NULL DEFAULT '{}'::jsonb,
        "quality_report" jsonb NOT NULL DEFAULT '{}'::jsonb,
        "created_at" timestamptz NOT NULL DEFAULT now(),
        "updated_at" timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT "FK_clean_credit_profiles_application"
          FOREIGN KEY ("application_id") REFERENCES ${q}."credit_applications"("id")
      )
    `);

    await queryRunner.query(`
      CREATE TABLE ${q}."glue_job_runs" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        "application_id" uuid NOT NULL,
        "job_name" text NOT NULL,
        "job_run_id" text NOT NULL,
        "job_type" text NOT NULL,
        "status" text NOT NULL DEFAULT 'STARTING',
        "input_path" text,
        "output_path" text,
        "created_at" timestamptz NOT NULL DEFAULT now(),
        "updated_at" timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT "FK_glue_job_runs_application"
          FOREIGN KEY ("application_id") REFERENCES ${q}."credit_applications"("id")
      )
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;
    await queryRunner.query(`DROP TABLE IF EXISTS ${q}."glue_job_runs"`);
    await queryRunner.query(`DROP TABLE IF EXISTS ${q}."clean_credit_profiles"`);
  }

}
