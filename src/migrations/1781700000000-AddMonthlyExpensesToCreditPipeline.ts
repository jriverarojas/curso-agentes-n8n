import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddMonthlyExpensesToCreditPipeline1781700000000
  implements MigrationInterface
{
  name = 'AddMonthlyExpensesToCreditPipeline1781700000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;

    await queryRunner.query(`
      ALTER TABLE ${q}."clean_credit_profiles"
      ADD COLUMN IF NOT EXISTS "monthly_expenses" numeric(14,2)
    `);

    await queryRunner.query(`
      ALTER TABLE ${q}."credit_feature_sets"
      ADD COLUMN IF NOT EXISTS "expense_to_income_ratio" numeric(8,4)
    `);

    await queryRunner.query(`
      ALTER TABLE ${q}."credit_feature_sets"
      ADD COLUMN IF NOT EXISTS "total_obligations_to_income_ratio" numeric(8,4)
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    const schema = process.env.DATABASE_SCHEMA ?? 'public';
    const q = `"${schema}"`;

    await queryRunner.query(`
      ALTER TABLE ${q}."credit_feature_sets"
      DROP COLUMN IF EXISTS "total_obligations_to_income_ratio"
    `);

    await queryRunner.query(`
      ALTER TABLE ${q}."credit_feature_sets"
      DROP COLUMN IF EXISTS "expense_to_income_ratio"
    `);

    await queryRunner.query(`
      ALTER TABLE ${q}."clean_credit_profiles"
      DROP COLUMN IF EXISTS "monthly_expenses"
    `);
  }
}

