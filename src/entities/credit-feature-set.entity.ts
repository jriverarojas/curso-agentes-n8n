import {
  Column,
  CreateDateColumn,
  Entity,
  PrimaryGeneratedColumn,
  UpdateDateColumn,
} from 'typeorm';

@Entity({ name: 'credit_feature_sets' })
export class CreditFeatureSet {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ name: 'application_id', type: 'uuid', unique: true })
  applicationId: string;

  @Column({ name: 'debt_to_income_ratio', type: 'numeric', nullable: true })
  debtToIncomeRatio?: number;

  @Column({ name: 'loan_to_value_ratio', type: 'numeric', nullable: true })
  loanToValueRatio?: number;

  @Column({ name: 'payment_to_income_ratio', type: 'numeric', nullable: true })
  paymentToIncomeRatio?: number;

  @Column({ name: 'expense_to_income_ratio', type: 'numeric', nullable: true })
  expenseToIncomeRatio?: number;

  @Column({ name: 'total_obligations_to_income_ratio', type: 'numeric', nullable: true })
  totalObligationsToIncomeRatio?: number;

  @Column({ name: 'employment_stability_score', type: 'numeric', nullable: true })
  employmentStabilityScore?: number;

  @Column({ name: 'banking_capacity_score', type: 'numeric', nullable: true })
  bankingCapacityScore?: number;

  @Column({ name: 'credit_history_score', type: 'numeric', nullable: true })
  creditHistoryScore?: number;

  @Column({ name: 'synthetic_risk_label', type: 'integer', nullable: true })
  syntheticRiskLabel?: number;

  @Column({ name: 'features_payload', type: 'jsonb', default: {} })
  featuresPayload: Record<string, unknown>;

  @Column({ name: 'schema_payload', type: 'jsonb', default: {} })
  schemaPayload: Record<string, unknown>;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt: Date;

  @UpdateDateColumn({ name: 'updated_at', type: 'timestamptz' })
  updatedAt: Date;
}
