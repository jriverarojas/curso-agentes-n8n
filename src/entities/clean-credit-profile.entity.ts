import {
  Column,
  CreateDateColumn,
  Entity,
  PrimaryGeneratedColumn,
  UpdateDateColumn,
} from 'typeorm';

@Entity({ name: 'clean_credit_profiles' })
export class CleanCreditProfile {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ name: 'application_id', type: 'uuid', unique: true })
  applicationId: string;

  @Column({ name: 'applicant_name', type: 'text', nullable: true })
  applicantName?: string;

  @Column({ name: 'identity_number', type: 'text', nullable: true })
  identityNumber?: string;

  @Column({ name: 'birth_date', type: 'date', nullable: true })
  birthDate?: string;

  @Column({ name: 'employer_name', type: 'text', nullable: true })
  employerName?: string;

  @Column({ name: 'job_title', type: 'text', nullable: true })
  jobTitle?: string;

  @Column({ name: 'employment_tenure_months', type: 'integer', nullable: true })
  employmentTenureMonths?: number;

  @Column({ name: 'net_monthly_income', type: 'numeric', nullable: true })
  netMonthlyIncome?: number;

  @Column({ name: 'gross_monthly_income', type: 'numeric', nullable: true })
  grossMonthlyIncome?: number;

  @Column({ name: 'average_monthly_balance', type: 'numeric', nullable: true })
  averageMonthlyBalance?: number;

  @Column({ name: 'requested_amount', type: 'numeric', nullable: true })
  requestedAmount?: number;

  @Column({ name: 'requested_term_months', type: 'integer', nullable: true })
  requestedTermMonths?: number;

  @Column({ name: 'property_value', type: 'numeric', nullable: true })
  propertyValue?: number;

  @Column({ name: 'reported_total_debt', type: 'numeric', nullable: true })
  reportedTotalDebt?: number;

  @Column({ name: 'monthly_debt_payment', type: 'numeric', nullable: true })
  monthlyDebtPayment?: number;

  @Column({ name: 'monthly_expenses', type: 'numeric', nullable: true })
  monthlyExpenses?: number;

  @Column({ name: 'active_loan_count', type: 'integer', nullable: true })
  activeLoanCount?: number;

  @Column({ name: 'has_late_payments', type: 'boolean', nullable: true })
  hasLatePayments?: boolean;

  @Column({ name: 'estimated_monthly_payment', type: 'numeric', nullable: true })
  estimatedMonthlyPayment?: number;

  @Column({ name: 'initial_debt_to_income_ratio', type: 'numeric', nullable: true })
  initialDebtToIncomeRatio?: number;

  @Column({ name: 'clean_payload', type: 'jsonb', default: {} })
  cleanPayload: Record<string, unknown>;

  @Column({ name: 'quality_report', type: 'jsonb', default: {} })
  qualityReport: Record<string, unknown>;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt: Date;

  @UpdateDateColumn({ name: 'updated_at', type: 'timestamptz' })
  updatedAt: Date;
}
