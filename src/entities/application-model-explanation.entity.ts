import { Column, CreateDateColumn, Entity, PrimaryGeneratedColumn } from 'typeorm';

@Entity({ name: 'application_model_explanations' })
export class ApplicationModelExplanation {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ name: 'application_id', type: 'uuid' })
  applicationId: string;

  @Column({ name: 'explanation_type', type: 'text' })
  explanationType: string;

  @Column({ name: 'model_type', type: 'text' })
  modelType: string;

  @Column({ name: 'explanation_payload', type: 'jsonb', default: {} })
  explanationPayload: Record<string, unknown>;

  @Column({ name: 's3_key', type: 'text', nullable: true })
  s3Key?: string;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt: Date;
}