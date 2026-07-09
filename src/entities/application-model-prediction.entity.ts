import { Column, CreateDateColumn, Entity, PrimaryGeneratedColumn } from 'typeorm';

@Entity({ name: 'application_model_predictions' })
export class ApplicationModelPrediction {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ name: 'application_id', type: 'uuid' })
  applicationId: string;

  @Column({ name: 'prediction_type', type: 'text' })
  predictionType: string;

  @Column({ name: 'model_type', type: 'text' })
  modelType: string;

  @Column({ name: 'result_payload', type: 'jsonb', default: {} })
  resultPayload: Record<string, unknown>;

  @Column({ name: 'features_payload', type: 'jsonb', default: {} })
  featuresPayload: Record<string, unknown>;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt: Date;
}