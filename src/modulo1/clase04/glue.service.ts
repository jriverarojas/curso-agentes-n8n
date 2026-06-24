import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
  GetJobRunCommand,
  GlueClient,
  StartJobRunCommand,
} from '@aws-sdk/client-glue';

@Injectable()
export class GlueService {
  private readonly client: GlueClient;

  constructor(private readonly config: ConfigService) {
    this.client = new GlueClient({
      region: this.config.getOrThrow<string>('AWS_REGION'),
      credentials: {
        accessKeyId: this.config.getOrThrow<string>('AWS_ACCESS_KEY_ID'),
        secretAccessKey: this.config.getOrThrow<string>('AWS_SECRET_ACCESS_KEY'),
      },
    });
  }

  async startCleanJob(args: {
    applicationId: string;
    inputKey: string;
    outputKey: string;
  }) {
    const jobName = this.config.getOrThrow<string>('AWS_GLUE_CLEAN_JOB_NAME');
    const bucket = this.config.getOrThrow<string>('AWS_S3_BUCKET');

    const command = new StartJobRunCommand({
      JobName: jobName,
      Arguments: {
        '--BUCKET': bucket,
        '--APPLICATION_ID': args.applicationId,
        '--INPUT_KEY': args.inputKey,
        '--OUTPUT_KEY': args.outputKey,
        '--CONFIDENCE_THRESHOLD': '70',
      },
    });

    const response = await this.client.send(command);
    return {
      jobName,
      jobRunId: response.JobRunId!,
    };
  }

  async getJobStatus(jobName: string, jobRunId: string) {
    const response = await this.client.send(
      new GetJobRunCommand({
        JobName: jobName,
        RunId: jobRunId,
        PredecessorsIncluded: false,
      }),
    );

    return response.JobRun?.JobRunState ?? 'UNKNOWN';
  }
  
  async startFeaturesJob(args: {
    applicationId: string;
    inputKey: string;
    outputKey: string;
  }) {
    const jobName = this.config.getOrThrow<string>('AWS_GLUE_FEATURES_JOB_NAME');

    const command = new StartJobRunCommand({
      JobName: jobName,
      Arguments: {
        '--BUCKET': this.config.getOrThrow<string>('AWS_S3_BUCKET'),
        '--APPLICATION_ID': args.applicationId,
        '--INPUT_KEY': args.inputKey,
        '--OUTPUT_KEY': args.outputKey,
      },
    });

    const response = await this.client.send(command);
    return {
      jobName,
      jobRunId: response.JobRunId!,
    };
  }
}
