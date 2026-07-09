import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AuthModule } from '../auth/auth.module';
import { ApplicationDocument } from '../entities/application-document.entity';
import { ApplicationExtractedData } from '../entities/application-extracted-data.entity';
import { CreditApplication } from '../entities/credit-application.entity';
import { DocumentType } from '../entities/document-type.entity';
import { RawDocumentText } from '../entities/raw-document-text.entity';
import { TextractQueryAnswer } from '../entities/textract-query-answer.entity';
import { TextractResult } from '../entities/textract-result.entity';
import { Clase01Controller } from './clase01/clase01.controller';
import { Clase01Service } from './clase01/clase01.service';
import { TextractService } from './clase01/textract.service';
import { Clase02Controller } from './clase02/clase02.controller';
import { Clase02Service } from './clase02/clase02.service';
import { Clase03Controller } from './clase03/clase03.controller';
import { Clase03Service } from './clase03/clase03.service';

import { CleanCreditProfile } from '../entities/clean-credit-profile.entity';
import { GlueJobRunEntity } from '../entities/glue-job-run.entity';
import { Clase04Controller } from './clase04/clase04.controller';
import { Clase04Service } from './clase04/clase04.service';
import { GlueService } from './clase04/glue.service';
import { ApplicationEvaluation } from '../entities/application-evaluation.entity';
import { ApplicationModelPrediction } from '../entities/application-model-prediction.entity';
import { ApplicationModelExplanation } from '../entities/application-model-explanation.entity';

import { CreditFeatureSet } from '../entities/credit-feature-set.entity';
import { Clase05Controller } from './clase05/clase05.controller';
import { Clase05Service } from './clase05/clase05.service';
import { Clase06Controller } from './clase06/clase06.controller';
import { Clase06Service } from './clase06/clase06.service';
import { Clase07Controller } from './clase07/clase07.controller';
import { Clase07Service } from './clase07/clase07.service';
import { Clase08Controller } from './clase08/clase08.controller';
import { Clase08Service } from './clase08/clase08.service';
import { Clase09Controller } from './clase09/clase09.controller';
import { Clase09Service } from './clase09/clase09.service';
import { Clase10Controller } from './clase10/clase10.controller';
import { Clase10Service } from './clase10/clase10.service';


@Module({
  imports: [
    AuthModule,
    TypeOrmModule.forFeature([
      RawDocumentText,
      CreditApplication,
      ApplicationDocument,
      DocumentType,
      TextractResult,
      TextractQueryAnswer,
      ApplicationExtractedData,
      CleanCreditProfile,
      GlueJobRunEntity,
      CreditFeatureSet,
      ApplicationEvaluation,
      ApplicationModelPrediction,
      ApplicationModelExplanation,
    ]),
  ],
  controllers: [
    Clase01Controller,
    Clase02Controller,
    Clase03Controller,
    Clase04Controller,
    Clase05Controller,
    Clase06Controller,
    Clase07Controller,
    Clase08Controller,
    Clase09Controller,
    Clase10Controller,
  ],
  providers: [
    Clase01Service,
    Clase02Service,
    Clase03Service,
    Clase04Service,
    Clase05Service,
    Clase06Service,
    Clase07Service,
    Clase08Service,
    Clase09Service,
    Clase10Service,
    TextractService,
    GlueService,
  ],
  exports: [Clase03Service],
})
export class Modulo1Module {}
