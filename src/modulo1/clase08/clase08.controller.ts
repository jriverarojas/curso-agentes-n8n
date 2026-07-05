import { Body, Controller, Param, Post, UseGuards } from '@nestjs/common';
import { ApiKeyGuard } from '../../auth/guards/api-key.guard';
import { Clase08Service } from './clase08.service';

@Controller('modulo1/clase08')
@UseGuards(ApiKeyGuard)
export class Clase08Controller {
  constructor(private readonly clase08: Clase08Service) {}

  @Post('credit-files/:applicationId/evaluate')
  async evaluateCreditFile(@Param('applicationId') applicationId: string) {
    return await this.clase08.evaluateCreditFile(applicationId);
  }

  @Post('models/risk')
  async predictRisk(@Body() body: { features: Record<string, number> }) {
    return await this.clase08.predictRisk(body.features);
  }

  @Post('models/amount')
  async predictAmount(@Body() body: { features: Record<string, number> }) {
    return await this.clase08.predictAmount(body.features);
  }
}