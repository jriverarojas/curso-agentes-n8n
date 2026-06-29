import { Body, Controller, Get, Param, Post, UseGuards } from '@nestjs/common';
import { ApiKeyGuard } from '../../auth/guards/api-key.guard';
import { Clase07Service } from './clase07.service';

@Controller('modulo1/clase07')
@UseGuards(ApiKeyGuard)
export class Clase07Controller {
  constructor(private readonly clase07: Clase07Service) {}

  @Get('models/amount/metrics')
  async getAmountMetrics() {
    return await this.clase07.getAmountMetrics();
  }

  @Get('models/compare')
  async compareModels() {
    return await this.clase07.compareModels();
  }

  @Post('applications/:applicationId/amount')
  async recommendApplicationAmount(@Param('applicationId') applicationId: string) {
    return await this.clase07.recommendApplicationAmount(applicationId);
  }

  @Post('models/amount/predict')
  async predictAmount(@Body() body: Record<string, number>) {
    return await this.clase07.predictAmount(body);
  }
}
