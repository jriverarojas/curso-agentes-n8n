import { Controller, Get, Param, Post, UseGuards } from '@nestjs/common';
import { ApiKeyGuard } from '../../auth/guards/api-key.guard';
import { Clase09Service } from './clase09.service';

@Controller('modulo1/clase09')
@UseGuards(ApiKeyGuard)
export class Clase09Controller {
  constructor(private readonly clase09: Clase09Service) {}

  @Get('explanations/umbrella')
  async getUmbrellaExplanation() {
    return await this.clase09.getUmbrellaExplanation();
  }

  @Get('explanations/risk/global')
  async getRiskGlobalExplanation() {
    return await this.clase09.getRiskGlobalExplanation();
  }

  @Get('explanations/amount/global')
  async getAmountGlobalExplanation() {
    return await this.clase09.getAmountGlobalExplanation();
  }

  @Get('explanations/compare')
  async compareExplanations() {
    return await this.clase09.compareExplanations();
  }

  @Get('applications/:applicationId/explanation')
  async getApplicationExplanation(@Param('applicationId') applicationId: string) {
    return await this.clase09.getApplicationExplanation(applicationId);
  }

  @Post('applications/:applicationId/explanation/generate')
  async generateApplicationExplanation(
    @Param('applicationId') applicationId: string,
  ) {
    return await this.clase09.generateApplicationExplanation(applicationId);
  }
}