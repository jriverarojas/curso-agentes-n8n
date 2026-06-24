import { Body, Controller, Get, Param, Post, UseGuards } from '@nestjs/common';
import { ApiKeyGuard } from '../../auth/guards/api-key.guard';
import { Clase05Service } from './clase05.service';

@Controller('modulo1/clase05')
@UseGuards(ApiKeyGuard)
export class Clase05Controller {
  constructor(private readonly clase05: Clase05Service) {}

  @Post('credit-files/features')
  async generateFeatures(@Body() body: { applicationId: string }) {
    return await this.clase05.generateFeatures(body);
  }

  @Get('credit-files/:applicationId/features-status')
  async getFeaturesStatus(@Param('applicationId') applicationId: string) {
    return await this.clase05.getFeaturesStatus(applicationId);
  }

  @Get('credit-files/:applicationId/features')
  async getFeatures(@Param('applicationId') applicationId: string) {
    return await this.clase05.getFeatures(applicationId);
  }
}