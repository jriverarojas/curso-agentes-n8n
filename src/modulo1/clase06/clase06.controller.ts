import { Body, Controller, Get, Param, Post, UseGuards } from '@nestjs/common';
import { ApiKeyGuard } from '../../auth/guards/api-key.guard';
import { Clase06Service } from './clase06.service';

@Controller('modulo1/clase06')
@UseGuards(ApiKeyGuard)
export class Clase06Controller {
  constructor(private readonly clase06: Clase06Service) {}

  @Get('models/risk/metrics')
  async getRiskMetrics() {
    return await this.clase06.getRiskMetrics();
  }

  @Post('applications/:applicationId/risk')
  async predictApplicationRisk(@Param('applicationId') applicationId: string) {
    return await this.clase06.predictApplicationRisk(applicationId);
  }

  @Post('models/risk/predict')
  async predictRisk(
    @Body()
    body: {
      debt_to_income_ratio: number;
      loan_to_value_ratio: number;
      payment_to_income_ratio: number;
      expense_to_income_ratio: number;
      total_obligations_to_income_ratio: number;
      employment_stability_score: number;
      banking_capacity_score: number;
      credit_history_score: number;
    },
  ) {
    return await this.clase06.predictRisk(body);
  }
}
