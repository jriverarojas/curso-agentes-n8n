import { Injectable } from '@nestjs/common';
import { Clase06Service } from '../clase06/clase06.service';
import { Clase07Service } from '../clase07/clase07.service';

type RiskPrediction = {
  defaultProbability: number;
  riskLabel: 'HIGH' | 'LOW';
  threshold: number;
  modelType: string;
  features: Record<string, number>;
};

type AmountPrediction = {
  recommendedAmount: number;
  modelType: string;
  features: Record<string, number>;
};

@Injectable()
export class Clase08Service {
  constructor(
    private readonly clase06: Clase06Service,
    private readonly clase07: Clase07Service,
  ) {}

  async evaluateCreditFile(applicationId: string) {
    const risk = (await this.clase06.predictApplicationRisk(
      applicationId,
    )) as RiskPrediction;

    const amount = (await this.clase07.recommendApplicationAmount(
      applicationId,
    )) as AmountPrediction;

    const requestedAmount = Number(amount.features.requested_amount ?? 0);

    const decision = this.makeDecision(
      risk.defaultProbability,
      requestedAmount,
      amount.recommendedAmount,
    );

    return {
      applicationId,
      risk: {
        defaultProbability: risk.defaultProbability,
        threshold: risk.threshold,
        riskLabel: risk.riskLabel,
        modelType: risk.modelType,
      },
      amount: {
        requestedAmount,
        recommendedAmount: amount.recommendedAmount,
        modelType: amount.modelType,
      },
      decision,
      reasons: this.buildReasons(
        risk.defaultProbability,
        risk.riskLabel,
        requestedAmount,
        amount.recommendedAmount,
      ),
    };
  }

  async predictRisk(features: Record<string, number>) {
    return await this.clase06.predictRisk({
      debt_to_income_ratio: Number(features.debt_to_income_ratio),
      loan_to_value_ratio: Number(features.loan_to_value_ratio),
      payment_to_income_ratio: Number(features.payment_to_income_ratio),
      expense_to_income_ratio: Number(features.expense_to_income_ratio),
      total_obligations_to_income_ratio: Number(
        features.total_obligations_to_income_ratio,
      ),
      employment_stability_score: Number(features.employment_stability_score),
      banking_capacity_score: Number(features.banking_capacity_score),
      credit_history_score: Number(features.credit_history_score),
    });
  }

  async predictAmount(features: Record<string, number>) {
    return await this.clase07.predictAmount(features);
  }

  private makeDecision(
    defaultProbability: number,
    requestedAmount: number,
    recommendedAmount: number,
  ) {
    if (defaultProbability >= 0.6) {
      return 'REJECT_OR_MANUAL_REVIEW';
    }

    if (requestedAmount > recommendedAmount * 1.1) {
      return 'REVIEW_AMOUNT';
    }

    return 'PRE_APPROVE_FOR_REVIEW';
  }

  private buildReasons(
    defaultProbability: number,
    riskLabel: string,
    requestedAmount: number,
    recommendedAmount: number,
  ) {
    const reasons: string[] = [];

    if (riskLabel === 'HIGH') {
      reasons.push('El modelo de riesgo clasifico la solicitud como HIGH.');
    }

    if (defaultProbability >= 0.6) {
      reasons.push('La probabilidad de incumplimiento supera el 60%.');
    }

    if (requestedAmount > recommendedAmount * 1.1) {
      reasons.push('El monto solicitado supera en mas de 10% al monto recomendado.');
    }

    if (reasons.length === 0) {
      reasons.push('Riesgo y monto recomendado se mantienen dentro de rangos de revision.');
    }

    return reasons;
  }
}