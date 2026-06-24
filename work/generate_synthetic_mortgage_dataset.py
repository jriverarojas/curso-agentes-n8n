import argparse
import numpy as np
import pandas as pd


def clip(values, low, high):
    return np.minimum(np.maximum(values, low), high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--output", default="synthetic_mortgage_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = args.rows

    net_monthly_income = clip(rng.normal(8500, 3200, rows), 2500, 30000)
    property_value = clip(net_monthly_income * rng.normal(70, 18, rows), 120000, 1800000)
    requested_amount = property_value * clip(rng.normal(0.72, 0.13, rows), 0.35, 0.98)
    requested_term_months = rng.choice([120, 180, 240, 300], rows, p=[0.15, 0.25, 0.45, 0.15])
    monthly_debt_payment = net_monthly_income * clip(rng.normal(0.22, 0.14, rows), 0, 0.75)
    monthly_expenses = net_monthly_income * clip(rng.normal(0.42, 0.16, rows), 0.12, 0.85)
    active_loan_count = rng.poisson(1.2, rows)
    has_late_payments = rng.binomial(1, clip(0.08 + active_loan_count * 0.04, 0.05, 0.45))
    employment_tenure_months = clip(rng.gamma(3.0, 18.0, rows), 1, 240)
    average_monthly_balance = net_monthly_income * clip(rng.normal(1.15, 0.9, rows), 0.02, 5.5)

    estimated_monthly_payment = requested_amount / requested_term_months
    debt_to_income_ratio = monthly_debt_payment / net_monthly_income
    loan_to_value_ratio = requested_amount / property_value
    payment_to_income_ratio = estimated_monthly_payment / net_monthly_income
    expense_to_income_ratio = monthly_expenses / net_monthly_income
    total_obligations_to_income_ratio = (
        monthly_debt_payment + monthly_expenses + estimated_monthly_payment
    ) / net_monthly_income

    employment_stability_score = np.select(
        [
            employment_tenure_months >= 60,
            employment_tenure_months >= 24,
            employment_tenure_months >= 12,
        ],
        [100, 80, 60],
        default=35,
    )
    banking_capacity_score = np.select(
        [
            average_monthly_balance / net_monthly_income >= 3,
            average_monthly_balance / net_monthly_income >= 1,
            average_monthly_balance / net_monthly_income >= 0.3,
        ],
        [100, 75, 55],
        default=30,
    )
    credit_history_score = clip(100 - has_late_payments * 35 - active_loan_count * 8, 0, 100)

    risk_signal = (
        2.4 * debt_to_income_ratio
        + 2.0 * payment_to_income_ratio
        + 1.8 * total_obligations_to_income_ratio
        + 0.8 * expense_to_income_ratio
        + 1.7 * (loan_to_value_ratio > 0.85)
        + 1.2 * has_late_payments
        + 0.5 * (active_loan_count >= 3)
        - 0.012 * employment_stability_score
        - 0.007 * banking_capacity_score
        - 0.009 * credit_history_score
        + rng.normal(0, 0.25, rows)
    )
    default_probability = 1 / (1 + np.exp(-(risk_signal - 0.9)))
    default_flag = rng.binomial(1, clip(default_probability, 0.02, 0.85))

    affordability_amount = (net_monthly_income * 0.35 - monthly_debt_payment) * requested_term_months
    collateral_amount = property_value * 0.8
    history_factor = clip(credit_history_score / 100, 0.35, 1.0)
    recommended_amount = clip(
        np.minimum.reduce([requested_amount, affordability_amount, collateral_amount]) * history_factor,
        0,
        requested_amount,
    )
    recommended_amount = np.round(recommended_amount / 1000) * 1000

    df = pd.DataFrame(
        {
            "application_id": [f"APP-{i:06d}" for i in range(rows)],
            "net_monthly_income": np.round(net_monthly_income, 2),
            "monthly_debt_payment": np.round(monthly_debt_payment, 2),
            "monthly_expenses": np.round(monthly_expenses, 2),
            "property_value": np.round(property_value, 2),
            "requested_amount": np.round(requested_amount, 2),
            "requested_term_months": requested_term_months,
            "active_loan_count": active_loan_count,
            "has_late_payments": has_late_payments,
            "employment_tenure_months": np.round(employment_tenure_months).astype(int),
            "average_monthly_balance": np.round(average_monthly_balance, 2),
            "estimated_monthly_payment": np.round(estimated_monthly_payment, 2),
            "debt_to_income_ratio": np.round(debt_to_income_ratio, 4),
            "loan_to_value_ratio": np.round(loan_to_value_ratio, 4),
            "payment_to_income_ratio": np.round(payment_to_income_ratio, 4),
            "expense_to_income_ratio": np.round(expense_to_income_ratio, 4),
            "total_obligations_to_income_ratio": np.round(total_obligations_to_income_ratio, 4),
            "employment_stability_score": employment_stability_score,
            "banking_capacity_score": banking_capacity_score,
            "credit_history_score": credit_history_score,
            "default_flag": default_flag,
            "recommended_amount": recommended_amount,
        }
    )

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
