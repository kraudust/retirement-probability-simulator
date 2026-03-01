import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.stats import t as t_dist
import yaml


# ==============================
# CONFIG STRUCTURES
# ==============================

@dataclass
class RetirementResult:
    retirement_age: int
    success_probability: float


# ==============================
# UTILITY FUNCTIONS
# ==============================

def real_return(nominal, inflation):
    return (1 + nominal) / (1 + inflation) - 1


def monthly_rate(annual_rate):
    return (1 + annual_rate) ** (1/12) - 1


def random_death_age(mean=90, std=8, min_age=70, max_age=100):
    age = int(np.random.normal(mean, std))
    return max(min(age, max_age), min_age)


# Approximate Social Security adjustment factors
def ss_benefit_factor(claim_age, full_retirement_age=67):
    months_diff = round((claim_age - full_retirement_age) * 12)
    
    if months_diff >= 0:
        # Delayed: 8% per year (simple)
        return 1 + 0.08 * (months_diff / 12)
    else:
        # Early: 5/9% per month for first 36 months, 5/12% beyond that
        months_early = abs(months_diff)
        if months_early <= 36:
            reduction = (5/9 * 0.01) * months_early
        else:
            reduction = (5/9 * 0.01) * 36 + (5/12 * 0.01) * (months_early - 36)
        return 1 - reduction


# Mild: -0.08/12, vol_mult 1.5 — models a correction, not a crash
# Moderate: -0.12/12, vol_mult 2.0 — models a typical bear market
# Severe: -0.20/12, vol_mult 2.5 — models 2008-style crashes
REGIMES = {
    'normal': {
        'return_boost': 0.0,
        'vol_mult': 1.0,
        'p_switch': 0.015
    },
    'crisis': {
        'return_boost': -0.12 / 12,
        'vol_mult': 2.0,
        'p_switch': 0.15
    }
}

# ==============================
# CORE LIFE SIMULATION
# ==============================
def simulate_life(
    current_age,
    retirement_age,
    roth,
    traditional,
    brokerage,
    cash,
    annual_roth_contribution,
    annual_traditional_contribution,
    annual_brokerage_contribution,
    annual_cash_contribution,
    initial_annual_expenses,
    stock_return,
    bond_return,
    stock_volatility,
    bond_volatility,
    inflation,
    cash_return,
    tax_rate,
    ss_income,
    ss_claim_age,
    mean_death_age,
    death_age_std,
    death_age_min,
    death_age_max,
    spending_decline_start_age,
    annual_spending_decline_rate,
    spending_decline_end_age,
    annual_healthcare_increase_rate,
    guardrail_cut_threshold,
    guardrail_cut_amount,
    guardrail_cut_floor,
    guardrail_raise_threshold,
    guardrail_raise_amount,
    guardrail_raise_ceiling,
    stock_bond_correlation,
    glide_path,
    glide_path_start_stock_pct,
    glide_path_end_stock_pct,
    glide_path_years,
    static_stock_allocation,
    return_distribution_degrees_of_freedom,
):
    regime = 'normal'

    death_age = random_death_age(
        mean=mean_death_age,
        std=death_age_std,
        min_age=death_age_min,
        max_age=death_age_max
    )
    living_months = (death_age - current_age) * 12
    retirement_month = (retirement_age - current_age) * 12

    real_stock_return = real_return(stock_return, inflation)
    real_bond_return = real_return(bond_return, inflation)
    real_cash_return = real_return(cash_return, inflation)

    monthly_stock_rate = monthly_rate(real_stock_return)
    monthly_bond_rate = monthly_rate(real_bond_return)
    monthly_cash_rate = monthly_rate(real_cash_return)

    base_monthly_expense = initial_annual_expenses / 12
    spending_multiplier = 1.0
    peak_portfolio = roth + traditional + brokerage + cash
    monthly_roth_contrib = annual_roth_contribution / 12
    monthly_traditional_contrib = annual_traditional_contribution / 12
    monthly_brokerage_contrib = annual_brokerage_contribution / 12
    monthly_cash_contrib = annual_cash_contribution / 12
    monthly_ss = ss_income / 12

    min_portfolio = float("inf")

    # Pre-generate all random numbers for this life simulation
    df = return_distribution_degrees_of_freedom
    t_scale = np.sqrt((df - 2) / df)
    stock_shocks_raw = t_dist.rvs(df=df, size=living_months) * t_scale * (stock_volatility / np.sqrt(12))
    bond_randoms = np.random.normal(0, bond_volatility / np.sqrt(12), size=living_months)
    regime_rolls = np.random.random(size=living_months)
    ss_start_month = (ss_claim_age - current_age) * 12

    # Derive glide path slope from config
    glide_path_slope = (glide_path_start_stock_pct - glide_path_end_stock_pct) / glide_path_years

    initial_withdrawal_rate = None

    for month in range(living_months):
        age = current_age + month / 12

        if glide_path:
            years_since_retirement = max(0, age - retirement_age)
            stock_allocation = max(
                glide_path_end_stock_pct,
                glide_path_start_stock_pct - glide_path_slope * years_since_retirement
            )
        else:
            stock_allocation = static_stock_allocation

        bond_allocation = 1 - stock_allocation

        r = REGIMES[regime]
        if regime_rolls[month] < r['p_switch']:
            regime = 'crisis' if regime == 'normal' else 'normal'
        r = REGIMES[regime]

        raw_shock = stock_shocks_raw[month]
        stock_shock = raw_shock * r['vol_mult']
        bond_shock = stock_bond_correlation * raw_shock * r['vol_mult'] + bond_randoms[month]

        stock_growth = monthly_stock_rate + r['return_boost'] + stock_shock
        bond_growth = monthly_bond_rate + bond_shock

        growth_factor = max(0, 1 + stock_growth * stock_allocation + bond_growth * bond_allocation)

        roth *= growth_factor
        traditional *= growth_factor
        brokerage *= growth_factor
        cash *= (1 + monthly_cash_rate)

        total = roth + traditional + brokerage + cash

        if month < retirement_month:
            roth += monthly_roth_contrib
            traditional += monthly_traditional_contrib
            brokerage += monthly_brokerage_contrib
            cash += monthly_cash_contrib
        else:
            total = roth + traditional + brokerage + cash
            if month == retirement_month:
                peak_portfolio = total
                retirement_portfolio = total

            # ---- Age-based spending adjustment ----
            decline_factor = 1.0
            if age <= spending_decline_end_age:
                years_over_start = max(0, age - spending_decline_start_age)
                decline_factor = annual_spending_decline_rate ** years_over_start
            else:
                years_over_start = spending_decline_end_age - spending_decline_start_age
                decline_at_end = annual_spending_decline_rate ** years_over_start
                years_over_end = age - spending_decline_end_age
                healthcare_ramp = 1 + annual_healthcare_increase_rate * years_over_end
                decline_factor = decline_at_end * healthcare_ramp

            # ---- Guardrails ----
            if (month - retirement_month) % 12 == 0:
                if total < guardrail_cut_threshold * peak_portfolio:
                    spending_multiplier = max(spending_multiplier * guardrail_cut_amount, guardrail_cut_floor)
                elif total > guardrail_raise_threshold * peak_portfolio:
                    spending_multiplier = min(spending_multiplier * guardrail_raise_amount, guardrail_raise_ceiling)

            peak_portfolio = max(peak_portfolio, total)

            adjusted_monthly_expense = base_monthly_expense * decline_factor * spending_multiplier

            income = 0
            if month >= ss_start_month:
                income += monthly_ss

            withdrawal_needed = max(adjusted_monthly_expense - income, 0)

            if month == retirement_month:
                initial_withdrawal_rate = (withdrawal_needed * 12) / retirement_portfolio

            remaining_needed = withdrawal_needed

            if remaining_needed > 0 and cash > 0:
                withdraw = min(cash, remaining_needed)
                cash -= withdraw
                remaining_needed -= withdraw

            if remaining_needed > 0 and brokerage > 0:
                withdraw = min(brokerage, remaining_needed)
                brokerage -= withdraw
                remaining_needed -= withdraw

            if remaining_needed > 0 and traditional > 0:
                gross_needed = remaining_needed / (1 - tax_rate)
                withdraw = min(traditional, gross_needed)
                traditional -= withdraw
                remaining_needed -= withdraw * (1 - tax_rate)

            if remaining_needed > 0 and roth > 0:
                withdraw = min(roth, remaining_needed)
                roth -= withdraw
                remaining_needed -= withdraw

            if remaining_needed > 0:
                return False, min_portfolio, initial_withdrawal_rate

        total = roth + traditional + brokerage + cash
        min_portfolio = min(min_portfolio, total)

        if total <= 0:
            return False, min_portfolio, initial_withdrawal_rate

    return True, min_portfolio, initial_withdrawal_rate


# ==============================
# MONTE CARLO
# ==============================

def retirement_probability(retirement_age, runs, **params):
    successes = 0
    min_values = []
    withdrawal_rates = []

    for _ in range(runs):
        survived, min_portfolio, withdrawal_rate = simulate_life(
            retirement_age=retirement_age,
            **params
        )
        if survived:
            successes += 1
        min_values.append(min_portfolio)
        withdrawal_rates.append(withdrawal_rate)

    probability = successes / runs
    avg_min = np.mean(min_values)
    avg_withdrawal_rate = np.mean([w for w in withdrawal_rates if w is not None])

    return probability, avg_min, avg_withdrawal_rate


# ==============================
# FIND EARLIEST RETIREMENT AGE
# ==============================

def find_retirement_age(
    min_age,
    max_age,
    target_probability,
    runs,
    **params
) -> Optional[RetirementResult]:

    for age in range(min_age, max_age + 1):
        prob, avg_min, avg_wr = retirement_probability(age, runs, **params)
        print(f"Testing age {age}: {prob:.2%} success | Avg Min ${avg_min:,.0f} | Avg WR {avg_wr:.2%}")

        if prob >= target_probability:
            return RetirementResult(age, prob)

    return None


# ==============================
# PROBABILITY CURVE PLOT
# ==============================

def plot_probability_curve(min_age, max_age, runs, **params):
    ages = []
    probs = []

    for age in range(min_age, max_age + 1):
        prob, _, _ = retirement_probability(age, runs, **params)
        ages.append(age)
        probs.append(prob)

    plt.figure()
    plt.plot(ages, probs)
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Retirement Age")
    plt.ylabel("Success Probability")
    plt.title("Retirement Probability Curve")
    plt.show()


def load_params(config_path='simulation_params.yaml'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ss_claim_age = cfg['life_events']['ss_claim_age']
    ss_benefit = cfg['life_events']['ss_annual_full_retirement_benefit']
    sim = cfg['simulation']
    spending = cfg['spending']
    market = cfg['market']
    life = cfg['life_events']
    normal = sim['normal_regime']
    crisis = sim['crisis_regime']

    # Build REGIMES from config
    global REGIMES
    REGIMES = {
        'normal': {
            'return_boost': normal['return_boost'],
            'vol_mult': normal['volatility_multiplier'],
            'p_switch': normal['monthly_crisis_probability'],
        },
        'crisis': {
            'return_boost': crisis['annual_return_drag'] / 12,
            'vol_mult': crisis['volatility_multiplier'],
            'p_switch': crisis['monthly_recovery_probability'],
        }
    }

    params = dict(
        # Personal
        current_age=sim['current_age'],
        roth=cfg['accounts']['roth'],
        traditional=cfg['accounts']['traditional'],
        brokerage=cfg['accounts']['brokerage'],
        cash=cfg['accounts']['cash'],
        annual_roth_contribution=cfg['contributions']['annual_roth'],
        annual_traditional_contribution=cfg['contributions']['annual_traditional'],
        annual_brokerage_contribution=cfg['contributions']['annual_brokerage'],
        annual_cash_contribution=cfg['contributions']['annual_cash'],

        # Spending
        initial_annual_expenses=spending['initial_annual_expenses'],
        spending_decline_start_age=spending['spending_decline_start_age'],
        annual_spending_decline_rate=spending['annual_spending_decline_rate'],
        spending_decline_end_age=spending['spending_decline_end_age'],
        annual_healthcare_increase_rate=spending['annual_healthcare_increase_rate'],
        guardrail_cut_threshold=spending['guardrail_cut_threshold'],
        guardrail_cut_amount=spending['guardrail_cut_amount'],
        guardrail_cut_floor=spending['guardrail_cut_floor'],
        guardrail_raise_threshold=spending['guardrail_raise_threshold'],
        guardrail_raise_amount=spending['guardrail_raise_amount'],
        guardrail_raise_ceiling=spending['guardrail_raise_ceiling'],

        # Market
        stock_return=market['stock_return'],
        bond_return=market['bond_return'],
        stock_volatility=market['stock_volatility'],
        bond_volatility=market['bond_volatility'],
        inflation=market['inflation'],
        cash_return=market['cash_return'],
        tax_rate=market['tax_rate'],
        stock_bond_correlation=market['stock_bond_correlation'],

        # Life events
        ss_claim_age=ss_claim_age,
        ss_income=ss_benefit * ss_benefit_factor(ss_claim_age),
        mean_death_age=life['death_age_mean'],
        death_age_std=life['death_age_std'],
        death_age_min=life['death_age_min'],
        death_age_max=life['death_age_max'],

        # Simulation
        glide_path=sim['glide_path'],
        glide_path_start_stock_pct=sim['glide_path_start_stock_pct'],
        glide_path_end_stock_pct=sim['glide_path_end_stock_pct'],
        glide_path_years=sim['glide_path_years'],
        static_stock_allocation=sim['static_stock_allocation'],
        return_distribution_degrees_of_freedom=sim['return_distribution_degrees_of_freedom'],
    )

    return params

# ==============================
# CLI ENTRY
# ==============================

if __name__ == "__main__":
    params = load_params('simulation_params.yaml')
    sim_cfg = yaml.safe_load(open('simulation_params.yaml'))['simulation']

    result = find_retirement_age(
        min_age=sim_cfg['min_retirement_age'],
        max_age=sim_cfg['max_retirement_age'],
        target_probability=sim_cfg['target_success_probability'],
        runs=sim_cfg['monte_carlo_runs'],
        **params
    )

    if result:
        print(f"\nEarliest Retirement Age: {result.retirement_age}")
        print(f"Success Probability: {result.success_probability:.2%}")
    else:
        print("\nNo retirement age met the target probability.")

    plot_probability_curve(
        min_age=sim_cfg['min_retirement_age'],
        max_age=sim_cfg['max_retirement_age'],
        runs=sim_cfg['plot_runs'],
        **params
    )

