import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from scipy.stats import t as t_dist
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# ==============================
# CONFIG CLASSES
# ==============================
@dataclass
class Accounts:
    roth: float
    traditional: float
    brokerage: float
    cash: float


@dataclass
class Contributions:
    annual_roth: float
    annual_traditional: float
    annual_brokerage: float
    annual_cash: float


@dataclass
class LifeEvents:
    death_age_mean: int
    death_age_std: float
    death_age_min: int
    death_age_max: int
    ss_claim_age: int
    ss_annual_full_retirement_benefit: float


@dataclass
class Market:
    stock_return: float
    bond_return: float
    stock_volatility: float
    bond_volatility: float
    inflation: float
    cash_return: float
    tax_rate: float
    stock_bond_correlation: float


@dataclass
class NormalRegime:
    return_boost: float
    volatility_multiplier: float
    monthly_crisis_probability: float


@dataclass
class CrisisRegime:
    annual_return_drag: float
    volatility_multiplier: float
    monthly_recovery_probability: float


@dataclass
class Simulation:
    current_age: int
    min_retirement_age: int
    max_retirement_age: int
    target_success_probability: float
    monte_carlo_runs: int

    glide_path: bool
    glide_path_start_stock_pct: float
    glide_path_end_stock_pct: float
    glide_path_years: int

    return_distribution_degrees_of_freedom: int
    static_stock_allocation: float

    normal_regime: NormalRegime
    crisis_regime: CrisisRegime


@dataclass
class Spending:
    initial_annual_expenses: float
    spending_decline_start_age: int
    annual_spending_decline_rate: float
    spending_decline_end_age: int
    annual_healthcare_increase_rate: float
    guardrail_cut_threshold: float
    guardrail_cut_amount: float
    guardrail_cut_floor: float
    guardrail_raise_threshold: float
    guardrail_raise_amount: float
    guardrail_raise_ceiling: float


@dataclass
class Config:
    accounts: Accounts
    contributions: Contributions
    life_events: LifeEvents
    market: Market
    simulation: Simulation
    spending: Spending


@dataclass
class RetirementResult:
    retirement_age: int
    success_probability: float

_worker_simulator = None
def init_worker(config):
    global _worker_simulator
    _worker_simulator = RetirementSimulator(config)
def simulate_worker(retirement_age):
    return _worker_simulator.simulate_life(retirement_age)

# ==============================
# CORE LIFE SIMULATION
# ==============================
class RetirementSimulator:
    def __init__(self, config: Config):
        self.cfg = config
        self.regimes = {
            'normal': {
                'return_boost': self.cfg.simulation.normal_regime.return_boost,
                'vol_mult': self.cfg.simulation.normal_regime.volatility_multiplier,
                'p_switch': self.cfg.simulation.normal_regime.monthly_crisis_probability
            },
            'crisis': {
                'return_boost': self.cfg.simulation.crisis_regime.annual_return_drag / 12,
                'vol_mult': self.cfg.simulation.crisis_regime.volatility_multiplier,
                'p_switch': self.cfg.simulation.crisis_regime.monthly_recovery_probability
            }
        }
        self.probability_results = {}
        self.ss_income = (
            self.cfg.life_events.ss_annual_full_retirement_benefit * self.ss_benefit_factor(self.cfg.life_events.ss_claim_age)
        )

    def real_return(self, nominal):
        return (1 + nominal) / (1 + self.cfg.market.inflation) - 1


    def monthly_rate(self, annual_rate):
        return (1 + annual_rate) ** (1/12) - 1


    def random_death_age(self):
        age = int(np.random.normal(self.cfg.life_events.death_age_mean, self.cfg.life_events.death_age_std))
        return max(min(age, self.cfg.life_events.death_age_max), self.cfg.life_events.death_age_min)


    # Approximate Social Security adjustment factors
    def ss_benefit_factor(self, claim_age, full_retirement_age=67):
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

    def simulate_life(self, retirement_age: int):
        regime = 'normal'
        roth = self.cfg.accounts.roth
        traditional = self.cfg.accounts.traditional
        brokerage = self.cfg.accounts.brokerage
        cash = self.cfg.accounts.cash

        death_age = self.random_death_age()
        living_months = (death_age - self.cfg.simulation.current_age) * 12
        retirement_month = (retirement_age - self.cfg.simulation.current_age) * 12

        real_stock_return = self.real_return(self.cfg.market.stock_return)
        real_bond_return = self.real_return(self.cfg.market.bond_return)
        real_cash_return = self.real_return(self.cfg.market.cash_return)

        monthly_stock_rate = self.monthly_rate(real_stock_return)
        monthly_bond_rate = self.monthly_rate(real_bond_return)
        monthly_cash_rate = self.monthly_rate(real_cash_return)

        base_monthly_expense = self.cfg.spending.initial_annual_expenses / 12
        spending_multiplier = 1.0
        peak_portfolio = self.cfg.accounts.roth + self.cfg.accounts.traditional + self.cfg.accounts.brokerage + self.cfg.accounts.cash
        monthly_roth_contrib = self.cfg.contributions.annual_roth / 12
        monthly_traditional_contrib = self.cfg.contributions.annual_traditional / 12
        monthly_brokerage_contrib = self.cfg.contributions.annual_brokerage / 12
        monthly_cash_contrib = self.cfg.contributions.annual_cash / 12
        monthly_ss = self.ss_income / 12

        min_portfolio = float("inf")

        # Pre-generate all random numbers for this life simulation
        df = self.cfg.simulation.return_distribution_degrees_of_freedom
        t_scale = np.sqrt((df - 2) / df)
        stock_shocks_raw = t_dist.rvs(df=df, size=living_months) * t_scale * (self.cfg.market.stock_volatility / np.sqrt(12))
        bond_randoms = np.random.normal(0, self.cfg.market.bond_volatility / np.sqrt(12), size=living_months)
        regime_rolls = np.random.random(size=living_months)
        ss_start_month = (self.cfg.life_events.ss_claim_age - self.cfg.simulation.current_age) * 12

        # Derive glide path slope from config
        glide_years = max(self.cfg.simulation.glide_path_years, 1)
        glide_path_slope = (self.cfg.simulation.glide_path_start_stock_pct - self.cfg.simulation.glide_path_end_stock_pct) / glide_years

        initial_withdrawal_rate = None

        for month in range(living_months):
            age = self.cfg.simulation.current_age + month / 12

            if self.cfg.simulation.glide_path:
                years_since_retirement = max(0, age - retirement_age)
                stock_allocation = max(
                    self.cfg.simulation.glide_path_end_stock_pct,
                    self.cfg.simulation.glide_path_start_stock_pct - glide_path_slope * years_since_retirement
                )
            else:
                stock_allocation = self.cfg.simulation.static_stock_allocation

            bond_allocation = 1 - stock_allocation

            regime_params = self.regimes[regime]
            if regime_rolls[month] < regime_params['p_switch']:
                regime = 'crisis' if regime == 'normal' else 'normal'
            regime_params = self.regimes[regime]

            raw_shock = stock_shocks_raw[month]
            stock_shock = raw_shock * regime_params['vol_mult']
            bond_shock = self.cfg.market.stock_bond_correlation * raw_shock * regime_params['vol_mult'] + bond_randoms[month]

            stock_growth = monthly_stock_rate + regime_params['return_boost'] + stock_shock
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
                if age <= self.cfg.spending.spending_decline_end_age:
                    years_over_start = max(0, age - self.cfg.spending.spending_decline_start_age)
                    decline_factor = self.cfg.spending.annual_spending_decline_rate ** years_over_start
                else:
                    years_over_start = self.cfg.spending.spending_decline_end_age - self.cfg.spending.spending_decline_start_age
                    decline_at_end = self.cfg.spending.annual_spending_decline_rate ** years_over_start
                    years_over_end = age - self.cfg.spending.spending_decline_end_age
                    healthcare_ramp = 1 + self.cfg.spending.annual_healthcare_increase_rate * years_over_end
                    decline_factor = decline_at_end * healthcare_ramp

                # ---- Guardrails ----
                if (month - retirement_month) % 12 == 0:
                    if total < self.cfg.spending.guardrail_cut_threshold * peak_portfolio:
                        spending_multiplier = max(spending_multiplier * self.cfg.spending.guardrail_cut_amount, self.cfg.spending.guardrail_cut_floor)
                    elif total > self.cfg.spending.guardrail_raise_threshold * peak_portfolio:
                        spending_multiplier = min(spending_multiplier * self.cfg.spending.guardrail_raise_amount, self.cfg.spending.guardrail_raise_ceiling)

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
                    gross_needed = remaining_needed / (1 - self.cfg.market.tax_rate)
                    withdraw = min(traditional, gross_needed)
                    traditional -= withdraw
                    remaining_needed -= withdraw * (1 - self.cfg.market.tax_rate)

                if remaining_needed > 0 and roth > 0:
                    withdraw = min(roth, remaining_needed)
                    roth -= withdraw
                    remaining_needed -= withdraw

                if remaining_needed > 0:
                    return False, min_portfolio, initial_withdrawal_rate, total

            total = roth + traditional + brokerage + cash
            min_portfolio = min(min_portfolio, total)

            if total <= 0:
                return False, min_portfolio, initial_withdrawal_rate, total

        return True, min_portfolio, initial_withdrawal_rate, total


    # ==============================
    # MONTE CARLO
    # ==============================
    def compute_probability_curve(self):
        for age in range(self.cfg.simulation.min_retirement_age,
                        self.cfg.simulation.max_retirement_age + 1):

            self.probability_results[age] = self.retirement_probability(age)


    def retirement_probability(self, retirement_age):
        runs = self.cfg.simulation.monte_carlo_runs

        successes = 0
        min_sum = 0
        final_sum = 0
        wr_sum = 0
        wr_count = 0

        with Pool(
            processes=cpu_count(),
            initializer=init_worker,
            initargs=(self.cfg,)
        ) as pool:

            results = list(
                tqdm(
                    pool.imap(simulate_worker, [retirement_age] * runs),
                    total=runs,
                    desc=f"Age {retirement_age}"
                )
            )

        for survived, min_portfolio, withdrawal_rate, final_portfolio in results:

            if survived:
                successes += 1

            min_sum += min_portfolio
            final_sum += final_portfolio

            if withdrawal_rate is not None:
                wr_sum += withdrawal_rate
                wr_count += 1

        probability = successes / runs
        avg_min = min_sum / runs
        avg_withdrawal_rate = wr_sum / wr_count if wr_count else 0
        avg_final = final_sum / runs

        return probability, avg_min, avg_withdrawal_rate, avg_final


    # ==============================
    # FIND EARLIEST RETIREMENT AGE
    # ==============================
    def find_retirement_age(self) -> Optional[RetirementResult]:
        for age in sorted(self.probability_results):
            prob, avg_min, avg_wr, final_portfolio = self.probability_results[age]
            print(f"Testing age {age}: {prob:.2%} success | Avg Min ${avg_min:,.0f} | Avg WR {avg_wr:.2%} | Final ${final_portfolio:,.0f}")

            if prob >= self.cfg.simulation.target_success_probability:
                return RetirementResult(age, prob)

        return None


    # ==============================
    # PROBABILITY CURVE PLOT
    # ==============================
    def plot_probability_curve(self):
        ages = sorted(self.probability_results)
        probs = [self.probability_results[a][0] for a in ages]

        plt.figure()
        plt.plot(ages, probs)
        plt.axhline(self.cfg.simulation.target_success_probability, linestyle="--")
        plt.xlabel("Retirement Age")
        plt.ylabel("Success Probability")
        plt.title("Retirement Probability Curve")
        plt.show()


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    sim = raw["simulation"]
    return Config(
        accounts=Accounts(**raw["accounts"]),
        contributions=Contributions(**raw["contributions"]),
        life_events=LifeEvents(**raw["life_events"]),
        market=Market(**raw["market"]),
        simulation=Simulation(
            current_age=sim["current_age"],
            min_retirement_age=sim["min_retirement_age"],
            max_retirement_age=sim["max_retirement_age"],
            target_success_probability=sim["target_success_probability"],
            monte_carlo_runs=sim["monte_carlo_runs"],
            glide_path=sim["glide_path"],
            glide_path_start_stock_pct=sim["glide_path_start_stock_pct"],
            glide_path_end_stock_pct=sim["glide_path_end_stock_pct"],
            glide_path_years=sim["glide_path_years"],
            return_distribution_degrees_of_freedom=sim["return_distribution_degrees_of_freedom"],
            static_stock_allocation=sim["static_stock_allocation"],
            normal_regime=NormalRegime(**sim["normal_regime"]),
            crisis_regime=CrisisRegime(**sim["crisis_regime"])
        ),
        spending=Spending(**raw["spending"]),
    )

# ==============================
# CLI ENTRY
# ==============================

if __name__ == "__main__":
    config = load_config('simulation_params_dustan.yaml')
    sim = RetirementSimulator(config)
    sim.compute_probability_curve()
    retirement_age = sim.find_retirement_age()
    print("Earliest retirement age:", retirement_age)
    sim.plot_probability_curve()
