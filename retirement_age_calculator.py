"""Monte Carlo retirement age simulator.

EVERYTHING IN THIS MODEL IS IN REAL (TODAY'S) DOLLARS.
You enter nominal returns in the YAML and they are converted to real returns once,
up front, by dividing out inflation. Balances and expenses are never re-inflated
afterwards. So a portfolio that reads $2,000,000 at age 70 means "$2,000,000 of
today's purchasing power", not "$2,000,000 of year-2060 dollars".

The single most important consequence: any growth rate you enter for a quantity that
is already in today's dollars (contributions, spending) is growth ABOVE inflation.
To hold contributions constant in purchasing power, set the growth rate to 0.0.

The one deliberate exception is the brokerage cost basis, which is a frozen nominal
figure the IRS never indexes. Its real value is decayed by realised inflation each
month, so inflation alone produces a taxable capital gain -- as it does in life.

What one simulated life contains, in order:
  - a random lifespan per person (SSA-calibrated Gompertz mortality, or a clipped
    normal), so longevity risk is part of every answer
  - monthly market paths: fat-tailed (Student-t) stock shocks, two-state calm/crisis
    regime switching, correlated bonds, stochastic inflation eroding bonds and cash
  - accumulation with monthly contributions until retirement
  - retirement, planned one tax year at a time: the spending target (smile, guardrails,
    survivor adjustment, healthcare premiums) minus Social Security gives a net need,
    and a bisection solver finds the exact gross withdrawal through the
    cash -> brokerage -> traditional -> Roth ladder whose after-tax proceeds cover it
  - taxes computed from the real 2026 rules: progressive federal brackets by filing
    status, LTCG stacking with the 0% band, Social Security provisional-income
    taxation, NIIT, the 10% early-distribution penalty (with 72(t) and rule-of-55
    escapes), annual tax drag on taxable-account dividends and interest, and RMDs
    from the configured start age using the IRS Uniform Lifetime Table
  - Social Security from the progressive PIA bend-point formula, the 40-credit
    eligibility rule, exact claim-age factors, spousal benefits, and survivor
    benefits on their own schedule (delayed credits only for months actually lived,
    and the 82.5%-of-PIA widow's limit)

Reproducibility: every simulated life is driven by its own seeded numpy Generator.
With simulation.common_random_numbers on, every candidate retirement age is tested
against the SAME set of lifetimes, so the difference between two ages (or between two
configs run with the same seed) reflects the decision being tested, not Monte Carlo
noise. That is the property that makes "how does X change my retirement age?"
comparisons trustworthy.

Sources for the hard numbers baked in here:
  - Federal/LTCG brackets, standard deductions (2026): IRS Rev. Proc. 2025-32
  - NIIT: IRC section 1411 (3.8% over $200k single / $250k MFJ, not indexed)
  - SS provisional-income thresholds: IRC section 86 (fixed since 1983/1993)
  - SS PIA bend points and factors (2026): ssa.gov/oact/cola/bendpoints.html
  - SS claim-age factors: ssa.gov/oact/quickcalc/early_late.html
  - RMD divisors: IRS Pub. 590-B Uniform Lifetime Table; start age per SECURE 2.0
  - Mortality calibration targets: SSA cohort life tables (2025 Trustees Report)

This is a planning model, not tax, legal, or Social Security advice. The
`assumption_report` prints what the model is actually using -- read it.
"""

import math
import os
from dataclasses import dataclass, asdict, fields, is_dataclass
from multiprocessing import Pool, cpu_count
from typing import List, Optional

import numpy as np
import yaml
from tqdm import tqdm

# matplotlib.pyplot is deliberately NOT imported here. Every worker process
# re-imports this module (macOS/Windows spawn), and pyplot costs ~170ms and a
# font-cache check per worker -- for a pool of ten, on every one of the swept
# ages, to draw nothing at all. In a PyInstaller build the workers could not
# persist that cache and rebuilt it on every launch, which turned a two-age smoke
# run into 36 seconds. plot_results imports it at call time instead; the drawing
# methods take an Axes and never need it.

# Runs behind the portfolio trajectory chart. The low percentile lines need a large
# sample to be stable run-to-run, and later ages are backed by only the subset of
# runs still alive at that age.
TRAJECTORY_SAMPLES = 5000

# Fewest still-alive runs a trajectory column needs before it is worth drawing.
# The fan chart's lowest line is the 1st percentile, and numpy has no 1st
# percentile to compute once n < 100 -- it interpolates one from the minimum. Past
# the mid-90s the surviving cohort collapses (for a 35-year-old male: ~400 runs
# alive at 95, ~20 at 104, ~2 at 107), and with a handful of samples every
# percentile lands on the same one or two rich survivors. The lines then converge
# and shoot upward, which reads as a real late-life spike but is pure
# small-sample noise. compute_trajectory_percentiles still RETURNS those columns
# (with their honest counts in percentiles["n"]); draw_trajectory just declines to
# plot them.
MIN_TRAJECTORY_SAMPLES = 100


# ==============================
# CONFIG CLASSES
# ==============================
@dataclass
class Accounts:
    """Current balances, in today's dollars.

    brokerage_cost_basis is what you actually paid for the brokerage holdings. Only
    the gain above basis is taxable when you sell. Example: brokerage=$100k with
    basis=$60k means $40k (40%) of any withdrawal is a taxable capital gain.
    The basis is tracked through the whole simulation: it falls pro-rata when you
    sell, rises when dividends/interest are taxed and reinvested and when new
    contributions are made, and erodes with inflation -- exactly as in a real
    account. That last part is the one place this model departs from "everything is
    real": the basis is a frozen historical dollar figure that the IRS never
    indexes, so its purchasing power decays and inflation alone turns into taxable
    gain. Enter it in TODAY'S dollars like everything else; the decay starts today.
    """
    roth: float
    traditional: float
    brokerage: float
    cash: float
    brokerage_cost_basis: float


@dataclass
class Contributions:
    """Annual savings, in today's dollars.

    annual_contribution_growth_rate is REAL growth, i.e. growth above inflation,
    because this model is denominated in today's dollars. Set it to 0.0 to keep
    contributions flat in purchasing power (a raise that merely matches inflation).
    Setting it to 0.03 means you expect a 3% raise EVERY YEAR ON TOP of inflation --
    over 20 years that compounds to 1.81x your starting contribution in real terms.
    """
    annual_roth: float
    annual_traditional: float
    annual_brokerage: float
    annual_cash: float
    annual_contribution_growth_rate: float


@dataclass
class LifeEvents:
    """Mortality and Social Security for the primary person.

    Mortality: mortality_model chooses how the death age is drawn each run.
      - "ssa_inspired": a Gompertz hazard curve calibrated to SSA cohort life tables
        (see RetirementSimulator._actuarial_death_age). death_age_max caps the draw
        (and sizes the trajectory chart); death_age_mean/std/min are ignored.
      - "normal": a normal draw clipped to [death_age_min, death_age_max]. Simple and
        fully user-controlled; useful for "what if I plan to exactly 95" experiments.

    Social Security: ss_annual_full_retirement_benefit is the benefit you would
    receive at your full retirement age (67) IF you worked a full
    `ss_benefit_years_required` (35) year career. Check ssa.gov/myaccount for your
    real number. The simulator converts it to a Primary Insurance Amount, shrinks the
    underlying 35-year earnings average for the zero years an early retirement leaves
    behind, and re-applies the progressive bend-point formula -- so a short career
    costs less than a linear proration would suggest, matching how SSA actually
    computes benefits.

    ss_credits_at_current_age: SSA work credits you already have (4 per year worked,
    40 needed for any retirement benefit at all). Someone retiring at 30 with 8 years
    of work never reaches 40 credits and receives NOTHING -- this field is what lets
    the model catch that.
    """
    death_age_mean: int
    death_age_std: float
    death_age_min: int
    death_age_max: int
    mortality_model: str
    mortality_sex: str
    ss_claim_age: int
    ss_annual_full_retirement_benefit: float
    ss_earnings_years_at_current_age: int
    ss_credits_at_current_age: int
    ss_benefit_years_required: int
    ss_retirement_eligibility_credits: int


@dataclass
class Spouse:
    """Optional second person. When enabled=False every field here is ignored.

    age_offset is the spouse's age relative to yours: +2 means they are two years
    older, -3 means three years younger.

    Mortality fields mirror LifeEvents (each spouse gets their own model and sex).

    survivor_spending_factor is how household spending changes after the first death.
    0.75 means the survivor spends 75% of what the couple spent -- housing and
    utilities do not halve, so a factor well above 0.5 is normal.

    A spouse with little or no earnings record still collects the SPOUSAL benefit
    (up to 50% of the primary's PIA) -- see RetirementSimulator.spouse_ss_income.
    """
    enabled: bool
    age_offset: int
    death_age_mean: int
    death_age_std: float
    death_age_min: int
    death_age_max: int
    mortality_model: str
    mortality_sex: str
    ss_claim_age: int
    ss_annual_full_retirement_benefit: float
    ss_earnings_years_at_current_age: int
    ss_credits_at_current_age: int
    survivor_spending_factor: float


@dataclass
class Market:
    """Return assumptions. Enter NOMINAL returns; they are converted to real internally.

    inflation is the MEAN inflation rate; inflation_volatility is its annual standard
    deviation. Inflation uncertainty matters because bonds and cash pay a fixed
    nominal amount: if inflation comes in 2% above expectation for a year, the real
    return on those assets is 2% worse than planned. Stocks are modelled as
    inflation-neutral in the long run, so their real return is taken as given.

    stock_dividend_yield / bond_taxable_yield model the ANNUAL TAX DRAG on the
    taxable brokerage account (and cash interest is taxed via cash_return). Total
    return is unchanged -- dividends and interest are already inside stock_return and
    bond_return -- but the distributed portion is taxed every year rather than
    deferred until sale, which is exactly what makes a taxable account less efficient
    than an IRA. Dividends are taxed at qualified-dividend (LTCG) rates, bond and
    cash interest at ordinary rates, and reinvested distributions raise the cost
    basis. During accumulation this tax is assumed paid from salary, not the account
    (and pre-retirement distributions do not step up basis -- both mildly
    conservative simplifications).

    stock_bond_correlation is the correlation between stock and bond shocks.
    -0.3 means bonds tend to rise modestly when stocks fall (a partial hedge).
    Note this is applied to STANDARDISED shocks, so the realised bond volatility is
    exactly bond_volatility and the realised correlation is exactly this number,
    in both the normal and crisis regimes.
    """
    stock_return: float
    bond_return: float
    stock_volatility: float
    bond_volatility: float
    inflation: float
    inflation_volatility: float
    cash_return: float
    stock_dividend_yield: float
    bond_taxable_yield: float
    stock_bond_correlation: float


@dataclass
class Taxes:
    """Progressive tax model, all thresholds in today's dollars.

    Nearly every threshold here is indexed to inflation by the IRS each year, so in a
    real-dollar model they correctly stay constant. (The exceptions -- the SS
    provisional-income thresholds and NIIT thresholds are fixed in NOMINAL dollars by
    statute -- are held real-constant anyway, which assumes Congress eventually
    indexes them; see README limitations.)

    Every bracket table is a mapping from FILING STATUS to that status's numbers,
    because the status changes within a simulation: when the first spouse dies, the
    survivor files as single from the next tax year, with half the standard deduction
    and compressed brackets -- the "widow's tax penalty" that single-status models
    silently ignore.

    federal_brackets / ltcg_brackets values are [upper_bound, rate] pairs, lowest
    first, with a null upper bound on the final (top) bracket, in TAXABLE income
    (after the standard deduction). Example entry [50400, 0.12] means "taxable income
    up to $50,400 in this bracket is taxed at 12%".

    early_withdrawal_penalty is the extra 10% the IRS charges on traditional
    401k/IRA withdrawals before penalty_free_age (59.5).
      - use_72t_sepp: set true if you plan a Rule 72(t) substantially-equal-payments
        schedule or a Roth conversion ladder, either of which legally avoids the
        penalty (the model waives the penalty but still taxes withdrawals as
        ordinary income; it does not enforce a SEPP schedule for you).
      - assume_qualified_plan_age55_exception: employer-plan (401k/403b) money can be
        withdrawn penalty-free after separating from service in or after the year
        you turn 55. Only enable if the traditional bucket is genuinely employer-plan
        money you would leave in the plan; IRAs never qualify.

    niit_rate / niit_thresholds: the 3.8% Net Investment Income Tax (IRC 1411) on
    investment income to the extent AGI exceeds the threshold.

    additional_standard_deductions_65plus is the extra deduction IRC 63(f) gives
    every filer who has reached 65, PER QUALIFYING PERSON: 2026 is $2,050 for an
    unmarried filer and $1,650 for each spouse on a joint return (so $3,300 when
    both are 65+). It is permanent law and it applies in nearly every year a
    retirement model simulates, so leaving it out overstates tax by roughly
    $350-600 a year for a typical retiree. (The separate OBBBA "senior deduction"
    is deliberately NOT modelled: it expires after 2028, so a model held constant
    in real terms would wrongly grant it for life.)
    """
    filing_status: str
    standard_deductions: dict
    additional_standard_deductions_65plus: dict
    federal_brackets: dict
    ltcg_brackets: dict
    state_tax_rate: float
    early_withdrawal_penalty: float
    penalty_free_age: float
    use_72t_sepp: bool
    assume_qualified_plan_age55_exception: bool
    ss_provisional_thresholds: dict
    ss_max_taxable_fraction: float
    niit_rate: float
    niit_thresholds: dict


@dataclass
class Healthcare:
    """Health insurance costs during retirement, per person, in today's dollars.

    These are ADDED ON TOP of spending.initial_annual_expenses, so your expenses
    figure should be your spending EXCLUDING health premiums.

    Before Medicare eligibility you buy your own coverage (ACA marketplace), which is
    far more expensive than an employer plan -- this is the single biggest cost most
    early-retirement plans forget. Set pre_medicare_annual_premium to 0 if your
    expenses figure already includes it. After spending_decline_end_age these
    premiums grow at spending.annual_healthcare_increase_rate (real medical trend
    above CPI).
    """
    pre_medicare_annual_premium: float
    medicare_annual_premium: float
    medicare_age: int


@dataclass
class NormalRegime:
    """The calm market state.

    return_boost is an ANNUAL rate, e.g. 0.01 = one extra percentage point per year.
    (It is converted to a monthly log rate internally, exactly like the crisis drag.)
    """
    return_boost: float
    volatility_multiplier: float
    monthly_crisis_probability: float


@dataclass
class CrisisRegime:
    """The bear-market state.

    annual_return_drag is an ANNUAL rate, e.g. -0.12 = twelve points per year worse
    than trend while the crisis lasts.

    monthly_recovery_probability sets how long crises last: 0.055 gives an average
    spell of 1/0.055 = 18 months, matching the historical average bear market.
    """
    annual_return_drag: float
    volatility_multiplier: float
    monthly_recovery_probability: float


@dataclass
class Simulation:
    """Sweep settings, reproducibility, and the market regime model.

    random_seed: seed for the whole run. Any value >= 0 makes results exactly
    reproducible -- run the same config twice, get the same numbers. Set -1 to use
    fresh entropy each run (results then vary by the Monte Carlo error bar,
    roughly +/-0.6% on a 95% success rate at 5,000 runs).

    common_random_numbers: when true, every candidate retirement age is simulated
    against the SAME set of random lifetimes (same deaths, same market paths). This
    is the classic variance-reduction technique for comparing alternatives: the
    DIFFERENCE between two ages is then driven by the decision, not by each age
    getting its own luck, so the success curve is far less noisy exactly where you
    read it -- at the crossing of your target. Leave it on.

    rmd_start_age: SECURE 2.0 sets Required Minimum Distributions at age 73 for
    those born 1951-1959 and 75 for those born 1960 or later. Set yours accordingly.

    compensate_crisis_drag: because crises are a recurring state rather than a
    one-off, they permanently drag down the average return. With a crisis 21% of the
    time at -12%/yr, the long-run return is about 1.2 points below the stock_return
    you entered. When this is true, the simulator raises the normal-regime return
    just enough that the LONG-RUN AVERAGE equals the stock_return you configured,
    so 8% means 8%. Crises still hurt -- they are just offset by better calm periods,
    which is how the historical average already works.
    """
    current_age: int
    min_retirement_age: int
    max_retirement_age: int
    target_success_probability: float
    monte_carlo_runs: int
    random_seed: int
    common_random_numbers: bool

    glide_path: bool
    glide_path_start_stock_pct: float
    glide_path_end_stock_pct: float
    glide_path_years: int

    return_distribution_degrees_of_freedom: int
    static_stock_allocation: float
    compensate_crisis_drag: bool
    rmd_start_age: int

    normal_regime: NormalRegime
    crisis_regime: CrisisRegime


@dataclass
class Spending:
    """Spending path through retirement, in today's dollars.

    The "retirement spending smile" (Blanchett 2014): real spending drifts down
    through your 70s as you travel and drive less -- annual_spending_decline_rate=0.99
    means 1% less each year between the start and end ages -- and the late-life
    upturn is modelled through the healthcare side: after spending_decline_end_age
    the per-person health premiums grow at annual_healthcare_increase_rate (medical
    costs outrunning CPI). Base spending stays at its declined level. Long-term-care
    shocks are NOT modelled; if that risk worries you, raise the late-life healthcare
    rate or hold a reserve outside this plan.

    Guardrails cut or raise spending in response to the portfolio's own return over
    the past year, which is how real retirees behave and is what makes higher
    withdrawal rates survivable (Guyton-Klinger style). Example:
    guardrail_cut_return_threshold=-0.10 with guardrail_cut_amount=0.90 means "if the
    portfolio returned worse than -10% over the past year, cut spending by 10%",
    never going below guardrail_cut_floor (70%) of plan.
    """
    initial_annual_expenses: float
    spending_decline_start_age: int
    annual_spending_decline_rate: float
    spending_decline_end_age: int
    annual_healthcare_increase_rate: float
    guardrail_cut_return_threshold: float
    guardrail_cut_amount: float
    guardrail_cut_floor: float
    guardrail_raise_return_threshold: float
    guardrail_raise_amount: float
    guardrail_raise_ceiling: float


@dataclass
class Config:
    """The complete parameter set: nine sections mirroring the YAML's top-level
    keys. Built by config_from_dict, checked by validate_config; both front ends
    read and write it only through get_field/set_field."""
    accounts: Accounts
    contributions: Contributions
    life_events: LifeEvents
    spouse: Spouse
    market: Market
    taxes: Taxes
    healthcare: Healthcare
    simulation: Simulation
    spending: Spending


@dataclass
class RetirementResult:
    """The headline answer: the earliest swept retirement age whose success
    probability met the target, and the probability it achieved."""
    retirement_age: int
    success_probability: float


class ValidationError(ValueError):
    """A config value is missing, out of range, or inconsistent. The message lists
    every problem found, not just the first."""


# ==============================
# REFERENCE TABLES
# ==============================
# IRS Uniform Lifetime Table (Pub. 590-B, effective 2022). Once RMDs start, the IRS
# forces you to withdraw at least prior_year_end_balance/divisor from traditional
# accounts each year, whether you need the money or not, so the deferred tax finally
# gets paid. Example: at 75 with $500,000 traditional, the forced withdrawal is
# 500000/24.6 = $20,325. The table legally runs to 120+; divisor 2.0 applies beyond.
RMD_TABLE = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
    87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
    94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
    101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
    115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}

# The only filing status covering TWO people, and so the only one that can claim
# the IRC 63(f) age-65 addition twice. A qualifying surviving spouse files at joint
# RATES but is one person, and claims it once.
TWO_PERSON_FILING_STATUSES = ("married_filing_jointly",)

# SSA PIA formula, 2026 bend points (MONTHLY dollars of AIME -- Average Indexed
# Monthly Earnings). The formula is progressive: 90% of AIME up to the first bend
# point, 32% to the second, 15% above. Because this model is real-dollar, today's
# published formula is held constant in real terms (bend points are AWI-indexed,
# which historically tracks slightly above CPI -- a mildly conservative choice).
SS_PIA_BEND1 = 1286.0
SS_PIA_BEND2 = 7749.0
SS_PIA_RATE1, SS_PIA_RATE2, SS_PIA_RATE3 = 0.90, 0.32, 0.15

# SSA's "widow(er)'s limit" (RIB-LIM, 20 CFR 404.338): when the deceased had already
# claimed EARLY, the survivor inherits the reduced benefit but never less than 82.5%
# of the deceased's PIA. It stops an early claim from permanently halving a
# survivor's income.
SS_WIDOW_LIMIT_FRACTION = 0.825

# Gompertz adult-mortality parameters per sex: (b, q75) where the annual death
# probability is q(x) = a * exp(b*x) with a chosen so q(75) = q75. Calibrated so the
# implied survival CONDITIONAL ON REACHING 65 approximates SSA cohort life tables
# (2025 Trustees Report, intermediate assumptions) for someone in their 30s today:
#
#            survive 65->80   ->90    ->95    ->100     median death (from 65)
#   male          ~70%        ~33%    ~15%     ~5%            ~86
#   female        ~78%        ~44%    ~24%     ~8%            ~89
#
# A Gompertz curve understates accident-driven mortality below ~55, which errs
# conservative here: fewer early deaths means more retirements that must be funded.
GOMPERTZ_PARAMS = {
    "male": (0.095, 0.028),
    "female": (0.105, 0.019),
}


# ==============================
# TAX ENGINE
# ==============================
class TaxCalculator:
    """Federal + state tax on one year of retirement income, by filing status.

    Four income types are handled separately because they are taxed differently:
      - ordinary:    traditional 401k/IRA withdrawals, RMDs, bond and cash interest.
                     Taxed at progressive bracket rates.
      - gains:       realised long-term capital gains plus qualified dividends.
                     Taxed at the lower LTCG rates, and stacked ON TOP of ordinary
                     income when deciding which LTCG rate applies.
      - ss:          Social Security. Between 0% and 85% of it is taxable depending
                     on other income (see taxable_social_security).
      - investment:  the NIIT base -- dividends, interest and realised gains -- taxed
                     an extra 3.8% to the extent AGI exceeds the NIIT threshold.
    Roth withdrawals and returning your own brokerage cost basis are not income at
    all and never appear here.
    """

    def __init__(self, cfg: Taxes):
        """Bind the Taxes config; all thresholds are read per call by filing status."""
        self.cfg = cfg

    # ---------- per-filing-status lookups ----------
    # Each table in the config maps filing status -> that status's numbers; these
    # thin accessors exist so every consumer resolves the status the same way.
    def standard_deduction(self, status: str) -> float:
        """BASE standard deduction for `status` (2026: 16,100 single / 32,200 MFJ).

        This is the under-65 figure. Use total_standard_deduction for the amount a
        return actually claims -- retirees are overwhelmingly 65+.
        """
        return float(self.cfg.standard_deductions[status])

    def total_standard_deduction(self, status: str, age: float,
                                 spouse_age: Optional[float] = None) -> float:
        """Standard deduction actually claimed, including the IRC 63(f) age addition.

        63(f) adds a fixed amount for each filer who has attained 65 by the close of
        the tax year: 2026 is $2,050 for an unmarried filer, $1,650 for EACH spouse
        on a joint return. So a couple both past 65 deducts 32,200 + 2 x 1,650 =
        35,500, and a survivor filing single deducts 16,100 + 2,050 = 18,150.

        `age` is the filer's age and `spouse_age` the other spouse's, passed only
        when the return is joint and both are alive; a second person is counted only
        on a joint return. Ages are taken at the plan anniversary, so someone turning
        65 mid-year first claims it the following plan year -- marginally
        conservative and one line simpler than tracking birthdays.
        """
        base = self.standard_deduction(status)
        per_person = float(self.cfg.additional_standard_deductions_65plus[status])
        qualifying = 1 if age >= 65 else 0
        if status in TWO_PERSON_FILING_STATUSES and spouse_age is not None and spouse_age >= 65:
            qualifying += 1
        return base + per_person * qualifying

    def federal_brackets(self, status: str):
        """Ordinary-income brackets for `status`: [upper_bound, rate] pairs."""
        return self.cfg.federal_brackets[status]

    def ltcg_brackets(self, status: str):
        """Long-term capital gains brackets for `status` (0% / 15% / 20% bands)."""
        return self.cfg.ltcg_brackets[status]

    def ss_thresholds(self, status: str):
        """The two provisional-income thresholds for SS taxability (IRC 86)."""
        return self.cfg.ss_provisional_thresholds[status]

    def niit_threshold(self, status: str) -> float:
        """AGI threshold above which the 3.8% NIIT applies (IRC 1411)."""
        return float(self.cfg.niit_thresholds[status])

    @staticmethod
    def _bracket_tax(taxable: float, brackets) -> float:
        """Walk progressive brackets, taxing each slice at its own rate.

        Example with taxable=$60,000 and 2026 single brackets
        [[12400,0.10],[50400,0.12],[null,0.22]]:
          first $12,400          at 10% = $1,240.00
          next  $38,000          at 12% = $4,560.00
          final  $9,600          at 22% = $2,112.00
          total                         = $7,912.00  (an effective rate of 13.2%)
        """
        if taxable <= 0:
            return 0.0
        tax = 0.0
        lower = 0.0
        for upper, rate in brackets:
            cap = taxable if upper is None else min(taxable, upper)
            if cap > lower:
                tax += (cap - lower) * rate
            lower = cap
            if upper is not None and taxable <= upper:
                break
        return tax

    def taxable_social_security(self, ss: float, other_income: float, status: str) -> float:
        """How much of a Social Security benefit is subject to income tax.

        The IRS uses "provisional income" = other income + half your benefit, then
        phases the taxable portion in across two thresholds (IRC 86). Below the
        first, none of it is taxed; between them, up to 50%; above the second, up
        to 85%.

        Example (single, thresholds 25k/34k): $40,000 of SS and $20,000 of other
        income gives provisional income of $20,000 + $20,000 = $40,000, above the
        upper threshold, so a large share of the benefit becomes taxable.
        """
        if ss <= 0:
            return 0.0
        t1, t2 = self.ss_thresholds(status)
        provisional = other_income + 0.5 * ss
        if provisional <= t1:
            return 0.0
        if provisional <= t2:
            return min(0.5 * (provisional - t1), 0.5 * ss)
        tier1 = min(0.5 * (t2 - t1), 0.5 * ss)
        return min(0.85 * (provisional - t2) + tier1,
                   self.cfg.ss_max_taxable_fraction * ss)

    def tax_breakdown(self, ordinary: float, gains: float, ss: float,
                      investment_income: float, status: str, age: float,
                      traditional_withdrawal: float = 0.0,
                      spouse_age: Optional[float] = None) -> dict:
        """Every component of one year's tax bill, itemised.

        Args:
          ordinary:               taxable ordinary income EXCLUDING Social Security
                                  (traditional withdrawals + RMDs + bond/cash interest)
          gains:                  realised LTCG + qualified dividends
          ss:                     gross Social Security received
          investment_income:      NIIT base (dividends + interest + realised gains)
          status:                 filing status key into the bracket tables
          age:                    household age, for the early-withdrawal penalty
          traditional_withdrawal: portion of `ordinary` that came out of the
                                  traditional account (the penalty base)
          spouse_age:             the other spouse's age on a joint return where
                                  both are alive, else None. Only affects how many
                                  times the IRC 63(f) age-65 addition is claimed.
        """
        taxable_ss = self.taxable_social_security(ss, ordinary + gains, status)
        agi = ordinary + gains + taxable_ss
        standard = self.total_standard_deduction(status, age, spouse_age)
        # The standard deduction offsets ordinary income first (the standard
        # ordering); gains then stack on top of what is left.
        taxable_ordinary = max(0.0, ordinary + taxable_ss - standard)
        taxable_total = max(0.0, agi - standard)

        federal_ordinary = self._bracket_tax(taxable_ordinary, self.federal_brackets(status))

        # Long-term gains sit on top of ordinary income: the ordinary income fills
        # the lower LTCG brackets first, so gains are taxed at the rate that applies
        # at that stacked height. This is why a retiree with little ordinary income
        # can realise substantial gains at the 0% rate.
        ltcg = self.ltcg_brackets(status)
        federal_ltcg = (self._bracket_tax(taxable_total, ltcg)
                        - self._bracket_tax(taxable_ordinary, ltcg))

        # Flat state approximation on the same base as federal (no state brackets,
        # no retirement-income exclusions -- see README limitations).
        state = self.cfg.state_tax_rate * max(0.0, agi - standard)

        # Net Investment Income Tax (IRC 1411): 3.8% of the LESSER of net investment
        # income and the AGI excess over the threshold.
        niit_base = min(max(agi - self.niit_threshold(status), 0.0),
                        max(investment_income, 0.0))
        niit = self.cfg.niit_rate * niit_base

        # 10% additional tax on early traditional distributions, unless a 72(t)
        # SEPP / Roth-ladder plan is assumed, or the rule-of-55 exception applies
        # (employer-plan money, separated in or after the year you turn 55).
        penalty = 0.0
        if traditional_withdrawal > 0 and age < self.cfg.penalty_free_age:
            penalty_free = self.cfg.use_72t_sepp
            if self.cfg.assume_qualified_plan_age55_exception and age >= 55:
                penalty_free = True
            if not penalty_free:
                penalty = self.cfg.early_withdrawal_penalty * traditional_withdrawal

        return {
            "taxable_social_security": taxable_ss,
            "agi": agi,
            "federal_ordinary": federal_ordinary,
            "federal_ltcg": federal_ltcg,
            "state": state,
            "niit": niit,
            "early_penalty": penalty,
            "total": federal_ordinary + federal_ltcg + state + niit + penalty,
        }

    def total_tax(self, ordinary: float, gains: float, ss: float,
                  investment_income: float, status: str, age: float,
                  traditional_withdrawal: float = 0.0,
                  spouse_age: Optional[float] = None) -> float:
        """Total federal + state + NIIT + penalty for one year. See tax_breakdown."""
        return self.tax_breakdown(ordinary, gains, ss, investment_income, status,
                                  age, traditional_withdrawal, spouse_age)["total"]


# ==============================
# CORE LIFE SIMULATION
# ==============================
class RetirementSimulator:
    """Owns one Config and everything derived from it: the regime model, the
    per-month rate constants, the tax calculator, the Monte Carlo sweep, and the
    shared results rendering used by both the CLI and the GUI."""

    def __init__(self, config: Config):
        """Validate the pieces that would fail silently, pre-compute the regime
        drifts (including the crisis-drag compensation) and the monthly rate
        constants, and resolve the session seed once (see the comment below)."""
        self.cfg = config
        self.tax = TaxCalculator(config.taxes)
        self.probability_results = {}

        sim = config.simulation
        m = config.market

        crisis_drag_monthly = math.log1p(sim.crisis_regime.annual_return_drag) / 12
        normal_boost_monthly = math.log1p(sim.normal_regime.return_boost) / 12

        # Fraction of all months spent in crisis, in the long run. This is the
        # stationary distribution of the two-state Markov chain: with a 1.5% chance
        # of entering a crisis each month and a 5.5% chance of leaving one, the chain
        # sits in crisis 0.015/(0.015+0.055) = 21% of the time.
        p_enter = sim.normal_regime.monthly_crisis_probability
        p_exit = sim.crisis_regime.monthly_recovery_probability
        self.crisis_fraction = p_enter / (p_enter + p_exit) if (p_enter + p_exit) > 0 else 0.0

        if sim.compensate_crisis_drag and self.crisis_fraction < 1.0:
            # Solve (1-f)*boost + f*drag = configured_boost so that the long-run
            # average log return is exactly what the user asked for. Without this the
            # crisis drag silently subtracts ~1.2 points a year from stock_return.
            normal_boost_monthly = ((normal_boost_monthly - self.crisis_fraction * crisis_drag_monthly)
                                    / (1 - self.crisis_fraction))

        self.regimes = {
            'normal': {
                'return_boost': normal_boost_monthly,
                'vol_mult': sim.normal_regime.volatility_multiplier,
                'p_switch': p_enter,
            },
            'crisis': {
                'return_boost': crisis_drag_monthly,
                'vol_mult': sim.crisis_regime.volatility_multiplier,
                'p_switch': p_exit,
            },
        }

        # Convert annual assumptions to monthly ones, once. Log rates are used for
        # the risky assets so that shocks compound multiplicatively and the portfolio
        # can never go negative from a return.
        self.monthly_log_stock = math.log1p(self.real_return(m.stock_return)) / 12
        self.monthly_log_bond = math.log1p(self.real_return(m.bond_return)) / 12
        self.monthly_cash_rate = self.monthly_rate(self.real_return(m.cash_return))
        self.monthly_stock_vol = m.stock_volatility / math.sqrt(12)
        self.monthly_bond_vol = m.bond_volatility / math.sqrt(12)
        # Expected log price-level rise per month. Used to erode the brokerage cost
        # basis, which is the one quantity in this model that is frozen in NOMINAL
        # dollars (see _market_path).
        self.monthly_log_inflation = math.log1p(m.inflation) / 12

        # Resolve the session seed ONCE so that common random numbers hold across
        # every retirement age of this simulator, even when the user asked for fresh
        # entropy (-1). Resolving per-age would silently break the CRN guarantee.
        self._session_seed = (sim.random_seed if sim.random_seed >= 0
                              else int(np.random.SeedSequence().entropy))

    # ---------- small helpers ----------
    def real_return(self, nominal: float) -> float:
        """Strip expected inflation out of a nominal return.

        Example: 8% nominal with 3% inflation is 1.08/1.03 - 1 = 4.854% real, NOT
        8% - 3% = 5%. The division form is exact; subtraction is an approximation.
        """
        return (1 + nominal) / (1 + self.cfg.market.inflation) - 1

    @staticmethod
    def monthly_rate(annual_rate: float) -> float:
        """Convert an annual rate to the monthly rate that compounds to it.

        Example: 4%/yr becomes 1.04^(1/12) - 1 = 0.327%/mo, which compounds back to
        exactly 4% over twelve months. Dividing by 12 would overstate it.
        """
        return (1 + annual_rate) ** (1 / 12) - 1

    # ---------- mortality ----------
    @staticmethod
    def _actuarial_death_age(sex: str, start_age: float, max_age: float, rng) -> float:
        """Draw one lifespan from a Gompertz mortality curve, starting from today.

        Gompertz (1825) mortality -- the death rate rising exponentially with age,
        q(x) = a*e^(b*x) -- fits adult human mortality remarkably well and is the
        standard actuarial form. Parameters per sex are in GOMPERTZ_PARAMS, chosen so
        survival from 65 matches SSA cohort life table targets (see that comment for
        the table). Each year of age is one Bernoulli trial against q(age); death
        lands uniformly within the year. The draw is capped at max_age
        (death_age_max), which also sizes the trajectory chart.
        """
        b, q75 = GOMPERTZ_PARAMS[sex]
        a = q75 * math.exp(-b * 75)
        age = float(start_age)
        while age < max_age:
            q = min(0.99, a * math.exp(b * age))
            if rng.random() < q:
                # Clamp: the yearly step can straddle the cap whenever
                # max_age - start_age is not a whole number, and death_age_max also
                # sizes the trajectory chart, so an overshoot would index past it.
                return min(age + rng.random(), float(max_age))
            age += 1.0
        return float(max_age)

    def draw_death_age(self, person, person_current_age: float, rng) -> float:
        """One random death age for `person` (LifeEvents or Spouse), in THEIR age scale.

        "ssa_inspired" uses the Gompertz curve above (death_age_min/mean/std ignored,
        death_age_max caps the tail). "normal" is a clipped normal draw -- fully
        user-controlled, e.g. for "plan to exactly 95" experiments.
        """
        if person.mortality_model == "ssa_inspired":
            return self._actuarial_death_age(person.mortality_sex, person_current_age,
                                             person.death_age_max, rng)
        return float(np.clip(rng.normal(person.death_age_mean, person.death_age_std),
                             person.death_age_min, person.death_age_max))

    # ---------- Social Security ----------
    @staticmethod
    def _pia_from_aime(aime: float) -> float:
        """SSA's progressive benefit formula: monthly AIME -> monthly PIA.

        90% of AIME up to the first bend point, 32% between the bend points, 15%
        above (ssa.gov/oact/cola/piaformula.html). The progressivity is why a short
        career costs LESS than a proportional share of the benefit: the earliest
        dollars of average earnings are replaced at 90 cents each.
        """
        if aime <= SS_PIA_BEND1:
            return SS_PIA_RATE1 * aime
        if aime <= SS_PIA_BEND2:
            return SS_PIA_RATE1 * SS_PIA_BEND1 + SS_PIA_RATE2 * (aime - SS_PIA_BEND1)
        return (SS_PIA_RATE1 * SS_PIA_BEND1
                + SS_PIA_RATE2 * (SS_PIA_BEND2 - SS_PIA_BEND1)
                + SS_PIA_RATE3 * (aime - SS_PIA_BEND2))

    @staticmethod
    def _aime_from_pia(pia: float) -> float:
        """Exact inverse of _pia_from_aime (the formula is piecewise linear)."""
        knee1 = SS_PIA_RATE1 * SS_PIA_BEND1
        knee2 = knee1 + SS_PIA_RATE2 * (SS_PIA_BEND2 - SS_PIA_BEND1)
        if pia <= knee1:
            return pia / SS_PIA_RATE1
        if pia <= knee2:
            return SS_PIA_BEND1 + (pia - knee1) / SS_PIA_RATE2
        return SS_PIA_BEND2 + (pia - knee2) / SS_PIA_RATE3

    @staticmethod
    def ss_benefit_factor(claim_age: int, full_retirement_age: int = 67) -> float:
        """Adjust an OWN-RECORD benefit for claiming early or late.

        Early claiming (down to 62): reduced 5/9 of 1% per month for the first 36
        months before FRA, then 5/12 of 1% per month beyond -- claim at 62 with FRA
        67 and the factor is 1 - (36*5/9 + 24*5/12)/100 = 0.70.

        Delayed claiming (up to 70): credited 2/3 of 1% per month -- 8% per year
        SIMPLE, not compounded (ssa.gov/benefits/retirement/planner/delayret.html).
        Claim at 70 with FRA 67 and the factor is 1 + 36*(2/3)/100 = 1.24.

        The default FRA of 67 is correct for anyone born 1960 or later; those born
        1955-1959 have an FRA a few months earlier, which this model does not adjust.
        """
        claim_age = max(62, min(70, int(claim_age)))
        months_diff = (claim_age - full_retirement_age) * 12
        if months_diff >= 0:
            return 1.0 + (2 / 3 * 0.01) * months_diff
        months_early = -months_diff
        if months_early <= 36:
            reduction = (5 / 9 * 0.01) * months_early
        else:
            reduction = (5 / 9 * 0.01) * 36 + (5 / 12 * 0.01) * (months_early - 36)
        return max(0.0, 1 - reduction)

    @staticmethod
    def ss_survivor_factor(claim_age: int, death_age: float,
                           full_retirement_age: int = 67) -> float:
        """Factor on the DECEASED's FRA benefit that a survivor at their own FRA gets.

        A survivor benefit is NOT simply "whatever the deceased's claim age implied",
        because a claim age is only a plan until it is reached:

          - Died BEFORE filing: nothing was ever reduced, so there is no early-claim
            cut no matter how early they had intended to file. Delayed-retirement
            credits, though, are earned month by month, so they accrue only for the
            months actually LIVED past full retirement age. Someone who planned to
            file at 70 but died at 63 leaves a survivor exactly their PIA -- not the
            1.24x they never lived to earn.
          - Died AFTER filing: the survivor steps into the benefit that was being
            paid, but the widow(er)'s limit floors it at 82.5% of PIA
            (SS_WIDOW_LIMIT_FRACTION), so an early claim cannot cut a survivor all
            the way to 70%.

        Returns the factor only; the caller multiplies by the deceased's FRA benefit
        and takes the greater of it and the survivor's own benefit.
        """
        if death_age >= claim_age:
            return max(RetirementSimulator.ss_benefit_factor(claim_age,
                                                             full_retirement_age),
                       SS_WIDOW_LIMIT_FRACTION)
        months_late = max(0.0, min(death_age, 70.0) - full_retirement_age) * 12
        return 1.0 + (2 / 3 * 0.01) * months_late

    @staticmethod
    def ss_spousal_factor(claim_age: int, full_retirement_age: int = 67) -> float:
        """Adjust a SPOUSAL benefit for the claim age. Different rules than own-record:

        Early: reduced 25/36 of 1% per month for the first 36 months, then 5/12 of
        1% per month -- at 62 with FRA 67 that is 25% + 10% = a 35% cut (factor
        0.65), steeper than the 30% cut on an own-record benefit.
        Late: spousal benefits earn NO delayed credits, so the factor caps at 1.0.
        """
        claim_age = max(62, min(70, int(claim_age)))
        months_early = max(0, (full_retirement_age - claim_age) * 12)
        if months_early <= 36:
            reduction = (25 / 36 * 0.01) * months_early
        else:
            reduction = (25 / 36 * 0.01) * 36 + (5 / 12 * 0.01) * (months_early - 36)
        return max(0.0, 1 - reduction)

    def ss_fra_benefit(self, person, years_after_current: float) -> float:
        """Annual benefit at full retirement age, adjusted for an incomplete career.

        The user supplies the FRA benefit for a FULL 35-year career (from
        ssa.gov/myaccount). SSA averages your best `ss_benefit_years_required` (35)
        years of indexed earnings into the AIME; every year short of that is a zero
        in the average. This method:

          1. checks the 40-credit eligibility rule -- roughly ten years of covered
             work. Fewer credits at retirement means NO benefit at all;
          2. inverts the full-career benefit to its AIME;
          3. scales the AIME by covered_years/35 (zeros diluting the average is
             exactly linear in years, assuming level real earnings);
          4. re-applies the progressive PIA formula.

        Because the formula replaces the first dollars at 90%, a 60% career yields
        MORE than 60% of the full benefit -- unlike the old linear proration, which
        SSA's own formula does not use and which understates early retirees'
        benefits.

        Example: full benefit $40,000/yr (PIA $3,333/mo -> AIME $8,314). Retiring
        with 21 of 35 years gives AIME $4,989 -> PIA $2,342 -> $28,100/yr. That is
        70% of the full benefit from 60% of the career.
        """
        years_worked = person.ss_earnings_years_at_current_age + max(0, years_after_current)
        credits = min(40, int(round(person.ss_credits_at_current_age
                                    + 4 * max(0, years_after_current))))
        if credits < self.cfg.life_events.ss_retirement_eligibility_credits:
            return 0.0
        full_pia_monthly = max(0.0, person.ss_annual_full_retirement_benefit / 12.0)
        if full_pia_monthly == 0.0:
            return 0.0
        # The 35-year averaging window is statutory and identical for everyone, so
        # it is configured once, on life_events.
        averaging_years = self.cfg.life_events.ss_benefit_years_required
        years_ratio = min(1.0, years_worked / max(1, averaging_years))
        full_aime = self._aime_from_pia(full_pia_monthly)
        return self._pia_from_aime(full_aime * years_ratio) * 12.0

    def primary_ss_income(self, retirement_age: int,
                          work_years_after_current: Optional[float] = None) -> float:
        """Annual Social Security for the primary person, in today's dollars.

        work_years_after_current caps the covered-work accrual: by default work
        continues until retirement_age, but simulate_life passes the years actually
        worked in THAT lifetime -- someone who dies at 45 while a retirement age of
        55 is being tested stopped accruing earnings (and credits) at 45, and the
        survivor benefit built on their record must reflect that.
        """
        le = self.cfg.life_events
        if work_years_after_current is None:
            work_years_after_current = retirement_age - self.cfg.simulation.current_age
        fra_benefit = self.ss_fra_benefit(le, work_years_after_current)
        return fra_benefit * self.ss_benefit_factor(le.ss_claim_age)

    def spouse_ss_income(self, retirement_age: int,
                         primary_work_years: Optional[float] = None,
                         spouse_work_years: Optional[float] = None) -> float:
        """Annual Social Security for the spouse. Zero when no spouse is configured.

        Social Security pays a spouse the GREATER of two things:
          - their own benefit, based on their own earnings record, or
          - the SPOUSAL benefit, worth up to 50% of the primary's PIA, which requires
            no earnings record of their own at all.

        This matters enormously for a stay-at-home partner. With a primary benefit of
        $40,000 and no work history, the spouse still collects up to $20,000/yr once
        they reach their claim age -- not zero.

        Two rules the spousal benefit does NOT share with an own-record benefit:
          - delaying past full retirement age earns no extra credit, and claiming
            early cuts it on a steeper schedule (see ss_spousal_factor), and
          - it is unaffected by how few years the spouse worked (though the PRIMARY's
            record still determines its size).
        Approximation: the model does not require the primary to have filed first.

        Both persons' covered work is assumed to continue until the household
        retires. The *_work_years overrides let simulate_life cap each record at
        that lifetime's actual work stop (the earlier of retirement and death), so
        a record frozen by an early death is not credited with phantom years.
        """
        sp = self.cfg.spouse
        if not sp.enabled:
            return 0.0
        default_years = retirement_age - self.cfg.simulation.current_age
        if primary_work_years is None:
            primary_work_years = default_years
        if spouse_work_years is None:
            spouse_work_years = default_years

        own = (self.ss_fra_benefit(sp, spouse_work_years)
               * self.ss_benefit_factor(sp.ss_claim_age))

        # The spousal benefit keys off the primary's PIA -- their benefit at THEIR
        # full retirement age, before any adjustment for when the primary claims.
        primary_pia = self.ss_fra_benefit(self.cfg.life_events, primary_work_years)
        spousal = 0.5 * primary_pia * self.ss_spousal_factor(sp.ss_claim_age)

        return max(own, spousal)

    # ---------- filing status and spending ----------
    def household_filing_status(self, primary_alive: bool, spouse_alive: bool) -> str:
        """Filing status for the coming tax year, given who is alive.

        A joint filer whose spouse has died files single thereafter -- half the
        standard deduction and compressed brackets, the "widow's tax penalty".
        (The actual rules allow MFJ in the year of death and qualifying-surviving-
        spouse for two years with dependents; retirees rarely qualify, so the model
        switches to single from the next plan year -- slightly conservative.)
        """
        if primary_alive and spouse_alive:
            return self.cfg.taxes.filing_status
        if self.cfg.taxes.filing_status in ("married_filing_jointly",
                                            "qualifying_surviving_spouse"):
            return "single"
        return self.cfg.taxes.filing_status

    def spending_smile(self, household_age: float):
        """The retirement spending path: (base_spending_factor, healthcare_multiplier).

        Between decline_start_age and decline_end_age, base spending falls by
        annual_spending_decline_rate each year (0.99 = -1%/yr, per Blanchett's
        observed "retirement spending smile"). After decline_end_age the base holds
        at its declined level, and the late-life upturn is carried by the healthcare
        side: the per-person premiums grow at annual_healthcare_increase_rate
        (medical costs outrunning CPI). Keyed to household_age so a surviving spouse
        follows their OWN curve rather than that of the person who died.
        """
        s = self.cfg.spending
        years_declining = max(0.0, min(household_age, s.spending_decline_end_age)
                              - s.spending_decline_start_age)
        base_factor = s.annual_spending_decline_rate ** years_declining
        years_past_end = max(0.0, household_age - s.spending_decline_end_age)
        healthcare_multiplier = (1 + s.annual_healthcare_increase_rate) ** years_past_end
        return base_factor, healthcare_multiplier

    # ---------- market path ----------
    def _market_path(self, months: int, rng):
        """Pre-generate the whole life's monthly market path from one seeded rng.

        Returns (stock_growth, bond_growth, cash_growth, basis_decay). The first
        three are arrays of simple monthly REAL returns; the fourth is the monthly
        factor by which a frozen NOMINAL dollar loses real value (see below).
        Drawing everything up front from the run's own Generator is what makes runs
        reproducible and lets common-random-numbers hold across retirement ages (the
        path depends only on the seed and the number of months, never on the
        retirement age being tested).

        Distribution details:
          - Stock shocks are Student-t with `return_distribution_degrees_of_freedom`.
            A t with df degrees of freedom has variance df/(df-2), so the draws are
            scaled by sqrt((df-2)/df) to unit variance -- that way stock_volatility
            means exactly what it says while the t still supplies fat tails (df=6
            reproduces the historical frequency of large one-year losses).
          - The regime chain (calm/crisis) modulates the stock drift and volatility.
            Crisis LENGTH is geometrically distributed with mean
            1/monthly_recovery_probability.
          - Bond shocks are built from STANDARDISED draws: corr*z_stock +
            sqrt(1-corr^2)*z_bond, scaled by the bond's own volatility. This makes
            the realised bond volatility equal bond_volatility and the realised
            correlation equal stock_bond_correlation in BOTH regimes. (Correlating
            the already-scaled stock shock would leak stock volatility -- and the
            crisis multiplier -- into the bond series.)
          - Bonds and cash promise a fixed NOMINAL payment, so a positive inflation
            surprise comes straight off their real return. Stocks are treated as
            inflation-neutral over the long run and are left alone.
          - basis_decay carries the same realised inflation to the brokerage cost
            basis. Every balance in this model floats with the price level and is
            therefore constant in real terms, but a cost basis does NOT: it is the
            historical dollar figure on the 1099, frozen the day you bought, and the
            IRS never indexes it. So its REAL value must shrink every month by
            exactly the realised inflation -- which is why inflation alone creates a
            taxable "gain". Holding the real basis flat would tax only the real
            appreciation and understate the bill (a $10k lot held 30 years at 8%
            nominal / 3% inflation is 90.1% taxable gain, not 75.9%).
        """
        m = self.cfg.market
        df = self.cfg.simulation.return_distribution_degrees_of_freedom

        t_scale = math.sqrt((df - 2) / df)
        # numpy's own Student-t, not scipy's. scipy.stats.t.rvs(random_state=rng)
        # delegates straight to this and consumes the stream identically -- verified
        # bit-for-bit at df 3/6/30 -- so dropping the dependency changes no result
        # and takes ~97MB out of a packaged build.
        z_stock = rng.standard_t(df, size=months) * t_scale
        z_bond = rng.standard_normal(months)
        regime_rolls = rng.random(months)
        inflation_shocks = rng.normal(0.0, m.inflation_volatility / math.sqrt(12),
                                      size=months)

        # Walk the two-state Markov chain (inherently sequential), then vectorise
        # the return math over the whole path.
        #
        # The chain STARTS FROM ITS STATIONARY DISTRIBUTION, not deterministically
        # calm. Starting every life in 'normal' would be a burn-in bias: a finite
        # path would spend measurably less time in crisis than the `crisis_fraction`
        # the drag compensation in __init__ solves against, quietly handing every
        # simulated life a few free months of calm-market returns (~+4% terminal
        # wealth on the stock leg, and worse the shorter the horizon). Because a
        # stationary chain stays stationary after one step, drawing the state here
        # -- before the loop's own switch test -- keeps every month stationary.
        # This draw comes last so the arrays above are untouched by it.
        boosts = np.empty(months)
        vol_mults = np.empty(months)
        regime = 'crisis' if rng.random() < self.crisis_fraction else 'normal'
        for i in range(months):
            params = self.regimes[regime]
            if regime_rolls[i] < params['p_switch']:
                regime = 'crisis' if regime == 'normal' else 'normal'
                params = self.regimes[regime]
            boosts[i] = params['return_boost']
            vol_mults[i] = params['vol_mult']

        corr = m.stock_bond_correlation
        corr_complement = math.sqrt(max(0.0, 1 - corr * corr))

        stock_shock = z_stock * self.monthly_stock_vol * vol_mults
        bond_shock = self.monthly_bond_vol * (corr * z_stock + corr_complement * z_bond)

        stock_growth = np.exp(self.monthly_log_stock + boosts + stock_shock) - 1
        bond_growth = np.exp(self.monthly_log_bond + bond_shock - inflation_shocks) - 1
        cash_growth = self.monthly_cash_rate - inflation_shocks
        # Same realised price path the bonds just felt, applied to frozen dollars.
        basis_decay = np.exp(-(self.monthly_log_inflation + inflation_shocks))
        return stock_growth, bond_growth, cash_growth, basis_decay

    # ---------- the annual withdrawal plan ----------
    def _investment_income(self, brokerage: float, cash: float,
                           stock_alloc: float, bond_alloc: float):
        """This year's taxable-account income: (qualified dividends, bond interest,
        cash interest), estimated from the start-of-year balances.

        Total return already contains these distributions -- the balances are not
        credited again. What the distributions change is TAX: they are taxed every
        year even if nothing is sold (dividends at LTCG rates, interest at ordinary
        rates), and once taxed and reinvested they raise the brokerage cost basis.
        Cash interest is deliberately the NOMINAL yield: the IRS taxes nominal
        interest, which is precisely why cash loses real value after tax.
        """
        m = self.cfg.market
        qualified_dividends = brokerage * stock_alloc * m.stock_dividend_yield
        bond_interest = brokerage * bond_alloc * m.bond_taxable_yield
        cash_interest = cash * m.cash_return
        return qualified_dividends, bond_interest, cash_interest

    @staticmethod
    def _ladder_withdraw(gross: float, balances, basis_fraction: float) -> dict:
        """Take `gross` dollars through the ladder: cash -> brokerage -> traditional -> Roth.

        Ordered cheapest-tax-first, which also leaves the tax-free Roth compounding
        the longest:
          1) Cash: already taxed, a dollar out is a dollar spent.
          2) Brokerage: only the GAIN share of a sale is taxable, at LTCG rates.
             Selling is pro-rata: selling 10% of the account realises 10% of the gain
             and removes 10% of the basis.
          3) Traditional: fully taxable as ordinary income, plus the 10% penalty when
             it applies.
          4) Roth: tax-free, kept for last.

        `basis_fraction` is basis/balance. It is normally below 1, and then the gain
        share is simply 1 - basis_fraction. It exceeds 1 when the account is
        UNDERWATER (worth less than was paid for it), and the two quantities then
        part company: the taxable gain floors at zero (this model does not harvest
        capital losses), but the basis removal stays strictly pro-rata, because the
        shares sold carried their full share of the cost. Clamping the basis down to
        the balance instead would permanently delete the excess and tax the eventual
        recovery as if it were profit.

        Returns the amount taken from each rung plus the realised gain and the basis
        removed by the brokerage sale. Pure mechanics -- no tax is computed here.
        """
        cash, brokerage, traditional, roth = balances
        take_cash = min(cash, gross)
        remaining = gross - take_cash
        take_brokerage = min(brokerage, remaining)
        remaining -= take_brokerage
        take_traditional = min(traditional, remaining)
        remaining -= take_traditional
        take_roth = min(roth, remaining)
        remaining -= take_roth
        return {
            "cash": take_cash,
            "brokerage": take_brokerage,
            "traditional": take_traditional,
            "roth": take_roth,
            "realized_gain": take_brokerage * max(0.0, 1.0 - basis_fraction),
            "basis_returned": take_brokerage * basis_fraction,
            "unfunded": remaining,
        }

    def _annual_withdrawal_plan(self, net_need: float, balances, brokerage_basis: float,
                                household_age: float, status: str, ss_income: float,
                                stock_alloc: float, bond_alloc: float,
                                rmd_amount: float,
                                spouse_age: Optional[float] = None) -> Optional[dict]:
        """Plan one tax year exactly: how much to withdraw, from where, and the tax.

        Solves for the smallest total gross withdrawal whose AFTER-TAX proceeds cover
        `net_need` (the year's spending not covered by Social Security), where:
          - the RMD is a forced first slice out of the traditional account (its
            after-tax value counts toward the need -- forced money is still money);
          - anything further follows the cash -> brokerage -> traditional -> Roth
            ladder;
          - the tax bill is computed EXACTLY from the actual composition: ordinary
            income (traditional + RMD + bond/cash interest), gains (realised LTCG +
            qualified dividends), SS taxability, NIIT, and the early penalty. No
            marginal-rate approximation -- bracket crossings, the SS tax torpedo and
            the 0% LTCG band all land exactly where the real 1040 puts them.

        Because every marginal component taxes less than 100 cents per dollar, net
        proceeds rise monotonically with the gross withdrawal, so a bisection on the
        extra-above-RMD converges to the exact answer (to well under a cent here).

        Two housekeeping steps that mirror a real brokerage account:
          - this year's dividends and interest, being reinvested after tax, are added
            to the cost basis before the sale's gain fraction is computed;
          - if the RMD forces out MORE after-tax money than the year needs, the
            surplus is reinvested in the brokerage (raising its basis), not left idle.

        Returns None when even liquidating everything cannot cover the need -- the
        plan has failed. Otherwise a dict with the new balances/basis, the total
        gross and tax, and the per-rung takes.
        """
        cash, brokerage, traditional, roth = balances
        rmd_amount = min(rmd_amount, traditional)

        qdiv, bond_interest, cash_interest = self._investment_income(
            brokerage, cash, stock_alloc, bond_alloc)

        # Distributions were reinvested after tax, so they are new basis. The
        # balance itself is untouched (total return already includes them). Basis is
        # deliberately NOT capped at the balance: an account can be worth less than
        # was paid for it, and capping would delete the difference for good.
        brokerage_basis = brokerage_basis + qdiv + bond_interest
        basis_fraction = brokerage_basis / brokerage if brokerage > 0 else 0.0

        # Income that exists before any discretionary withdrawal.
        base_ordinary = bond_interest + cash_interest + rmd_amount
        base_gains = qdiv
        base_investment_income = qdiv + bond_interest + cash_interest

        # The RMD has already been carved out of the traditional rung.
        available = (cash, brokerage, traditional - rmd_amount, roth)
        max_extra = sum(available)

        def evaluate(extra: float):
            """Net after-tax proceeds if we withdraw `extra` beyond the RMD."""
            takes = self._ladder_withdraw(extra, available, basis_fraction)
            ordinary = base_ordinary + takes["traditional"]
            gains = base_gains + takes["realized_gain"]
            investment_income = base_investment_income + takes["realized_gain"]
            tax = self.tax.total_tax(
                ordinary, gains, ss_income, investment_income, status, household_age,
                traditional_withdrawal=rmd_amount + takes["traditional"],
                spouse_age=spouse_age)
            return takes, tax, rmd_amount + extra - tax

        takes, tax, net = evaluate(max_extra)
        if net + 1e-9 < net_need:
            return None                       # broke: the whole portfolio can't fund the year

        takes, tax, net = evaluate(0.0)
        if net < net_need:
            # Bisect on the extra withdrawal. 60 halvings of even a $100M interval
            # resolve to far below a cent; the tolerance exits long before that.
            lo, hi = 0.0, max_extra
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                _, _, net_mid = evaluate(mid)
                if net_mid >= net_need:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < 0.005:
                    break
            takes, tax, net = evaluate(hi)    # hi always satisfies net >= net_need
            extra = hi
        else:
            extra = 0.0

        cash -= takes["cash"]
        brokerage -= takes["brokerage"]
        traditional -= rmd_amount + takes["traditional"]
        roth -= takes["roth"]
        brokerage_basis -= takes["basis_returned"]

        # An RMD larger than the year's need leaves after-tax money on the table;
        # a real retiree reinvests it in the taxable account (new shares, new basis).
        surplus = net - net_need
        if surplus > 0:
            brokerage += surplus
            brokerage_basis += surplus
        # Floor at zero only. Basis legitimately exceeds the balance in an
        # underwater account, and capping it here would silently forgive the loss
        # and re-tax the recovery.
        brokerage_basis = max(0.0, brokerage_basis)

        return {
            "balances": (cash, brokerage, traditional, roth),
            "new_basis": brokerage_basis,
            "gross_total": rmd_amount + extra,
            "tax": tax,
            "rmd": rmd_amount,
            "surplus_reinvested": max(0.0, surplus),
            "take_cash": takes["cash"],
            "take_brokerage": takes["brokerage"],
            "take_traditional": rmd_amount + takes["traditional"],
            "take_roth": takes["roth"],
            "realized_gain": takes["realized_gain"],
        }

    def rmd_divisor(self, age: float) -> Optional[float]:
        """IRS Uniform Lifetime Table divisor, or None before the configured start age."""
        if age < self.cfg.simulation.rmd_start_age:
            return None
        return RMD_TABLE.get(min(int(age), 120), RMD_TABLE[120])

    # ---------- the model ----------
    def simulate_life(self, retirement_age: int, random_seed: Optional[int] = None,
                      record_trajectory: bool = False):
        """Simulate ONE random lifetime, month by month.

        Returns (survived, min_portfolio_in_retirement, initial_withdrawal_rate,
                 final_portfolio, trajectory).

        survived is False if the money ran out before the last death.
        initial_withdrawal_rate is the first year's GROSS withdrawal (spending
        shortfall plus the taxes to fund it) over the portfolio at retirement -- the
        number comparable to the "4% rule". trajectory is None unless
        record_trajectory is set, in which case it holds the portfolio total once per
        year from the tested retirement age onwards.

        Structure: markets move monthly; money is PLANNED annually. At each
        retirement anniversary the model sets the coming year's spending (smile,
        guardrails, survivor factor, healthcare premiums), nets off Social Security,
        computes any RMD, and calls _annual_withdrawal_plan to pull the year's needs
        out of the accounts in one exactly-taxed transaction into a spending bucket,
        which then pays the bills monthly. This mirrors how taxes actually work
        (annually) and costs a small, realistic cash drag: the year's spending money
        sits out of the market once withdrawn.

        Two timing consequences of annual planning, both mildly conservative: a
        death mid-plan-year leaves the rest of that year's (already-taxed) funding
        in the spending bucket, where it simply counts toward the final estate; and
        a mid-year death or spending change is not re-planned until the next
        anniversary.
        """
        cfg = self.cfg
        sp, hc = cfg.spouse, cfg.healthcare
        current_age = cfg.simulation.current_age
        rng = np.random.default_rng(random_seed)

        # ---- how long does this simulated life last? ----
        # Each person gets their own random death age, drawn FIRST so that with
        # common random numbers every retirement age sees the same lifespans. The
        # simulation must run until the LAST death: the survivor keeps spending.
        death_age = self.draw_death_age(cfg.life_events, current_age, rng)
        if sp.enabled:
            # The spouse's death age is in THEIR age scale; convert to the primary's
            # clock. A spouse two years older (offset +2) who dies at 90 does so when
            # the primary is 88.
            spouse_death_own = self.draw_death_age(sp, current_age + sp.age_offset, rng)
            spouse_death_primary_clock = spouse_death_own - sp.age_offset
            last_age = max(death_age, spouse_death_primary_clock)
        else:
            spouse_death_own = -math.inf
            spouse_death_primary_clock = -math.inf
            last_age = death_age

        living_months = max(int(round((last_age - current_age) * 12)), 1)
        retirement_month = (retirement_age - current_age) * 12

        # How many yearly trajectory slots a life this long fills: one per
        # retirement anniversary actually lived through. A run that goes broke early
        # returns before recording the rest, so `_broke` below pads the remainder
        # with zeros -- the household really is at $0 for those years, and it stays
        # in the sample only for as long as it is ALIVE. Padding past death instead
        # (to death_age_max) would leave dead-and-broke runs in the percentile while
        # dead-and-solvent runs correctly drop out, dragging the low percentiles of
        # the fan chart toward zero at the ages with the fewest survivors.
        trajectory_slots = max(0, -(-(living_months - retirement_month) // 12))

        # ---- pre-generate every random number this life needs ----
        stock_g, bond_g, cash_g, basis_decay = self._market_path(living_months, rng)

        roth = float(cfg.accounts.roth)
        traditional = float(cfg.accounts.traditional)
        brokerage = float(cfg.accounts.brokerage)
        brokerage_basis = min(float(cfg.accounts.brokerage_cost_basis), brokerage)
        cash = float(cfg.accounts.cash)
        # The "checking account": funded once a year by the withdrawal plan, drained
        # monthly by the bills. Counted in the portfolio total, earns nothing.
        spending_cash = 0.0

        base_contribs = (cfg.contributions.annual_roth / 12,
                         cfg.contributions.annual_traditional / 12,
                         cfg.contributions.annual_brokerage / 12,
                         cfg.contributions.annual_cash / 12)
        contrib_growth_rate = cfg.contributions.annual_contribution_growth_rate

        # ---- Social Security, from the work record THIS lifetime produces ----
        # Covered work stops at the earlier of the tested retirement age and death,
        # so a worker who dies young leaves a smaller record -- and the survivor
        # benefit built on that record shrinks accordingly. Without the cap, every
        # lifetime would be credited with earnings through the tested retirement
        # age, quietly overpaying survivors of an early death.
        primary_work_years = max(0.0, min(retirement_age, death_age) - current_age)
        spouse_work_years = (max(0.0, min(retirement_age, spouse_death_primary_clock)
                                  - current_age) if sp.enabled else 0.0)
        primary_ss_annual = self.primary_ss_income(retirement_age, primary_work_years)
        spouse_ss_annual = self.spouse_ss_income(retirement_age, primary_work_years,
                                                 spouse_work_years)

        # What each person's death would leave the other, from THEIR OWN record.
        # Built from the FRA benefit and ss_survivor_factor rather than reusing the
        # living benefits above, because a survivor benefit follows different rules:
        # no early-claim cut if the deceased never filed, delayed credits only for
        # months actually lived, and an 82.5%-of-PIA floor. A spousal benefit is
        # never inheritable, so the spouse's side uses their OWN record only -- a
        # partner with no earnings record leaves no survivor benefit.
        primary_survivor_annual = (
            self.ss_fra_benefit(cfg.life_events, primary_work_years)
            * self.ss_survivor_factor(cfg.life_events.ss_claim_age, death_age))
        spouse_survivor_annual = (
            self.ss_fra_benefit(sp, spouse_work_years)
            * self.ss_survivor_factor(sp.ss_claim_age, spouse_death_own)
            if sp.enabled else 0.0)

        glide_years = max(cfg.simulation.glide_path_years, 1)

        retired = False
        retirement_start_month = None    # first month of drawdown (may precede the
        retirement_portfolio = None      # tested age if the primary dies early)
        initial_withdrawal_rate = None
        spending_multiplier = 1.0        # moved by guardrails; 1.0 = spending to plan
        year_start_investable = None     # invested balances right after each annual plan
        monthly_draw = 0.0
        min_portfolio = math.inf         # tracked ONLY from the tested retirement age
        trajectory = [] if record_trajectory else None

        def _broke():
            """The one way this simulation fails: the money ran out while someone
            was still alive. Fills the rest of THIS life's trajectory with zeros so
            every returned trajectory is exactly as long as the retirement the
            person lived -- the invariant compute_trajectory_percentiles relies on
            to keep each column's sample to those still alive at that age."""
            if trajectory is not None:
                trajectory.extend([0.0] * (trajectory_slots - len(trajectory)))
            return (False, min(min_portfolio, 0.0), initial_withdrawal_rate,
                    0.0, trajectory)

        for month in range(living_months):
            age = current_age + month / 12
            primary_alive = age < death_age
            spouse_alive = sp.enabled and age < spouse_death_primary_clock
            if not primary_alive and not spouse_alive:
                break

            # Age of whoever is still here, used for the spending smile, filing
            # status, penalties and RMDs. Once the primary dies the household clock
            # belongs to the survivor: a spouse three years younger reaches the RMD
            # age three years later.
            household_age = age if primary_alive else age + sp.age_offset

            # Contributions come from the primary's earnings, so they stop the moment
            # the primary dies. A surviving spouse has no salary to save from and must
            # start living off the portfolio immediately -- which means the household
            # is effectively retired from that date, even if it is years before the
            # retirement age being tested.
            is_retired = (month >= retirement_month) or (sp.enabled and not primary_alive)

            # ---- asset allocation ----
            # The glide path is anchored at RETIREMENT, not at today: you hold
            # glide_path_start_stock_pct right up to the day you retire, then move
            # linearly to glide_path_end_stock_pct over glide_path_years (rising OR
            # falling -- both directions are valid strategies).
            if cfg.simulation.glide_path:
                frac = min(1.0, max(0.0, age - retirement_age) / glide_years)
                stock_alloc = (cfg.simulation.glide_path_start_stock_pct
                               + (cfg.simulation.glide_path_end_stock_pct
                                  - cfg.simulation.glide_path_start_stock_pct) * frac)
            else:
                stock_alloc = cfg.simulation.static_stock_allocation
            bond_alloc = 1 - stock_alloc

            # ---- this month's growth ----
            # Roth, traditional and brokerage all hold the same stock/bond mix.
            # Cash is a separate bucket and is NOT part of that mix -- so if cash is
            # 17% of your portfolio, a "90% stocks" glide path is really 74% equity.
            # Floor at a tiny positive number: a portfolio can approach zero but a
            # single month can never make it negative.
            growth_factor = max(1e-10, 1 + stock_alloc * stock_g[month]
                                + bond_alloc * bond_g[month])
            roth *= growth_factor
            traditional *= growth_factor
            brokerage *= growth_factor        # basis is unchanged by growth
            cash *= (1 + cash_g[month])

            # The cost basis is the only figure here fixed in nominal dollars: the
            # IRS taxes the gain over what you actually paid and never indexes it.
            # Expressed in this model's today's-dollars, it therefore shrinks with
            # the price level -- so inflation itself manufactures a taxable gain.
            brokerage_basis *= basis_decay[month]

            if not is_retired:
                # ================= ACCUMULATION =================
                # Contributions grow at annual_contribution_growth_rate. Because this
                # model is in today's dollars, that rate is growth ABOVE inflation:
                # 0.0 keeps your savings flat in purchasing power.
                growth = (1 + contrib_growth_rate) ** (month / 12)
                roth += base_contribs[0] * growth
                traditional += base_contribs[1] * growth
                brokerage += base_contribs[2] * growth
                brokerage_basis += base_contribs[2] * growth   # you paid for these
                cash += base_contribs[3] * growth
            else:
                # ================= RETIREMENT =================
                if not retired:
                    # First month of drawing down, whether that is the retirement age
                    # being tested or an earlier forced start caused by the primary's
                    # death.
                    retired = True
                    retirement_start_month = month
                    retirement_portfolio = roth + traditional + brokerage + cash

                # ---- plan one tax year at a time, on the retirement anniversary ----
                if (month - retirement_start_month) % 12 == 0:
                    investable = roth + traditional + brokerage + cash

                    # -- guardrails, judged on the completed year's own return --
                    # Withdrawals happen only at anniversaries, so the invested
                    # balances evolve untouched in between and the ratio below IS the
                    # exact market return the retiree just experienced.
                    if month != retirement_start_month and year_start_investable:
                        implied_return = investable / year_start_investable - 1
                        if implied_return < cfg.spending.guardrail_cut_return_threshold:
                            spending_multiplier = max(
                                spending_multiplier * cfg.spending.guardrail_cut_amount,
                                cfg.spending.guardrail_cut_floor)
                        elif implied_return > cfg.spending.guardrail_raise_return_threshold:
                            spending_multiplier = min(
                                spending_multiplier * cfg.spending.guardrail_raise_amount,
                                cfg.spending.guardrail_raise_ceiling)

                    # -- the coming year's spending target --
                    base_factor, health_multiplier = self.spending_smile(household_age)

                    # After one death the household spends less, but nowhere near
                    # half -- rent, utilities and insurance barely change. Only a
                    # modelled couple can drop to one person; a single-person plan
                    # always spends the full amount for as long as they are alive.
                    survivors = int(primary_alive) + int(spouse_alive)
                    household_factor = (sp.survivor_spending_factor
                                        if sp.enabled and survivors == 1 else 1.0)

                    # Health insurance, per living person, ADDED to expenses.
                    # Pre-Medicare cover bought on the open market is dramatically
                    # more expensive, which is what makes retiring in your 40s and
                    # 50s so costly.
                    # The spouse's own age, on the primary's clock. None once the
                    # household is down to one person, which is exactly when the
                    # return stops being joint.
                    spouse_age = age + sp.age_offset if spouse_alive else None

                    annual_health = 0.0
                    if primary_alive:
                        annual_health += (hc.pre_medicare_annual_premium
                                          if age < hc.medicare_age
                                          else hc.medicare_annual_premium)
                    if spouse_alive:
                        annual_health += (hc.pre_medicare_annual_premium
                                          if spouse_age < hc.medicare_age
                                          else hc.medicare_annual_premium)
                    annual_health *= health_multiplier

                    annual_spending = (cfg.spending.initial_annual_expenses * base_factor
                                       * spending_multiplier * household_factor
                                       + annual_health)

                    # -- Social Security expected over the coming 12 months --
                    # Walked month by month so a mid-year death or claim start is
                    # captured: while both are alive each collects their own benefit
                    # once past their claim age; after the first death the survivor
                    # steps up to the LARGER of the two (the survivor benefit rule --
                    # household Social Security always falls when someone dies).
                    # steps up to the greater of their OWN benefit and the survivor
                    # benefit on the deceased's record (see ss_survivor_factor --
                    # which is emphatically NOT the deceased's living benefit).
                    # Household Social Security usually falls at a death, but it can
                    # rise when the deceased was the higher earner and had not yet
                    # filed, exactly as under the real rules.
                    # Approximation: the survivor's step-up starts at their own claim
                    # age; real widow(er) benefits can start at 60 on their own
                    # reduction schedule, which this model does not draw finely.
                    ss_total = 0.0
                    for mo in range(12):
                        test_age = age + mo / 12
                        p_alive = primary_alive and test_age < death_age
                        s_alive = spouse_alive and test_age < spouse_death_primary_clock
                        p_claiming = p_alive and test_age >= cfg.life_events.ss_claim_age
                        s_claiming = s_alive and (test_age + sp.age_offset) >= sp.ss_claim_age
                        if p_alive and s_alive:
                            ss_total += (primary_ss_annual / 12 if p_claiming else 0.0)
                            ss_total += (spouse_ss_annual / 12 if s_claiming else 0.0)
                        elif p_alive and p_claiming:
                            ss_total += max(primary_ss_annual, spouse_survivor_annual) / 12
                        elif s_alive and s_claiming:
                            ss_total += max(spouse_ss_annual, primary_survivor_annual) / 12

                    # SS above spending is assumed absorbed, not reinvested -- a
                    # small conservative simplification.
                    net_need = max(annual_spending - ss_total, 0.0)

                    # -- Required Minimum Distribution --
                    # Uses the balance at this anniversary, which IS the prior year's
                    # ending balance (the IRS December 31 rule, on the model's
                    # anniversary-anchored years). Keyed to household_age: a
                    # surviving spouse who rolls the account over takes RMDs on THEIR
                    # clock, and distributions do not stop merely because the
                    # original owner died.
                    divisor = self.rmd_divisor(household_age)
                    rmd_amount = (min(traditional, traditional / divisor)
                                  if divisor else 0.0)

                    status = self.household_filing_status(primary_alive, spouse_alive)
                    plan = self._annual_withdrawal_plan(
                        net_need, (cash, brokerage, traditional, roth), brokerage_basis,
                        household_age, status, ss_total, stock_alloc, bond_alloc,
                        rmd_amount,
                        # Only a live joint return has a second 63(f) claimant; once
                        # the primary dies household_age IS the survivor's age.
                        spouse_age=spouse_age if primary_alive else None)

                    if plan is None:
                        # Even liquidating everything cannot fund the year.
                        return _broke()

                    cash, brokerage, traditional, roth = plan["balances"]
                    brokerage_basis = plan["new_basis"]
                    spending_cash += net_need
                    monthly_draw = net_need / 12.0

                    # Anchor for next year's guardrail check: what stays invested.
                    year_start_investable = roth + traditional + brokerage + cash

                    if initial_withdrawal_rate is None and retirement_portfolio:
                        # The classic "safe withdrawal rate" headline: the first
                        # year's GROSS withdrawal (spending shortfall plus the taxes
                        # to fund it) divided by the portfolio at retirement. 4% is
                        # the usual rule-of-thumb benchmark.
                        initial_withdrawal_rate = plan["gross_total"] / retirement_portfolio

                # ---- pay this month's bills from the funded bucket ----
                # The bucket was funded with exactly twelve draws; this check only
                # guards floating-point dust and future refactors.
                if spending_cash + 1e-6 < monthly_draw:
                    return _broke()
                spending_cash -= monthly_draw

            total = roth + traditional + brokerage + cash + spending_cash

            # Record the portfolio once a year from the TESTED retirement age, for
            # the fan chart -- anchored there (not at a survivor-forced earlier
            # start) so every run's column i means "age retirement_age + i".
            if trajectory is not None and month >= retirement_month \
                    and (month - retirement_month) % 12 == 0:
                trajectory.append(total)

            # Track the worst point DURING RETIREMENT only. Including the
            # accumulation years would just report the dip you take in your 30s on a
            # small portfolio, which says nothing about whether retirement is safe.
            if month >= retirement_month:
                min_portfolio = min(min_portfolio, total)

            # Invariant guard only: every real failure is caught above (the plan
            # returning None, or the spending bucket running dry). Balances cannot
            # go negative by construction, and a portfolio at exactly $0 with all
            # bills paid is a SUCCESS (e.g. the last dollar spent the month before
            # death, or a retiree living entirely on Social Security) -- so this
            # must never fire on zero, only on an impossible negative state
            # introduced by a future refactor.
            if retired and total < 0:
                return _broke()

        final_total = roth + traditional + brokerage + cash + spending_cash
        if min_portfolio == math.inf:
            # Never reached the tested retirement age -- this simulated life ended
            # first. It counts as a success (the money did outlast the person),
            # which is the right accounting but does slightly flatter very late
            # retirement ages.
            min_portfolio = final_total
        return True, min_portfolio, initial_withdrawal_rate, final_total, trajectory

    # ==============================
    # MONTE CARLO
    # ==============================
    def _run_seeds(self, retirement_age: int, n: int) -> List[int]:
        """Deterministic per-run seeds.

        With common_random_numbers on, the seeds depend only on the run index, so
        every retirement age faces the SAME n lifetimes -- differences between ages
        are then pure decision effect, not sampling luck. With it off, each age gets
        its own independent scenario set (larger error bars on comparisons, but
        every age's estimate is independent). The stride is an arbitrary prime.
        """
        if self.cfg.simulation.common_random_numbers:
            return [self._session_seed + i * 104729 for i in range(n)]
        return [self._session_seed + retirement_age * 1000003 + i * 104729
                for i in range(n)]

    def retirement_probability(self, retirement_age: int):
        """Run monte_carlo_runs independent lifetimes for one retirement age.

        Returns (success_probability, median_min_portfolio,
                 median_initial_withdrawal_rate, median_final_portfolio).

        Every dollar figure is a MEDIAN, not a mean. Over a 40-year horizon at ~19%
        volatility the distribution of outcomes is extremely right-skewed: a handful
        of paths compound to enormous values and drag the mean far above any outcome
        you are likely to see. The low-water mark is worse still, because failed runs
        pile up at exactly $0 -- at a 47% success rate the MEAN low point reads $871k
        while the median is $0, and only the median describes the typical outcome.
        """
        runs = self.cfg.simulation.monte_carlo_runs
        seeds = self._run_seeds(retirement_age, runs)

        with Pool(processes=min(cpu_count(), max(1, runs)), initializer=init_worker,
                  initargs=(self.cfg,)) as pool:
            results = list(tqdm(
                pool.imap(simulate_worker, [(retirement_age, s) for s in seeds],
                          chunksize=32),
                total=runs, desc=f"Age {retirement_age}"))

        successes = sum(1 for r in results if r[0])
        mins = [r[1] for r in results]
        withdrawal_rates = [r[2] for r in results if r[2] is not None]
        finals = [r[3] for r in results]
        return (successes / runs,
                float(np.median(mins)),
                float(np.median(withdrawal_rates)) if withdrawal_rates else 0.0,
                float(np.median(finals)))

    def compute_probability_curve(self, progress=None):
        """Sweep every candidate retirement age.

        progress, if given, is called as progress(index, total, age) before each age
        so a front end can drive a status line or progress bar.
        """
        ages = range(self.cfg.simulation.min_retirement_age,
                     self.cfg.simulation.max_retirement_age + 1)
        total = len(ages)
        for i, age in enumerate(ages):
            if progress:
                progress(i, total, age)
            self.probability_results[age] = self.retirement_probability(age)

    def compute_trajectory_percentiles(self, retirement_age: int,
                                       n_samples: int = TRAJECTORY_SAMPLES):
        """Portfolio value percentiles for each year of retirement.

        Column i holds the portfolio at age retirement_age + i. A run contributes a
        value only while someone is still alive, so later columns rest on
        progressively fewer samples; the returned "n" entry reports how many runs
        backed each column. With common random numbers on and n_samples equal to
        monte_carlo_runs, these are the very same lifetimes behind the success rate.
        """
        # A life ending at death_age records one value per year from retirement
        # through its last full year, i.e. exactly (death_age - retirement_age)
        # values. So the widest possible trajectory is max_years columns, not
        # max_years + 1 -- an extra column could never be reached by any run and
        # would read as a drop to zero at the right-hand edge of the chart.
        horizon = max(self.cfg.life_events.death_age_max,
                      self.cfg.spouse.death_age_max - self.cfg.spouse.age_offset
                      if self.cfg.spouse.enabled else 0)
        n_cols = max(int(horizon - retirement_age), 0)
        all_traj = np.full((n_samples, n_cols), np.nan)
        seeds = self._run_seeds(retirement_age, n_samples)

        with Pool(processes=min(cpu_count(), max(1, n_samples)), initializer=init_worker,
                  initargs=(self.cfg,)) as pool:
            results = list(tqdm(
                pool.imap(simulate_trajectory_worker,
                          [(retirement_age, s) for s in seeds], chunksize=64),
                total=n_samples, desc=f"Trajectory {retirement_age}"))

        # Each trajectory is exactly as long as the retirement that life lived --
        # simulate_life zero-fills a run that went broke through to its own death,
        # so "broke" and "still solvent" leave the sample at the same point. Columns
        # past a run's death stay NaN and drop out of that column's percentile.
        for i, (_survived, traj) in enumerate(results):
            n = min(len(traj), n_cols)
            all_traj[i, :n] = traj[:n]

        # Percentile only the columns that actually have samples; an all-NaN column
        # would otherwise warn and return NaN.
        counts = np.count_nonzero(~np.isnan(all_traj), axis=0)
        filled = counts > 0
        ages = np.arange(retirement_age, retirement_age + n_cols)

        percentiles = {}
        for p in (1, 10, 25, 50):
            row = np.full(n_cols, np.nan)
            if filled.any():
                row[filled] = np.nanpercentile(all_traj[:, filled], p, axis=0)
            percentiles[p] = row
        percentiles["n"] = counts
        return ages, percentiles

    # ==============================
    # RESULTS
    # ==============================
    def find_retirement_age(self) -> Optional[RetirementResult]:
        """Earliest swept age clearing the target. Pure lookup -- see format_results_table."""
        for age in sorted(self.probability_results):
            prob = self.probability_results[age][0]
            if prob >= self.cfg.simulation.target_success_probability:
                return RetirementResult(age, prob)
        return None

    def format_results_table(self) -> str:
        """Render the swept results. Shared by the CLI stdout and the GUI textbox."""
        target = self.cfg.simulation.target_success_probability
        result = self.find_retirement_age()

        lines = []
        if result:
            lines.append(f"  Earliest retirement age meeting {target:.0%} target: "
                         f"AGE {result.retirement_age}  ({result.success_probability:.1%} success)")
        else:
            lines.append(f"  No retirement age met the {target:.0%} success target in the range tested.")
        lines.append("")
        lines.append("  Age   Success   Median Min In Ret   Median Initial W/D   Median Final Bal")
        lines.append("  " + "-" * 80)
        for age in sorted(self.probability_results):
            prob, med_min, med_wr, median_final = self.probability_results[age]
            lines.append(f"  {age:>4}   {prob:>6.1%}   ${med_min:>15,.0f}   "
                         f"{med_wr:>17.2%}   ${median_final:>15,.0f}")
        return "\n".join(lines)

    # ==============================
    # PLOTTING (shared by CLI and GUI)
    # ==============================
    def draw_probability_curve(self, ax):
        """Success probability vs retirement age onto `ax`, with the target line
        and the earliest passing age marked. Shared by the CLI and the GUI."""
        ages = sorted(self.probability_results)
        probs = [self.probability_results[a][0] for a in ages]
        target = self.cfg.simulation.target_success_probability
        result = self.find_retirement_age()

        ax.plot(ages, probs, "b-o", markersize=4, linewidth=2)
        ax.axhline(target, color="r", linestyle="--", alpha=0.7, label=f"Target ({target:.0%})")
        if result:
            ax.axvline(result.retirement_age, color="green", linestyle=":",
                       alpha=0.7, label=f"Earliest: age {result.retirement_age}")
        ax.set_xlabel("Retirement Age")
        ax.set_ylabel("Success Probability")
        ax.set_title("Retirement Success Probability by Age")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    def draw_trajectory(self, ax, traj_ages, pcts, retirement_age, n_samples):
        """The portfolio fan chart onto `ax`: median/25th/10th/1st percentile of
        portfolio value per year of retirement, from compute_trajectory_percentiles.

        Later ages rest on progressively fewer still-alive runs, and columns backed
        by fewer than MIN_TRAJECTORY_SAMPLES are dropped rather than drawn: with a
        handful of survivors every percentile lands on the same one or two lives,
        which paints a dramatic late-life spike out of nothing. The title reports
        the age the chart is trimmed to.
        """
        counts = np.asarray(pcts["n"])
        plottable = counts >= MIN_TRAJECTORY_SAMPLES

        def series(p):
            """Values in $M, blanked wherever the column is too thin to trust."""
            return np.where(plottable, np.asarray(pcts[p], dtype=float) / 1e6, np.nan)

        m, p25, p10, p1 = (series(p) for p in (50, 25, 10, 1))
        ax.plot(traj_ages, m, color="steelblue", linewidth=2, label="Median (50th)")
        ax.plot(traj_ages, p25, color="orange", linewidth=1.5, linestyle="--", label="25th percentile")
        ax.plot(traj_ages, p10, color="tomato", linewidth=1.5, linestyle="--", label="10th percentile")
        ax.plot(traj_ages, p1, color="darkred", linewidth=1.5, linestyle=":", label="1st percentile")
        ax.fill_between(traj_ages, p10, m, alpha=0.08, color="steelblue")
        ax.fill_between(traj_ages, p1, p10, alpha=0.08, color="tomato")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(retirement_age, color="green", linestyle=":", alpha=0.5)
        # The title has to survive the CLI's 9-inch figure, so the sampling caveat
        # lives on the x-axis -- it is a statement about the age axis anyway, and a
        # one-line title of this length overflowed and was clipped.
        last = int(traj_ages[plottable][-1]) if plottable.any() else retirement_age
        ax.set_xlabel(f"Age  (shown to {last}, while at least "
                      f"{MIN_TRAJECTORY_SAMPLES} runs are still alive)")
        ax.set_ylabel("Portfolio Value ($M)")
        ax.set_title(f"Portfolio Value in Retirement — Retiring at {retirement_age} "
                     f"({n_samples:,} simulations)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_results(self, trajectory_samples=TRAJECTORY_SAMPLES):
        """CLI figure: probability curve, plus the trajectory fan when a target age exists."""
        import matplotlib.pyplot as plt      # see the note at the top of the module

        result = self.find_retirement_age()
        n_plots = 2 if result else 1

        # Constrained layout re-solves on every draw, so the labels survive the user
        # resizing the window; tight_layout would only be correct at the size below.
        fig = plt.figure(figsize=(9, 3.5 * n_plots), layout="constrained")
        self.draw_probability_curve(fig.add_subplot(n_plots, 1, 1))

        if result:
            ages, pcts = self.compute_trajectory_percentiles(
                result.retirement_age, n_samples=trajectory_samples)
            self.draw_trajectory(fig.add_subplot(n_plots, 1, 2), ages, pcts,
                                 result.retirement_age, trajectory_samples)

        plt.show()

    def assumption_report(self) -> str:
        """What the model will ACTUALLY deliver, versus what the YAML asks for.

        Several configured numbers are modified by the model's own mechanics (crisis
        regimes changing the return and volatility, the PIA formula rescaling Social
        Security, RMD timing). This prints the effective values so the gap is never
        a surprise.
        """
        cfg = self.cfg
        m, sim, le = cfg.market, cfg.simulation, cfg.life_events
        f = self.crisis_fraction
        real_stock = self.real_return(m.stock_return)
        eff_log = (self.regimes['normal']['return_boost'] * (1 - f)
                   + self.regimes['crisis']['return_boost'] * f) * 12 + math.log1p(real_stock)
        # Total variance of the monthly log return = the within-regime variance
        # MIXTURE plus the variance of the regime MEANS themselves. Dropping that
        # second term (the regimes differ in drift, not just in spread) understates
        # the delivered volatility, so both halves are counted here.
        within = ((1 - f) * m.stock_volatility ** 2
                  + f * (m.stock_volatility * sim.crisis_regime.volatility_multiplier) ** 2)
        between = (f * (1 - f)
                   * (self.regimes['normal']['return_boost']
                      - self.regimes['crisis']['return_boost']) ** 2 * 12)
        eff_vol = math.sqrt(within + between)
        p_enter = sim.normal_regime.monthly_crisis_probability
        p_exit = sim.crisis_regime.monthly_recovery_probability

        out = ["  EFFECTIVE ASSUMPTIONS (what the simulation actually uses)",
               "  " + "-" * 68,
               f"  stock return   {m.stock_return:>7.2%} nominal -> {real_stock:>7.3%} real "
               f"-> {math.expm1(eff_log):>7.3%} after regimes",
               f"  drag compensation {'ON -- configured return is delivered' if sim.compensate_crisis_drag else 'OFF -- crises reduce the return below what you set'}",
               f"  stock volatility {m.stock_volatility:>6.1%} configured -> {eff_vol:>6.1%} effective (crises add variance)",
               # Either probability may legitimately be 0 -- that is how a user turns
               # the crisis model off -- so neither reciprocal can be taken blind.
               ("  crisis regime  disabled (no crises are generated)" if f == 0 else
                f"  crisis regime  every {1 / p_enter / 12:>4.1f} yrs, lasting "
                f"{(1 / p_exit) if p_exit > 0 else float('inf'):>4.1f} months, "
                f"{f:>5.1%} of all months"),
               f"  bond return    {m.bond_return:>7.2%} nominal -> {self.real_return(m.bond_return):>7.3%} real",
               f"  cash return    {m.cash_return:>7.2%} nominal -> {self.real_return(m.cash_return):>7.3%} real",
               f"  inflation      {m.inflation:>7.2%} mean, {m.inflation_volatility:>5.2%} volatility",
               f"  stock/bond correlation {m.stock_bond_correlation:>+6.2f} (exact in both regimes)",
               f"  taxable-account income taxed yearly: {m.stock_dividend_yield:.2%} dividends "
               f"(LTCG rates), {m.bond_taxable_yield:.2%} bond interest (ordinary)",
               f"  filing status  {cfg.taxes.filing_status}"
               + (" -> single after the first death" if cfg.spouse.enabled else ""),
               f"  mortality      {le.mortality_model} ({le.mortality_sex}), capped at {le.death_age_max}",
               f"  RMDs           from age {sim.rmd_start_age} on the prior year-end balance"]

        for age in (sim.min_retirement_age, sim.max_retirement_age):
            years_after = age - sim.current_age
            worked = le.ss_earnings_years_at_current_age + years_after
            credits = min(40, le.ss_credits_at_current_age + 4 * years_after)
            note = (f"({min(worked, le.ss_benefit_years_required)} covered years of "
                    f"{le.ss_benefit_years_required}, PIA formula)"
                    if credits >= le.ss_retirement_eligibility_credits
                    else f"(NOT ELIGIBLE -- only {credits} of "
                         f"{le.ss_retirement_eligibility_credits} credits)")
            out.append(f"  Social Security if you retire at {age}: "
                       f"${self.primary_ss_income(age):>8,.0f}/yr {note}")

        hc = cfg.healthcare
        out.append(f"  healthcare     ${hc.pre_medicare_annual_premium:,.0f}/yr per person before "
                   f"{hc.medicare_age}, ${hc.medicare_annual_premium:,.0f}/yr after (ON TOP of expenses)")
        if cfg.taxes.use_72t_sepp:
            pen = "disabled (72(t)/Roth ladder assumed)"
        elif cfg.taxes.assume_qualified_plan_age55_exception:
            pen = (f"{cfg.taxes.early_withdrawal_penalty:.0%} before age 55 "
                   f"(rule-of-55 assumed for employer-plan money)")
        else:
            pen = f"{cfg.taxes.early_withdrawal_penalty:.0%} before age {cfg.taxes.penalty_free_age}"
        out.append(f"  early withdrawal penalty: {pen}")
        out.append(f"  spouse         {'enabled' if cfg.spouse.enabled else 'not modelled (single person)'}")
        out.append(f"  randomness     seed {sim.random_seed}"
                   f"{' (fresh entropy)' if sim.random_seed < 0 else ''}, "
                   f"common random numbers {'ON -- ages share scenarios' if sim.common_random_numbers else 'OFF'}")
        return "\n".join(out)


# ==============================
# MULTIPROCESSING WORKERS
# ==============================
_worker_simulator = None


def init_worker(config):
    """Runs once per pool worker process at startup.

    Each simulated life builds its own np.random.default_rng from an explicit
    per-run seed, so workers never touch the global numpy RNG state. (Historical
    note: an earlier version drew from the global RNG, and forked workers inheriting
    the parent's state silently returned cpu_count copies of the same few lives --
    per-run seeding is what makes that failure impossible AND makes runs
    reproducible.)
    """
    global _worker_simulator
    _worker_simulator = RetirementSimulator(config)


def simulate_worker(args):
    """One pool task: simulate one seeded lifetime, drop the (None) trajectory."""
    retirement_age, seed = args
    survived, min_portfolio, iwr, total, _ = _worker_simulator.simulate_life(
        retirement_age, random_seed=seed)
    return survived, min_portfolio, iwr, total


def simulate_trajectory_worker(args):
    """One pool task for the fan chart: same lifetime, but keep the yearly totals."""
    retirement_age, seed = args
    survived, _, _, _, trajectory = _worker_simulator.simulate_life(
        retirement_age, random_seed=seed, record_trajectory=True)
    return survived, trajectory


# ==============================
# CONFIG SCHEMA (single source of truth for CLI and GUI)
# ==============================
def config_from_dict(raw: dict) -> Config:
    """Build a Config from a parsed YAML mapping.

    Every section is expanded with ** so a missing or misspelled key raises a
    TypeError naming the field, rather than being silently dropped.
    """
    sim = dict(raw["simulation"])
    normal = NormalRegime(**sim.pop("normal_regime"))
    crisis = CrisisRegime(**sim.pop("crisis_regime"))
    return Config(
        accounts=Accounts(**raw["accounts"]),
        contributions=Contributions(**raw["contributions"]),
        life_events=LifeEvents(**raw["life_events"]),
        spouse=Spouse(**raw["spouse"]),
        market=Market(**raw["market"]),
        taxes=Taxes(**raw["taxes"]),
        healthcare=Healthcare(**raw["healthcare"]),
        simulation=Simulation(normal_regime=normal, crisis_regime=crisis, **sim),
        spending=Spending(**raw["spending"]),
    )


def config_to_dict(config: Config) -> dict:
    """Inverse of config_from_dict. asdict recurses into the nested regime dataclasses."""
    return {section.name: asdict(getattr(config, section.name))
            for section in fields(config)}


VALID_FILING_STATUSES = ("single", "married_filing_jointly",
                         "qualifying_surviving_spouse", "head_of_household")
VALID_MORTALITY_MODELS = ("ssa_inspired", "normal")
VALID_MORTALITY_SEXES = ("male", "female")


def validate_config(config: Config) -> None:
    """Catch impossible or inconsistent inputs BEFORE hours of simulation.

    Raises ValidationError listing every problem found. Called by load_config and
    save_config, and by the GUI before a run, so a bad value fails loudly at the
    door instead of producing a quietly wrong retirement age.
    """
    c = config
    errors = []

    # -- accounts --
    for name in ("roth", "traditional", "brokerage", "cash"):
        if getattr(c.accounts, name) < 0:
            errors.append(f"accounts.{name} must be non-negative")
    if not 0 <= c.accounts.brokerage_cost_basis <= c.accounts.brokerage + 1e-9:
        errors.append("brokerage_cost_basis must be between 0 and the brokerage balance")

    # -- contributions --
    for name in ("annual_roth", "annual_traditional", "annual_brokerage", "annual_cash"):
        if getattr(c.contributions, name) < 0:
            errors.append(f"contributions.{name} must be non-negative")
    if c.contributions.annual_contribution_growth_rate <= -1:
        errors.append("annual_contribution_growth_rate must be > -1")

    # -- simulation sweep --
    if not 0 < c.simulation.target_success_probability <= 1:
        errors.append("target_success_probability must be in (0, 1]")
    if c.simulation.monte_carlo_runs < 1:
        errors.append("monte_carlo_runs must be at least 1")
    if not 0 < c.simulation.current_age < 120:
        errors.append("current_age must be between 0 and 120")
    if c.simulation.min_retirement_age < c.simulation.current_age:
        errors.append("min_retirement_age must be >= current_age")
    if c.simulation.max_retirement_age < c.simulation.min_retirement_age:
        errors.append("max_retirement_age must be >= min_retirement_age")
    for name in ("glide_path_start_stock_pct", "glide_path_end_stock_pct",
                 "static_stock_allocation"):
        if not 0 <= getattr(c.simulation, name) <= 1:
            errors.append(f"{name} must be between 0 and 1")
    if c.simulation.glide_path_years < 1:
        errors.append("glide_path_years must be at least 1")
    if c.simulation.return_distribution_degrees_of_freedom <= 2:
        errors.append("return_distribution_degrees_of_freedom must be > 2 "
                      "(a Student-t needs df > 2 to have finite variance)")
    if not 72 <= c.simulation.rmd_start_age <= 100:
        errors.append("rmd_start_age must be between 72 and 100 "
                      "(73 if born 1951-1959, 75 if born 1960+)")
    for regime, prob in (("normal_regime.monthly_crisis_probability",
                          c.simulation.normal_regime.monthly_crisis_probability),
                         ("crisis_regime.monthly_recovery_probability",
                          c.simulation.crisis_regime.monthly_recovery_probability)):
        if not 0 <= prob <= 1:
            errors.append(f"{regime} must be between 0 and 1")
    # Both go through math.log1p when the regime drifts are built.
    if c.simulation.crisis_regime.annual_return_drag <= -1:
        errors.append("crisis_regime.annual_return_drag must be greater than -1")
    if c.simulation.normal_regime.return_boost <= -1:
        errors.append("normal_regime.return_boost must be greater than -1")
    if c.simulation.crisis_regime.volatility_multiplier < 0:
        errors.append("crisis_regime.volatility_multiplier must be non-negative")
    if c.simulation.normal_regime.volatility_multiplier < 0:
        errors.append("normal_regime.volatility_multiplier must be non-negative")

    # -- market --
    if not -1 <= c.market.stock_bond_correlation <= 1:
        errors.append("stock_bond_correlation must be between -1 and 1")
    for name in ("stock_volatility", "bond_volatility", "inflation_volatility"):
        if getattr(c.market, name) < 0:
            errors.append(f"market.{name} must be non-negative")
    # A rate of exactly -1 wipes out the whole asset in a year. real_return divides
    # by (1 + inflation) and the risky legs take log1p of the real return, so -1 or
    # below is a ZeroDivisionError or a domain error deep in __init__ rather than a
    # readable message here.
    for name in ("stock_return", "bond_return", "cash_return", "inflation"):
        if getattr(c.market, name) <= -1:
            errors.append(f"market.{name} must be greater than -1")
    # Inflation shocks are subtracted from the cash rate arithmetically, so a large
    # enough sigma can push a month below -100% and drive the balance negative.
    if c.market.inflation_volatility > 0.25:
        errors.append("market.inflation_volatility above 0.25 is not modelled "
                      "(monthly shocks can exceed -100% on cash)")
    for name in ("stock_dividend_yield", "bond_taxable_yield"):
        if not 0 <= getattr(c.market, name) <= 0.2:
            errors.append(f"market.{name} must be between 0 and 0.2")

    # -- mortality and Social Security, both persons --
    for label, person in (("life_events", c.life_events), ("spouse", c.spouse)):
        if label == "spouse" and not c.spouse.enabled:
            continue
        if person.mortality_model not in VALID_MORTALITY_MODELS:
            errors.append(f"{label}.mortality_model must be one of {VALID_MORTALITY_MODELS}")
        if person.mortality_sex not in VALID_MORTALITY_SEXES:
            errors.append(f"{label}.mortality_sex must be one of {VALID_MORTALITY_SEXES}")
        if person.death_age_min > person.death_age_max:
            errors.append(f"{label}.death_age_min must be <= death_age_max")
        # death_age_max also sizes the trajectory chart, and _actuarial_death_age
        # walks the hazard curve a year at a time from today's age up to it.
        if not c.simulation.current_age < person.death_age_max <= 120:
            errors.append(f"{label}.death_age_max must be above current_age and at most 120")
        if not 62 <= person.ss_claim_age <= 70:
            errors.append(f"{label}.ss_claim_age must be between 62 and 70")
        if not 0 <= person.ss_credits_at_current_age <= 40:
            errors.append(f"{label}.ss_credits_at_current_age must be 0..40")
        if person.ss_earnings_years_at_current_age < 0:
            errors.append(f"{label}.ss_earnings_years_at_current_age must be non-negative")
        if person.ss_annual_full_retirement_benefit < 0:
            errors.append(f"{label}.ss_annual_full_retirement_benefit must be non-negative")
        if person.mortality_model == "normal" and person.death_age_std < 0:
            errors.append(f"{label}.death_age_std must be non-negative")
    if c.spouse.enabled:
        if not -50 < c.spouse.age_offset < 50:
            errors.append("spouse.age_offset must be within 50 years")
        if not 0 <= c.spouse.survivor_spending_factor <= 1:
            errors.append("spouse.survivor_spending_factor must be between 0 and 1 "
                          "(a survivor cannot spend more than the couple did)")
    if c.life_events.ss_benefit_years_required < 1:
        errors.append("ss_benefit_years_required must be at least 1")
    if not 1 <= c.life_events.ss_retirement_eligibility_credits <= 40:
        errors.append("ss_retirement_eligibility_credits must be 1..40 (current law: 40)")

    # -- taxes: the filing status must exist in every per-status table, and
    #    'single' must too, because a surviving spouse files single --
    if c.taxes.filing_status not in VALID_FILING_STATUSES:
        errors.append(f"filing_status must be one of {VALID_FILING_STATUSES}")
    for table_name in ("standard_deductions", "additional_standard_deductions_65plus",
                       "federal_brackets", "ltcg_brackets",
                       "ss_provisional_thresholds", "niit_thresholds"):
        table = getattr(c.taxes, table_name)
        if not isinstance(table, dict):
            errors.append(f"taxes.{table_name} must map filing status -> values")
            continue
        for status in {c.taxes.filing_status, "single"}:
            if status not in table:
                errors.append(f"taxes.{table_name} is missing an entry for '{status}'")
    # _bracket_tax walks the pairs in order and assumes each upper bound is above
    # the last. Out-of-order or negative bounds do not raise -- they silently return
    # a WRONG tax (a shuffled 2026 single table turns $7,912 into $16,520), which is
    # exactly the class of error a config check has to catch.
    for table_name in ("federal_brackets", "ltcg_brackets"):
        for status, brackets in (getattr(c.taxes, table_name) or {}).items():
            where = f"taxes.{table_name}[{status}]"
            if not brackets or brackets[-1][0] is not None:
                errors.append(f"{where} must end with a null-bounded top bracket")
                continue
            bounds = [b[0] for b in brackets[:-1]]
            if any(b is None for b in bounds):
                errors.append(f"{where} may only have a null upper bound on the LAST bracket")
            elif any(b <= 0 for b in bounds):
                errors.append(f"{where} upper bounds must be positive")
            elif any(a >= b for a, b in zip(bounds, bounds[1:])):
                errors.append(f"{where} upper bounds must increase (lowest bracket first)")
            if any(not 0 <= rate <= 1 for _, rate in brackets):
                errors.append(f"{where} rates must be between 0 and 1")
    for status, amount in (c.taxes.additional_standard_deductions_65plus or {}).items():
        if amount < 0:
            errors.append(f"taxes.additional_standard_deductions_65plus[{status}] "
                          "must be non-negative")
    for status, amount in (c.taxes.standard_deductions or {}).items():
        if amount < 0:
            errors.append(f"taxes.standard_deductions[{status}] must be non-negative")
    for status, thresholds in (c.taxes.ss_provisional_thresholds or {}).items():
        if len(thresholds) != 2:
            errors.append(f"taxes.ss_provisional_thresholds[{status}] must be [lower, upper]")
        elif not 0 <= thresholds[0] < thresholds[1]:
            errors.append(f"taxes.ss_provisional_thresholds[{status}] must be "
                          "non-negative and increasing")
    for status, threshold in (c.taxes.niit_thresholds or {}).items():
        if threshold < 0:
            errors.append(f"taxes.niit_thresholds[{status}] must be non-negative")
    if not 0 <= c.taxes.early_withdrawal_penalty <= 1:
        errors.append("early_withdrawal_penalty must be between 0 and 1")
    if not 0 <= c.taxes.penalty_free_age <= 120:
        errors.append("penalty_free_age must be a plausible age (current law: 59.5)")
    if not 0 <= c.taxes.ss_max_taxable_fraction <= 1:
        errors.append("ss_max_taxable_fraction must be between 0 and 1")
    if not 0 <= c.taxes.niit_rate <= 0.2:
        errors.append("niit_rate must be between 0 and 0.2")
    if not 0 <= c.taxes.state_tax_rate <= 0.2:
        errors.append("state_tax_rate must be between 0 and 0.2")

    # -- spending --
    if c.spending.initial_annual_expenses < 0:
        errors.append("initial_annual_expenses must be non-negative")
    # The smile compounds this rate over the whole decline window, so a value above
    # 1 does not "decline" at all -- 5.0 multiplies spending by 9.7 MILLION by 85.
    if not 0 < c.spending.annual_spending_decline_rate <= 1:
        errors.append("annual_spending_decline_rate must be in (0, 1] "
                      "(0.99 = spend 1% less each year; 1.0 = flat)")
    if c.spending.annual_healthcare_increase_rate <= -1:
        errors.append("annual_healthcare_increase_rate must be > -1")
    for name in ("spending_decline_start_age", "spending_decline_end_age"):
        if not 0 <= getattr(c.spending, name) <= 120:
            errors.append(f"spending.{name} must be a plausible age")
    if c.spending.spending_decline_start_age > c.spending.spending_decline_end_age:
        errors.append("spending_decline_start_age must be <= spending_decline_end_age")
    # The raise branch is an elif on the cut branch, so an inverted pair silently
    # makes raises unreachable rather than erroring.
    if (c.spending.guardrail_cut_return_threshold
            >= c.spending.guardrail_raise_return_threshold):
        errors.append("guardrail_cut_return_threshold must be below "
                      "guardrail_raise_return_threshold")
    if not 0 < c.spending.guardrail_cut_amount <= 1:
        errors.append("guardrail_cut_amount must be in (0, 1]")
    if not 0 < c.spending.guardrail_cut_floor <= 1:
        errors.append("guardrail_cut_floor must be in (0, 1]")
    if c.spending.guardrail_raise_amount < 1:
        errors.append("guardrail_raise_amount must be >= 1")
    if c.spending.guardrail_raise_ceiling < 1:
        errors.append("guardrail_raise_ceiling must be >= 1")

    # -- healthcare --
    for name in ("pre_medicare_annual_premium", "medicare_annual_premium"):
        if getattr(c.healthcare, name) < 0:
            errors.append(f"healthcare.{name} must be non-negative")
    if not 50 <= c.healthcare.medicare_age <= 100:
        errors.append("healthcare.medicare_age must be a plausible age (Medicare is 65)")

    if errors:
        raise ValidationError("; ".join(errors))


def load_config(path: str) -> Config:
    """Parse a YAML parameter file into a validated Config.

    Raises TypeError naming any missing/misspelled key, and ValidationError
    listing every out-of-range or inconsistent value.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = config_from_dict(raw)
    validate_config(config)
    return config


def save_config(config: Config, path: str) -> None:
    """Write a Config back to YAML (validated first, so a bad config is never
    persisted). Comments in a hand-written YAML are NOT preserved."""
    validate_config(config)
    with open(path, "w") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)


def _resolve_field(config: Config, path: str):
    """'simulation.normal_regime.monthly_crisis_probability' -> (owner_dataclass, name)."""
    obj = config
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    if not is_dataclass(obj) or not hasattr(obj, parts[-1]):
        raise AttributeError(f"no such config field: {path}")
    return obj, parts[-1]


def get_field(config: Config, path: str):
    """Read a config value by dotted path, e.g. get_field(cfg, "accounts.roth")."""
    owner, name = _resolve_field(config, path)
    return getattr(owner, name)


def set_field(config: Config, path: str, value) -> None:
    """Set a field, coercing to the type declared on the dataclass.

    Strings pass through untouched (filing_status, mortality_model, ...) and are
    caught by validate_config if invalid. NOTE: bool coercion follows Python
    truthiness -- pass real booleans, not the string "False" (the GUI's checkboxes
    do; there are no boolean text entries).
    """
    owner, name = _resolve_field(config, path)
    declared = next(f.type for f in fields(owner) if f.name == name)
    if declared in (int, "int"):
        value = int(float(value))
    elif declared in (float, "float"):
        value = float(value)
    elif declared in (bool, "bool"):
        value = bool(value)
    setattr(owner, name, value)


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "simulation_params.yaml")


def default_config() -> Config:
    """The shipped defaults, so the GUI and CLI start from the same numbers."""
    return load_config(DEFAULT_CONFIG_PATH)


# ==============================
# CLI ENTRY
# ==============================
if __name__ == "__main__":
    config = load_config(DEFAULT_CONFIG_PATH)
    sim = RetirementSimulator(config)
    print(sim.assumption_report())
    print()
    sim.compute_probability_curve()
    print()
    print(sim.format_results_table())
    print("\nEarliest retirement age:", sim.find_retirement_age())
    sim.plot_results()
