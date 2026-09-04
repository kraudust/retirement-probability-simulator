"""Shared fixtures and scenario builders for the test suite.

Hermeticity: every test starts from tests/baseline_config.yaml, a FROZEN copy of
the parameter file. Tests must never load simulation_params.yaml -- the user edits
that file freely, and their personal numbers must not be able to break the suite.
The hand-computed expected values in test_tax_engine.py correspond to the 2026 tax
tables in the frozen baseline; if those tables are ever updated, update the
expectations together.

Determinism: everything runs on fixed seeds (and the deterministic scenarios have
zero volatility), so every asserted number is exactly repeatable. Simulations run
SERIALLY through simulate_life -- the multiprocessing pool is exercised once, in
test_simulation.py, not relied on everywhere.
"""

import copy
import os
import sys

import pytest

# Make the engine importable when pytest is run from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retirement_age_calculator import RetirementSimulator, load_config  # noqa: E402

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "baseline_config.yaml")


@pytest.fixture()
def base_cfg():
    """A fresh, validated Config from the frozen baseline. Mutate freely."""
    return load_config(BASELINE_PATH)


def deterministic_cfg(base, real_return=0.02, spend=50_000.0, roth=1_000_000.0,
                      retire_age=65, death_age=95):
    """A scenario with NO randomness, so outcomes are computable by hand.

    - all volatility zero, crises impossible: every month's growth is exactly the
      configured real return, regardless of seed
    - everything in Roth, no dividends/interest: every tax is exactly zero
    - fixed death age, constant real spending, no SS/healthcare/smile/guardrails

    The engine's whole monthly loop then reduces to the recurrence checked in
    test_simulation.py: each plan year, R <- R*g - S, then eleven more months of
    growth -- so any change to the growth/withdrawal ordering shows up as a
    dollar-level mismatch.
    """
    cfg = copy.deepcopy(base)
    cfg.accounts.roth = roth
    cfg.accounts.traditional = 0.0
    cfg.accounts.brokerage = 0.0
    cfg.accounts.cash = 0.0
    cfg.accounts.brokerage_cost_basis = 0.0
    for f in ("annual_roth", "annual_traditional", "annual_brokerage", "annual_cash"):
        setattr(cfg.contributions, f, 0.0)

    cfg.simulation.current_age = retire_age
    cfg.simulation.min_retirement_age = retire_age
    cfg.simulation.max_retirement_age = retire_age
    cfg.simulation.glide_path = False
    cfg.simulation.static_stock_allocation = 1.0
    cfg.simulation.random_seed = 1
    # crises impossible -> the regime never leaves 'normal'
    cfg.simulation.normal_regime.monthly_crisis_probability = 0.0
    cfg.simulation.normal_regime.return_boost = 0.0

    # a nominal return that is EXACTLY `real_return` after 3% inflation
    cfg.market.inflation = 0.03
    cfg.market.stock_return = (1 + real_return) * 1.03 - 1
    cfg.market.bond_return = cfg.market.stock_return
    cfg.market.stock_volatility = 0.0
    cfg.market.bond_volatility = 0.0
    cfg.market.inflation_volatility = 0.0
    cfg.market.stock_dividend_yield = 0.0
    cfg.market.bond_taxable_yield = 0.0

    # death at an exact age: a "normal" draw with zero spread
    cfg.life_events.mortality_model = "normal"
    cfg.life_events.death_age_mean = death_age
    cfg.life_events.death_age_std = 0.0
    cfg.life_events.death_age_min = death_age
    cfg.life_events.death_age_max = death_age
    cfg.life_events.ss_annual_full_retirement_benefit = 0.0
    cfg.spouse.enabled = False

    cfg.healthcare.pre_medicare_annual_premium = 0.0
    cfg.healthcare.medicare_annual_premium = 0.0

    # constant real spending: smile and guardrails neutralised
    cfg.spending.initial_annual_expenses = spend
    cfg.spending.spending_decline_start_age = 119
    cfg.spending.spending_decline_end_age = 120
    cfg.spending.annual_healthcare_increase_rate = 0.0
    cfg.spending.guardrail_cut_return_threshold = -0.99
    cfg.spending.guardrail_raise_return_threshold = 0.99
    cfg.spending.guardrail_cut_amount = 1.0
    cfg.spending.guardrail_raise_amount = 1.0
    return cfg


def trinity_cfg(base, spend, years=30, stock_pct=0.5):
    """A Trinity-study-shaped benchmark: $1M, fixed horizon, constant real
    spending, tax-free (all Roth), no SS/healthcare/smile/guardrails, static
    allocation, Trinity-like capital markets (~7.1% real stocks, ~2.4% real bonds,
    the engine's own fat tails and crisis regimes supplying the risk).
    """
    cfg = copy.deepcopy(base)
    cfg.accounts.roth = 1_000_000.0
    cfg.accounts.traditional = 0.0
    cfg.accounts.brokerage = 0.0
    cfg.accounts.cash = 0.0
    cfg.accounts.brokerage_cost_basis = 0.0
    for f in ("annual_roth", "annual_traditional", "annual_brokerage", "annual_cash"):
        setattr(cfg.contributions, f, 0.0)

    cfg.simulation.current_age = 65
    cfg.simulation.min_retirement_age = 65
    cfg.simulation.max_retirement_age = 65
    cfg.simulation.glide_path = False
    cfg.simulation.static_stock_allocation = stock_pct
    cfg.simulation.random_seed = 12345          # pinned: benchmark numbers are exact
    cfg.simulation.common_random_numbers = True

    cfg.life_events.mortality_model = "normal"  # fixed horizon, like the studies
    cfg.life_events.death_age_mean = 65 + years
    cfg.life_events.death_age_std = 0.0
    cfg.life_events.death_age_min = 65 + years
    cfg.life_events.death_age_max = 65 + years
    cfg.life_events.ss_annual_full_retirement_benefit = 0.0
    cfg.spouse.enabled = False

    cfg.healthcare.pre_medicare_annual_premium = 0.0
    cfg.healthcare.medicare_annual_premium = 0.0

    cfg.market.stock_return = 0.103   # ~7.09% real at 3% inflation
    cfg.market.bond_return = 0.055    # ~2.43% real
    cfg.market.bond_volatility = 0.08
    cfg.market.stock_bond_correlation = 0.1
    cfg.market.stock_dividend_yield = 0.0
    cfg.market.bond_taxable_yield = 0.0

    cfg.spending.initial_annual_expenses = spend
    cfg.spending.spending_decline_start_age = 119
    cfg.spending.spending_decline_end_age = 120
    cfg.spending.annual_healthcare_increase_rate = 0.0
    cfg.spending.guardrail_cut_return_threshold = -0.99
    cfg.spending.guardrail_raise_return_threshold = 0.99
    cfg.spending.guardrail_cut_amount = 1.0
    cfg.spending.guardrail_raise_amount = 1.0
    return cfg


def serial_success(cfg, n=400):
    """Success rate over n lifetimes, run serially with the engine's own seed
    scheme -- the same lifetimes the multiprocessing pool would run, without the
    pool. Deterministic for a fixed random_seed."""
    sim = RetirementSimulator(cfg)
    age = cfg.simulation.min_retirement_age
    seeds = sim._run_seeds(age, n)
    return sum(1 for s in seeds if sim.simulate_life(age, random_seed=s)[0]) / n
