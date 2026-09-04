"""Tests of the capital-market model: rate conversions, the regime chain, the
Student-t scaling, correlation, and the numbers assumption_report prints.

Why this file exists: every other fixture in the suite deliberately switches the
market model OFF (zero volatility, crises disabled) so that outcomes are
hand-computable. That is the right way to test the accounting, but it left the
market model itself almost entirely unasserted -- a broken t-scaling, a dropped
correlation term or a biased regime chain would not have failed a single test.

These are statistical assertions over large samples on fixed seeds, so they are
deterministic but stated as tolerances rather than exact equalities.
"""

import copy
import math

import numpy as np
import pytest

from retirement_age_calculator import RetirementSimulator


def paths(sim, months, n, seed=0):
    """n independent monthly log-return paths from the engine's own generator."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        stock, bond, cash, _ = sim._market_path(
            months, np.random.default_rng(int(rng.integers(1 << 31))))
        out.append((np.log1p(stock), np.log1p(bond), cash))
    return out


# ---------------------------------------------------------------- rate conversion
def test_real_return_divides_rather_than_subtracts(base_cfg):
    """(1+n)/(1+i)-1, not n-i. At 8% nominal and 3% inflation that is 4.854%, and
    the 5% a subtraction would give is 3% too high after 30 years."""
    cfg = copy.deepcopy(base_cfg)
    cfg.market.inflation = 0.03
    sim = RetirementSimulator(cfg)
    assert sim.real_return(0.08) == pytest.approx(0.08 / 1.03 - 0.03 / 1.03, abs=1e-12)
    assert sim.real_return(0.08) == pytest.approx(0.04854368932, abs=1e-9)
    assert sim.real_return(0.03) == pytest.approx(0.0, abs=1e-15)


def test_monthly_rate_compounds_back_to_the_annual_rate(base_cfg):
    """(1+r)^(1/12)-1, never r/12: twelve of them must compound to exactly r."""
    sim = RetirementSimulator(base_cfg)
    for annual in (0.04, 0.08, -0.02, 0.5):
        monthly = sim.monthly_rate(annual)
        assert (1 + monthly) ** 12 - 1 == pytest.approx(annual, abs=1e-12)
        assert monthly < annual / 12 + 1e-12        # strictly below the naive form


# ---------------------------------------------------------------- the regime chain
def test_crisis_fraction_matches_the_stationary_distribution(base_cfg):
    """p_enter / (p_enter + p_exit), the fraction the drag compensation solves
    against."""
    sim = RetirementSimulator(base_cfg)
    p_enter = base_cfg.simulation.normal_regime.monthly_crisis_probability
    p_exit = base_cfg.simulation.crisis_regime.monthly_recovery_probability
    assert sim.crisis_fraction == pytest.approx(p_enter / (p_enter + p_exit))


def test_drag_compensation_delivers_the_configured_return(base_cfg):
    """compensate_crisis_drag must make the LONG-RUN mean log return equal the
    configured real return -- over a realistic 20-year horizon, not just
    asymptotically. This is the regression guard for the chain's starting state:
    seeding it deterministically in 'normal' gave every life ~4% too much wealth
    because a finite path then spends less time in crisis than the stationary
    fraction the compensation assumes."""
    cfg = copy.deepcopy(base_cfg)
    cfg.market.stock_volatility = 0.0          # isolate the regime chain
    cfg.market.inflation_volatility = 0.0
    sim = RetirementSimulator(cfg)
    target = math.log1p(sim.real_return(cfg.market.stock_return))

    months = 240
    realised = np.array([p[0].sum() / (months / 12) for p in paths(sim, months, 6000, seed=3)])
    assert realised.mean() == pytest.approx(target, abs=0.0025)


def test_crisis_drag_is_not_compensated_when_switched_off(base_cfg):
    """The flag must actually do something: with it off, recurring crises pull the
    delivered return well below the configured one."""
    cfg = copy.deepcopy(base_cfg)
    cfg.market.stock_volatility = 0.0
    cfg.market.inflation_volatility = 0.0
    cfg.simulation.compensate_crisis_drag = False
    sim = RetirementSimulator(cfg)
    target = math.log1p(sim.real_return(cfg.market.stock_return))
    months = 240
    realised = np.array([p[0].sum() / (months / 12) for p in paths(sim, months, 2000, seed=4)])
    assert realised.mean() < target - 0.015


def test_crisis_spells_last_the_configured_length(base_cfg):
    """Mean crisis spell = 1/monthly_recovery_probability months (~18 at 0.055).
    Short spells would understate the multi-year downturns that break retirements."""
    cfg = copy.deepcopy(base_cfg)
    cfg.market.stock_volatility = 0.0        # no shocks, so the log return IS the drift
    cfg.market.inflation_volatility = 0.0
    sim = RetirementSimulator(cfg)
    stock, _, _ = paths(sim, 400_000, 1, seed=5)[0]
    # with the shock removed the series takes exactly two values, one per regime
    in_crisis = np.isclose(stock, stock.min())
    spells, run = [], 0
    for flag in in_crisis:
        if flag:
            run += 1
        elif run:
            spells.append(run)
            run = 0
    expected = 1 / cfg.simulation.crisis_regime.monthly_recovery_probability
    assert np.mean(spells) == pytest.approx(expected, rel=0.06)
    assert in_crisis.mean() == pytest.approx(sim.crisis_fraction, abs=0.01)


# ---------------------------------------------------------------- shock distribution
def test_realised_stock_volatility_matches_the_configured_calm_figure(base_cfg):
    """The Student-t draws are rescaled by sqrt((df-2)/df) for unit variance, so
    `stock_volatility` means exactly what it says when crises are disabled. Drop
    that scaling and a df=6 t would inflate volatility by 22%."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.normal_regime.monthly_crisis_probability = 0.0
    cfg.market.inflation_volatility = 0.0
    sim = RetirementSimulator(cfg)
    stock, _, _ = paths(sim, 500_000, 1, seed=6)[0]
    assert stock.std() * math.sqrt(12) == pytest.approx(cfg.market.stock_volatility, rel=0.02)


def test_fat_tails_are_actually_fat(base_cfg):
    """df=6 must produce visibly more extreme months than a normal would: excess
    kurtosis of a t(6) is 6/(df-4) = 3."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.normal_regime.monthly_crisis_probability = 0.0
    cfg.market.inflation_volatility = 0.0
    sim = RetirementSimulator(cfg)
    stock, _, _ = paths(sim, 500_000, 1, seed=7)[0]
    z = (stock - stock.mean()) / stock.std()
    assert (z ** 4).mean() - 3 > 1.5             # normal would be ~0


def test_stock_bond_correlation_holds_within_each_regime(base_cfg):
    """Bond shocks are built from STANDARDISED draws so the configured correlation
    survives the crisis volatility multiplier. Correlating the already-scaled stock
    shock instead would leak stock volatility into bonds."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.normal_regime.monthly_crisis_probability = 0.0
    cfg.market.inflation_volatility = 0.0
    for corr in (-0.3, 0.0, 0.5):
        cfg.market.stock_bond_correlation = corr
        sim = RetirementSimulator(cfg)
        stock, bond, _ = paths(sim, 400_000, 1, seed=8)[0]
        assert np.corrcoef(stock, bond)[0, 1] == pytest.approx(corr, abs=0.01)
        assert bond.std() * math.sqrt(12) == pytest.approx(cfg.market.bond_volatility,
                                                           rel=0.02)


def test_inflation_shocks_erode_bonds_and_cash_but_not_stocks(base_cfg):
    """Bonds and cash promise a fixed NOMINAL payment, so an inflation surprise
    comes straight off their real return; stocks are treated as inflation-neutral."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.normal_regime.monthly_crisis_probability = 0.0
    cfg.market.bond_volatility = 0.0
    quiet = copy.deepcopy(cfg)
    quiet.market.inflation_volatility = 0.0
    noisy = copy.deepcopy(cfg)
    noisy.market.inflation_volatility = 0.04

    q_stock, q_bond, q_cash = paths(RetirementSimulator(quiet), 200_000, 1, seed=9)[0]
    n_stock, n_bond, n_cash = paths(RetirementSimulator(noisy), 200_000, 1, seed=9)[0]
    assert q_bond.std() == pytest.approx(0.0, abs=1e-12)
    assert n_bond.std() > 0.005                       # inflation now moves bonds
    assert np.std(n_cash) > np.std(q_cash)
    # stocks are untouched by the inflation setting
    assert n_stock.std() == pytest.approx(q_stock.std(), rel=0.02)


# ---------------------------------------------------------------- spending smile
def test_spending_smile_boundaries(base_cfg):
    """Flat before the decline starts, compounding through the window, frozen after
    -- and the healthcare multiplier only starts at the end age."""
    sim = RetirementSimulator(base_cfg)
    s = base_cfg.spending
    assert sim.spending_smile(s.spending_decline_start_age - 5) == (1.0, 1.0)
    assert sim.spending_smile(s.spending_decline_start_age) == (1.0, 1.0)

    mid_base, mid_health = sim.spending_smile(s.spending_decline_start_age + 5)
    assert mid_base == pytest.approx(s.annual_spending_decline_rate ** 5)
    assert mid_health == pytest.approx(1.0)

    years = s.spending_decline_end_age - s.spending_decline_start_age
    end_base, _ = sim.spending_smile(s.spending_decline_end_age)
    assert end_base == pytest.approx(s.annual_spending_decline_rate ** years)

    # past the end: base frozen, healthcare compounding
    late_base, late_health = sim.spending_smile(s.spending_decline_end_age + 10)
    assert late_base == pytest.approx(end_base)
    assert late_health == pytest.approx((1 + s.annual_healthcare_increase_rate) ** 10)


# ------------------------------------------------------- what the report promises
def test_assumption_report_effective_return_is_delivered(base_cfg):
    """assumption_report is what a user reads to know what the model is doing, so
    its 'after regimes' return must match what the engine actually produces."""
    cfg = copy.deepcopy(base_cfg)
    cfg.market.stock_volatility = 0.0
    cfg.market.inflation_volatility = 0.0
    sim = RetirementSimulator(cfg)

    line = next(l for l in sim.assumption_report().splitlines()
                if "stock return" in l)
    printed = float(line.rsplit("->", 1)[1].strip().rstrip("% real after regimes")) / 100

    months = 240
    realised = np.array([p[0].sum() / (months / 12)
                         for p in paths(sim, months, 6000, seed=11)])
    assert math.expm1(realised.mean()) == pytest.approx(printed, abs=0.0025)


def test_assumption_report_effective_volatility_is_delivered(base_cfg):
    """The printed effective volatility must include BOTH the within-regime
    variance mixture and the dispersion of the regime means."""
    sim = RetirementSimulator(base_cfg)
    line = next(l for l in sim.assumption_report().splitlines()
                if "stock volatility" in l)
    printed = float(line.split("->")[1].split("%")[0].strip()) / 100

    cfg = copy.deepcopy(base_cfg)
    cfg.market.inflation_volatility = 0.0
    stock, _, _ = paths(RetirementSimulator(cfg), 500_000, 1, seed=12)[0]
    assert stock.std() * math.sqrt(12) == pytest.approx(printed, abs=0.004)


def test_assumption_report_mentions_every_transformed_value(base_cfg):
    """A cheap guard on the report's completeness: every value the model TRANSFORMS
    before use should appear, so the printout cannot silently fall out of date."""
    text = RetirementSimulator(base_cfg).assumption_report()
    for fragment in ("nominal", "real", "after regimes", "crisis regime",
                     "effective", "Social Security", "healthcare",
                     "early withdrawal penalty", "RMDs", "filing status"):
        assert fragment in text


def test_lives_that_reach_the_cap_stop_exactly_there(base_cfg):
    """The Gompertz walk ends at death_age_max for anyone who survives the whole
    hazard curve, and a fractional age gap must not overshoot it (death_age_max
    also sizes the trajectory chart, so an overshoot would index past the end)."""
    sim = RetirementSimulator(base_cfg)
    for start, cap in ((65.0, 70.0), (65.0, 70.5), (35.5, 80.0)):
        draws = [sim._actuarial_death_age("male", start, cap, np.random.default_rng(i))
                 for i in range(3000)]
        assert max(draws) <= cap + 1e-12
        assert min(draws) >= start
        assert draws.count(cap) > 0            # the point mass at the cap is reached
