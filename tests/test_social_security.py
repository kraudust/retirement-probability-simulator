"""Method-level tests of the Social Security model against SSA's published rules.

Claim-age factors: ssa.gov/oact/quickcalc/early_late.html
PIA formula and 2026 bend points: ssa.gov/oact/cola/piaformula.html
Eligibility credits: ssa.gov/benefits/retirement/planner/credits.html
"""

import copy

import pytest

from retirement_age_calculator import RetirementSimulator


@pytest.fixture()
def sim(base_cfg):
    """A simulator over the frozen baseline (primary: $40k FRA benefit, 13 years
    and 40 credits at age 35)."""
    return RetirementSimulator(base_cfg)


# ---------------------------------------------------------------- claim factors
def test_own_record_claim_factors(sim):
    """SSA's exact schedule with FRA 67:
    62 -> 1 - (36 x 5/9 + 24 x 5/12)/100 = 0.70
    67 -> 1.00
    70 -> 1 + 36 x (2/3)/100 = 1.24  (8%/yr SIMPLE, not compounded 1.08^3)
    """
    assert sim.ss_benefit_factor(62) == pytest.approx(0.70)
    assert sim.ss_benefit_factor(67) == pytest.approx(1.00)
    assert sim.ss_benefit_factor(70) == pytest.approx(1.24)


def test_claim_age_clamped(sim):
    """Claims are only possible 62-70; out-of-range ages clamp to the bounds."""
    assert sim.ss_benefit_factor(55) == sim.ss_benefit_factor(62)
    assert sim.ss_benefit_factor(80) == sim.ss_benefit_factor(70)


def test_spousal_claim_factors(sim):
    """Spousal benefits reduce on a steeper schedule and earn no delayed credits:
    62 -> 1 - (36 x 25/36 + 24 x 5/12)/100 = 0.65 ; 70 -> capped at 1.00.
    """
    assert sim.ss_spousal_factor(62) == pytest.approx(0.65)
    assert sim.ss_spousal_factor(67) == pytest.approx(1.00)
    assert sim.ss_spousal_factor(70) == pytest.approx(1.00)


# ---------------------------------------------------------------- PIA formula
def test_pia_aime_round_trip(sim):
    """The AIME<->PIA inversion must be exact in all three bend-point segments."""
    for pia in (500.0, 2_000.0, 4_000.0):
        assert sim._pia_from_aime(sim._aime_from_pia(pia)) == pytest.approx(pia)


def test_pia_bend_point_values(sim):
    """Spot-check the formula at the 2026 bend points themselves:
    PIA(1,286) = 0.90 x 1,286 = 1,157.40
    PIA(7,749) = 1,157.40 + 0.32 x (7,749 - 1,286) = 3,225.56
    """
    assert sim._pia_from_aime(1_286.0) == pytest.approx(1_157.40)
    assert sim._pia_from_aime(7_749.0) == pytest.approx(3_225.56)


def test_progressivity_beats_linear(sim, base_cfg):
    """A 60% career (21 of 35 years) must pay MORE than 60% of the full benefit,
    because the formula replaces the first dollars of AIME at 90%."""
    le = copy.deepcopy(base_cfg.life_events)
    le.ss_earnings_years_at_current_age = 21
    partial = sim.ss_fra_benefit(le, 0)
    full = sim.ss_fra_benefit(base_cfg.life_events, 35)
    assert full == pytest.approx(40_000.0, abs=1.0)
    assert partial / full > 21 / 35 + 0.02


def test_forty_credit_eligibility(sim, base_cfg):
    """8 credits and no further work -> NO benefit; ten more working years (40
    more credits) restore eligibility."""
    le = copy.deepcopy(base_cfg.life_events)
    le.ss_credits_at_current_age = 8
    assert sim.ss_fra_benefit(le, 0) == 0.0
    assert sim.ss_fra_benefit(le, 10) > 0.0


# ---------------------------------------------------------------- household benefits
def test_stay_at_home_spouse_gets_half_pia(base_cfg):
    """A spouse with no record of their own collects exactly 50% of the primary's
    PIA when claiming at FRA (the baseline spouse has benefit/years/credits 0)."""
    cfg = copy.deepcopy(base_cfg)
    cfg.spouse.enabled = True
    sim = RetirementSimulator(cfg)
    primary_pia = sim.ss_fra_benefit(cfg.life_events, 65 - cfg.simulation.current_age)
    assert sim.spouse_ss_income(65) == pytest.approx(0.5 * primary_pia, abs=0.01)


def test_death_caps_the_work_record(base_cfg):
    """A primary who dies after only 10 more working years must leave a smaller
    record -- and a smaller spousal/survivor benefit -- than one who works to the
    tested retirement age. (Regression: benefits once assumed work continued to
    the tested age even in lifetimes that ended first.)"""
    cfg = copy.deepcopy(base_cfg)
    cfg.spouse.enabled = True
    sim = RetirementSimulator(cfg)
    full = sim.spouse_ss_income(65)
    died_young = sim.spouse_ss_income(65, primary_work_years=10.0, spouse_work_years=10.0)
    assert died_young < full - 1.0
    # explicit years equal to the default must change nothing
    default_years = 65 - cfg.simulation.current_age
    assert sim.spouse_ss_income(65, default_years, default_years) == pytest.approx(full)


def test_primary_income_scales_with_retirement_age(sim):
    """Retiring later adds covered years, so the benefit is non-decreasing in the
    retirement age and capped at the full-career figure times the claim factor."""
    incomes = [sim.primary_ss_income(a) for a in (40, 50, 60, 70)]
    assert all(a <= b + 1e-9 for a, b in zip(incomes, incomes[1:]))
    assert incomes[-1] == pytest.approx(40_000.0, abs=1.0)   # claim at FRA, full career


# ------------------------------------------------------------- claim-factor sweeps
def test_own_benefit_factor_at_every_claim_age(sim):
    """SSA's published own-record factors, every age 62-70 (5/9 of 1% per month for
    the first 36 months early, 5/12 beyond; 8%/yr SIMPLE delayed credits)."""
    expected = {62: 0.700000, 63: 0.750000, 64: 0.800000, 65: 0.866667, 66: 0.933333,
                67: 1.000000, 68: 1.080000, 69: 1.160000, 70: 1.240000}
    for age, factor in expected.items():
        assert sim.ss_benefit_factor(age) == pytest.approx(factor, abs=5e-7)


def test_spousal_factor_at_every_claim_age(sim):
    """Spousal benefits reduce on a STEEPER schedule (25/36 of 1% per month for the
    first 36 months) and earn NO delayed credits, so they flatten at 1.0 past FRA."""
    expected = {62: 0.650000, 63: 0.700000, 64: 0.750000, 65: 0.833333, 66: 0.916667,
                67: 1.000000, 68: 1.000000, 69: 1.000000, 70: 1.000000}
    for age, factor in expected.items():
        assert sim.ss_spousal_factor(age) == pytest.approx(factor, abs=5e-7)


def test_benefit_factors_honour_a_non_default_fra(sim):
    """The full_retirement_age argument must actually move the schedule -- with an
    FRA of 66, claiming at 66 is unreduced and 62 is a 25% cut, not 30%."""
    assert sim.ss_benefit_factor(66, full_retirement_age=66) == pytest.approx(1.0)
    assert sim.ss_benefit_factor(62, full_retirement_age=66) == pytest.approx(0.75, abs=5e-7)
    assert sim.ss_spousal_factor(66, full_retirement_age=66) == pytest.approx(1.0)


# ---------------------------------------------------------------- survivor benefits
def test_survivor_factor_follows_life_not_intent(sim):
    """A claim age is only a plan until it is reached. Delayed credits accrue month
    by month while ALIVE and unfiled, and dying before filing means no early-claim
    cut ever happened."""
    assert sim.ss_survivor_factor(70, 63) == pytest.approx(1.00)      # no DRCs earned
    assert sim.ss_survivor_factor(70, 68) == pytest.approx(1.08)      # one year of DRCs
    assert sim.ss_survivor_factor(70, 71) == pytest.approx(1.24)      # filed and lived
    assert sim.ss_survivor_factor(67, 63) == pytest.approx(1.00)      # died before FRA
    assert sim.ss_survivor_factor(70, 75) == pytest.approx(1.24)      # DRCs stop at 70


def test_survivor_factor_applies_the_widows_limit(sim):
    """A deceased who claimed EARLY leaves the survivor their reduced benefit, but
    never below 82.5% of PIA (RIB-LIM, 20 CFR 404.338)."""
    assert sim.ss_survivor_factor(62, 70) == pytest.approx(0.825)     # 0.70 floored up
    assert sim.ss_survivor_factor(63, 70) == pytest.approx(0.825)     # 0.75 floored up
    assert sim.ss_survivor_factor(65, 70) == pytest.approx(0.866667, abs=5e-7)  # above floor


def test_survivor_income_through_the_engine(base_cfg):
    """End to end, in dollars: a widow's household income after the higher earner
    dies at 63 having PLANNED to claim at 70.

    The deceased earned no delayed credits -- they died four years before FRA -- so
    the survivor benefit is the deceased's PIA ($50,000), not 1.24 x PIA ($62,000).
    The survivor's own record is empty, so $50,000 is the whole household benefit.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.current_age = 60
    cfg.spouse.enabled = True
    cfg.spouse.age_offset = 0
    cfg.spouse.ss_annual_full_retirement_benefit = 50_000.0
    cfg.spouse.ss_earnings_years_at_current_age = 35
    cfg.spouse.ss_credits_at_current_age = 40
    cfg.spouse.ss_claim_age = 70
    cfg.life_events.ss_annual_full_retirement_benefit = 0.0
    cfg.life_events.ss_earnings_years_at_current_age = 0
    cfg.life_events.ss_credits_at_current_age = 0
    sim = RetirementSimulator(cfg)

    spouse_fra = sim.ss_fra_benefit(cfg.spouse, 3.0)
    assert spouse_fra == pytest.approx(50_000.0, abs=1.0)
    # died at 63 with a claim age of 70: factor 1.0, so the survivor gets the PIA
    assert spouse_fra * sim.ss_survivor_factor(70, 63.0) == pytest.approx(50_000.0, abs=1.0)
    # and NOT the 1.24x the old code applied
    assert spouse_fra * sim.ss_benefit_factor(70) == pytest.approx(62_000.0, abs=1.0)


def test_spousal_benefit_is_not_inheritable(base_cfg):
    """A partner with no earnings record collects a SPOUSAL benefit while alive but
    leaves NO survivor benefit -- survivor benefits are built on the deceased's own
    record, and an empty record leaves nothing."""
    cfg = copy.deepcopy(base_cfg)
    cfg.spouse.enabled = True
    sim = RetirementSimulator(cfg)
    years = 65 - cfg.simulation.current_age
    assert sim.spouse_ss_income(65) > 0.0                       # spousal while alive
    assert sim.ss_fra_benefit(cfg.spouse, years) == 0.0         # nothing to inherit
