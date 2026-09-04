"""Method-level tests of TaxCalculator against hand-computed 2026 tax law.

Every expected value below is worked out by hand from the frozen baseline's 2026
tables (IRS Rev. Proc. 2025-32; IRC 86 for Social Security; IRC 1411 for NIIT).
The arithmetic is shown in each test so a future bracket update can be re-derived
line by line.
"""

import copy

import pytest

from retirement_age_calculator import RetirementSimulator, TaxCalculator


@pytest.fixture()
def tax(base_cfg):
    """A TaxCalculator over the frozen 2026 tables."""
    return TaxCalculator(base_cfg.taxes)


# ---------------------------------------------------------------- ordinary income
def test_mfj_ordinary_only(tax):
    """MFJ, $100,000 ordinary income, nothing else, both spouses 70.

    deduction = 32,200 + 2 x 1,650 (IRC 63(f), both 65+) = 35,500
    taxable = 100,000 - 35,500 = 64,500
    federal = 24,800 x 10% + (64,500 - 24,800) x 12% = 2,480 + 4,764 = 7,244
    state   = 5% x 64,500 = 3,225
    """
    assert tax.total_tax(100_000, 0, 0, 0, "married_filing_jointly", 70,
                         spouse_age=70) == pytest.approx(10_469.0)


def test_single_same_income_pays_more(tax):
    """The widow's tax penalty in one number: the SAME $100,000 filed single at 70.

    deduction = 16,100 + 2,050 = 18,150
    taxable = 100,000 - 18,150 = 81,850
    federal = 12,400 x 10% + 38,000 x 12% + 31,450 x 22% = 1,240 + 4,560 + 6,919 = 12,719
    state   = 5% x 81,850 = 4,092.50
    """
    assert tax.total_tax(100_000, 0, 0, 0, "single", 70) == pytest.approx(16_811.50)


def test_bracket_edge_exact(tax):
    """Taxable income landing exactly on the 10% bracket top is all taxed at 10%.

    ordinary = 24,800 + 35,500 so taxable is exactly 24,800 -> federal 2,480.
    """
    bd = tax.tax_breakdown(60_300, 0, 0, 0, "married_filing_jointly", 70, spouse_age=70)
    assert bd["federal_ordinary"] == pytest.approx(2_480.0)


def test_deduction_zeroes_small_income(tax):
    """Income below the standard deduction owes nothing at all."""
    assert tax.total_tax(30_000, 0, 0, 0, "married_filing_jointly", 70,
                         spouse_age=70) == pytest.approx(0.0)


# ------------------------------------------- the age-65 addition (IRC 63(f))
def test_additional_deduction_counts_qualifying_people(tax):
    """63(f) is claimed once per filer aged 65+, and only a joint return can claim
    it twice: 2026 is 1,650 per spouse on a joint return, 2,050 unmarried."""
    mfj = "married_filing_jointly"
    assert tax.total_standard_deduction(mfj, 64, 64) == pytest.approx(32_200.0)
    assert tax.total_standard_deduction(mfj, 65, 64) == pytest.approx(33_850.0)
    assert tax.total_standard_deduction(mfj, 70, 70) == pytest.approx(35_500.0)
    # a lone survivor on a joint-rate return claims it once, not twice
    assert tax.total_standard_deduction(mfj, 70, None) == pytest.approx(33_850.0)
    assert tax.total_standard_deduction("single", 64) == pytest.approx(16_100.0)
    assert tax.total_standard_deduction("single", 65) == pytest.approx(18_150.0)
    # the base accessor stays the under-65 figure
    assert tax.standard_deduction("single") == pytest.approx(16_100.0)


def test_age_65_addition_lowers_the_bill_at_exactly_65(tax):
    """The addition switches on at 65, not 64: $60k ordinary, single.

    at 64: taxable 43,900 -> 12,400 x 10% + 31,500 x 12% = 5,020; state 2,195
    at 65: taxable 41,850 -> 12,400 x 10% + 29,450 x 12% = 4,774; state 2,092.50
    """
    assert tax.total_tax(60_000, 0, 0, 0, "single", 64) == pytest.approx(7_215.00)
    assert tax.total_tax(60_000, 0, 0, 0, "single", 65) == pytest.approx(6_866.50)


# ---------------------------------------------------------------- capital gains
def test_ltcg_zero_band(tax):
    """MFJ (both 70), $20k ordinary + $60k gains: all inside the 0% LTCG band.

    taxable_total = 80,000 - 35,500 = 44,500 < 98,900 -> LTCG tax is exactly zero.
    """
    bd = tax.tax_breakdown(20_000, 60_000, 0, 60_000, "married_filing_jointly", 70,
                           spouse_age=70)
    assert bd["federal_ltcg"] == pytest.approx(0.0)


def test_ltcg_stacking(tax):
    """Gains stack ON TOP of ordinary income when choosing the LTCG rate.

    ordinary 90k -> taxable_ordinary 54,500; +60k gains -> taxable_total 114,500.
    Only the slice above the 98,900 0%-band top is taxed:
    (114,500 - 98,900) x 15% = 2,340.
    """
    bd = tax.tax_breakdown(90_000, 60_000, 0, 60_000, "married_filing_jointly", 70,
                           spouse_age=70)
    assert bd["federal_ltcg"] == pytest.approx(2_340.0)


def test_single_zero_band_is_half(tax):
    """The single filer's 0% band tops out near half the MFJ band (49,450 vs
    98,900) -- the regression that a wrong YAML value would reintroduce."""
    # 60k gains, no ordinary income: taxable_total = 41,850 -> still 0% for single
    bd = tax.tax_breakdown(0, 60_000, 0, 60_000, "single", 70)
    assert bd["federal_ltcg"] == pytest.approx(0.0)
    # 80k gains: taxable_total = 61,850 -> (61,850 - 49,450) x 15% = 1,860.00
    bd = tax.tax_breakdown(0, 80_000, 0, 80_000, "single", 70)
    assert bd["federal_ltcg"] == pytest.approx(1_860.00)


def test_top_brackets_reached(tax):
    """The 37% ordinary and 20% LTCG bands, and both open-ended top brackets --
    never touched by any other test, so a typo in the top row would pass silently.

    single, $900,000 ordinary: taxable = 900,000 - 18,150 = 881,850
      12,400x10 + 38,000x12 + 55,300x22 + 96,075x24 + 54,450x32 + 384,375x35
      + 241,250x37 = 1,240 + 4,560 + 12,166 + 23,058 + 17,424 + 134,531.25
      + 89,262.50 = 282,241.75
    """
    bd = tax.tax_breakdown(900_000, 0, 0, 0, "single", 70)
    assert bd["federal_ordinary"] == pytest.approx(282_241.75)
    # 20% LTCG band: single, no ordinary, $700k gains -> taxable_total 681,850
    #   49,450 at 0% + (545,500 - 49,450) x 15% + (681,850 - 545,500) x 20%
    #   = 74,407.50 + 27,270.00 = 101,677.50
    bd = tax.tax_breakdown(0, 700_000, 0, 700_000, "single", 70)
    assert bd["federal_ltcg"] == pytest.approx(101_677.50)


# ---------------------------------------------------------------- Social Security
def test_ss_below_first_threshold_untaxed(tax):
    """$40k SS alone: provisional income 20,000 < 25,000 -> none taxable."""
    assert tax.taxable_social_security(40_000, 0, "single") == pytest.approx(0.0)


def test_ss_above_second_threshold(tax):
    """$40k SS + $20k other income, single.

    provisional = 20,000 + 20,000 = 40,000 > 34,000
    tier1 = min(0.5 x (34,000 - 25,000), 0.5 x 40,000) = 4,500
    taxable = min(0.85 x (40,000 - 34,000) + 4,500, 0.85 x 40,000) = 9,600
    """
    assert tax.taxable_social_security(40_000, 20_000, "single") == pytest.approx(9_600.0)


def test_ss_middle_tier_excess_limited(tax):
    """The MIDDLE tier (t1 < provisional <= t2), where the taxable amount is half
    the excess over the first threshold. Single, $20k SS + $18k other income:

    provisional = 18,000 + 10,000 = 28,000, inside 25,000..34,000
    taxable = min(0.5 x (28,000 - 25,000), 0.5 x 20,000) = min(1,500, 10,000) = 1,500
    """
    assert tax.taxable_social_security(20_000, 18_000, "single") == pytest.approx(1_500.0)


def test_ss_middle_tier_half_benefit_cap(tax):
    """The other sub-case of the middle tier: the half-of-benefit cap binds instead
    of the excess. Single, $4k SS + $26k other income:

    provisional = 26,000 + 2,000 = 28,000, inside 25,000..34,000
    taxable = min(0.5 x 3,000, 0.5 x 4,000) = min(1,500, 2,000) = 1,500
    ...so raise other income to 31,000: min(0.5 x 6,000, 2,000) = 2,000, the cap.
    """
    assert tax.taxable_social_security(4_000, 26_000, "single") == pytest.approx(1_500.0)
    assert tax.taxable_social_security(4_000, 31_000, "single") == pytest.approx(2_000.0)


def test_ss_is_continuous_at_both_thresholds(tax):
    """No jump discontinuity at either kink -- a sign error in `tier1` would show
    up here as a step at the upper threshold."""
    below = tax.taxable_social_security(40_000, 13_999.99, "single")
    at = tax.taxable_social_security(40_000, 14_000.0, "single")     # provisional 34,000
    above = tax.taxable_social_security(40_000, 14_000.01, "single")
    assert at == pytest.approx(4_500.0)          # the full tier-1 amount
    assert below == pytest.approx(at, abs=0.02)
    assert above == pytest.approx(at, abs=0.02)


def test_ss_85_percent_cap(tax):
    """With huge other income the taxable share caps at exactly 85% of the benefit."""
    assert tax.taxable_social_security(40_000, 500_000, "single") == pytest.approx(34_000.0)


def test_ss_mfj_thresholds_differ(tax):
    """The MFJ thresholds (32k/44k) leave the same case fully untaxed."""
    assert tax.taxable_social_security(40_000, 10_000,
                                       "married_filing_jointly") == pytest.approx(0.0)


# ---------------------------------------------------------------- NIIT
def test_niit_lesser_of_rule(tax):
    """Single, 150k ordinary + 100k investment gains: AGI 250k.

    base = min(AGI - 200,000, investment income) = min(50,000, 100,000) = 50,000
    NIIT = 3.8% x 50,000 = 1,900
    """
    bd = tax.tax_breakdown(150_000, 100_000, 0, 100_000, "single", 70)
    assert bd["niit"] == pytest.approx(1_900.0)


def test_niit_not_below_threshold(tax):
    """No NIIT when AGI is under the threshold, however large investment income is."""
    bd = tax.tax_breakdown(0, 100_000, 0, 100_000, "single", 70)
    assert bd["niit"] == pytest.approx(0.0)


# ---------------------------------------------------------------- early penalty
def test_early_penalty_applies(tax):
    """10% of the traditional distribution before 59.5."""
    bd = tax.tax_breakdown(10_000, 0, 0, 0, "single", 50, traditional_withdrawal=10_000)
    assert bd["early_penalty"] == pytest.approx(1_000.0)


def test_penalty_free_after_595(tax):
    """No penalty from 59.5 onward, and the boundary itself is exact."""
    bd = tax.tax_breakdown(10_000, 0, 0, 0, "single", 60, traditional_withdrawal=10_000)
    assert bd["early_penalty"] == pytest.approx(0.0)
    just_under = tax.tax_breakdown(10_000, 0, 0, 0, "single", 59.49,
                                   traditional_withdrawal=10_000)
    exactly = tax.tax_breakdown(10_000, 0, 0, 0, "single", 59.5,
                                traditional_withdrawal=10_000)
    assert just_under["early_penalty"] == pytest.approx(1_000.0)
    assert exactly["early_penalty"] == pytest.approx(0.0)


def test_72t_waives_penalty(base_cfg):
    """A 72(t) SEPP / Roth-ladder plan disables the penalty at any age."""
    cfg = copy.deepcopy(base_cfg)
    cfg.taxes.use_72t_sepp = True
    bd = TaxCalculator(cfg.taxes).tax_breakdown(10_000, 0, 0, 0, "single", 45,
                                                traditional_withdrawal=10_000)
    assert bd["early_penalty"] == pytest.approx(0.0)


def test_rule_of_55_window(base_cfg):
    """The rule-of-55 exception waives the penalty at 55+ but not below."""
    cfg = copy.deepcopy(base_cfg)
    cfg.taxes.assume_qualified_plan_age55_exception = True
    calc = TaxCalculator(cfg.taxes)
    at56 = calc.tax_breakdown(10_000, 0, 0, 0, "single", 56, traditional_withdrawal=10_000)
    at53 = calc.tax_breakdown(10_000, 0, 0, 0, "single", 53, traditional_withdrawal=10_000)
    assert at56["early_penalty"] == pytest.approx(0.0)
    assert at53["early_penalty"] == pytest.approx(1_000.0)


# ---------------------------------------------------------------- consistency
def test_breakdown_sums_to_total(tax):
    """The itemised breakdown must reconcile with total_tax exactly."""
    bd = tax.tax_breakdown(80_000, 30_000, 25_000, 35_000, "single", 50,
                           traditional_withdrawal=40_000)
    parts = (bd["federal_ordinary"] + bd["federal_ltcg"] + bd["state"]
             + bd["niit"] + bd["early_penalty"])
    assert bd["total"] == pytest.approx(parts)
    assert tax.total_tax(80_000, 30_000, 25_000, 35_000, "single", 50,
                         traditional_withdrawal=40_000) == pytest.approx(bd["total"])


def test_survivor_filing_status_switch(base_cfg):
    """MFJ becomes single after the first death; single filers stay single."""
    sim = RetirementSimulator(base_cfg)
    assert sim.household_filing_status(True, True) == "married_filing_jointly"
    assert sim.household_filing_status(True, False) == "single"
    assert sim.household_filing_status(False, True) == "single"
