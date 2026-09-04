"""Tests of the annual withdrawal-plan solver: exactness, money conservation,
ladder order, RMD mechanics, and its known-answer tax integration.

The solver's contract: after-tax proceeds cover the year's net need to within a
cent, no dollar is created or destroyed, and every tax comes from the actual
withdrawal composition.
"""

import pytest

from retirement_age_calculator import RetirementSimulator


@pytest.fixture()
def sim(base_cfg):
    """Solver host; only tax tables and market yields are read from the config."""
    return RetirementSimulator(base_cfg)


def run_plan(sim, **kw):
    """_annual_withdrawal_plan with keyword defaults for readable test cases."""
    args = dict(net_need=80_000.0, balances=(20_000.0, 150_000.0, 400_000.0, 100_000.0),
                brokerage_basis=90_000.0, household_age=55.0,
                status="married_filing_jointly", ss_income=0.0,
                stock_alloc=0.7, bond_alloc=0.3, rmd_amount=0.0)
    args.update(kw)
    return sim._annual_withdrawal_plan(**args)


# ---------------------------------------------------------------- core contract
def test_money_conservation(sim):
    """Balance outflow must equal gross withdrawal minus reinvested surplus."""
    balances = (20_000.0, 150_000.0, 400_000.0, 100_000.0)
    plan = run_plan(sim, balances=balances)
    outflow = sum(balances) - sum(plan["balances"])
    assert outflow == pytest.approx(plan["gross_total"] - plan["surplus_reinvested"], abs=0.01)


def test_net_covers_need_tightly(sim):
    """After-tax proceeds reach the need, and overshoot by less than a dollar."""
    plan = run_plan(sim)
    net = plan["gross_total"] - plan["tax"]
    assert net >= 80_000.0 - 0.01
    assert net < 80_000.0 + 1.0


def test_ladder_order_cash_first(sim):
    """The cheapest rung (already-taxed cash) empties before anything else."""
    plan = run_plan(sim)
    assert plan["take_cash"] == pytest.approx(20_000.0)


def test_net_monotone_in_gross(sim):
    """The bisection is valid because net proceeds rise with the gross withdrawal
    (every marginal tax component stays below 100%)."""
    balances = (5_000.0, 10_000.0, 400_000.0, 100_000.0)
    nets = []
    for need in (20_000.0, 50_000.0, 90_000.0):
        plan = run_plan(sim, net_need=need, balances=balances, brokerage_basis=8_000.0,
                        household_age=50.0)
        nets.append(plan["gross_total"] - plan["tax"])
    assert nets == sorted(nets)


def test_returns_none_when_broke(sim):
    """Even liquidating everything cannot fund the year -> None, not a wrong plan."""
    assert run_plan(sim, balances=(1_000.0, 0.0, 0.0, 0.0), brokerage_basis=0.0) is None


# ---------------------------------------------------------------- tax integration
def test_zero_ltcg_band_makes_modest_year_tax_free(sim):
    """MFJ funding $80k mostly from basis-heavy brokerage sales lands inside the
    0% LTCG band and the standard deduction: essentially no tax at all."""
    plan = run_plan(sim)
    assert plan["tax"] < 500.0


def test_early_penalty_reaches_the_plan(sim):
    """With cash and brokerage nearly empty at 50, the ladder must dip into the
    traditional account and the 10% penalty must show up in the year's tax."""
    plan = run_plan(sim, balances=(5_000.0, 10_000.0, 400_000.0, 100_000.0),
                    brokerage_basis=8_000.0, household_age=50.0)
    assert plan["take_traditional"] > 0
    assert plan["tax"] > 0.09 * plan["take_traditional"]


def test_zero_need_still_pays_investment_income_tax(sim):
    """Dividends and interest are taxed even when spending needs nothing: the plan
    withdraws exactly enough to pay that tax bill."""
    plan = run_plan(sim, net_need=0.0, balances=(50_000.0, 3_000_000.0, 0.0, 0.0),
                    brokerage_basis=500_000.0, household_age=70.0, status="single")
    assert plan["tax"] > 1_000.0
    assert plan["gross_total"] - plan["tax"] == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------- RMD mechanics
def test_rmd_covers_need_without_extra_withdrawal(sim):
    """When the forced RMD alone more than covers the year, nothing extra is
    withdrawn -- the regression that once double-withdrew RMD years."""
    rmd = 500_000.0 / 23.7                      # age 76 divisor
    plan = run_plan(sim, net_need=10_000.0, balances=(0.0, 0.0, 500_000.0, 0.0),
                    brokerage_basis=0.0, household_age=76.0, status="single",
                    ss_income=30_000.0, rmd_amount=rmd)
    assert plan["gross_total"] == pytest.approx(rmd, abs=0.01)


def test_rmd_surplus_reinvested_with_basis(sim):
    """After-tax RMD money beyond the need lands in the brokerage as new shares
    with full basis, instead of idling at zero return."""
    rmd = 500_000.0 / 23.7
    plan = run_plan(sim, net_need=10_000.0, balances=(0.0, 0.0, 500_000.0, 0.0),
                    brokerage_basis=0.0, household_age=76.0, status="single",
                    ss_income=30_000.0, rmd_amount=rmd)
    assert plan["surplus_reinvested"] > 0
    assert plan["balances"][1] == pytest.approx(plan["surplus_reinvested"], abs=0.01)
    assert plan["new_basis"] == pytest.approx(plan["surplus_reinvested"], abs=0.01)


def test_rmd_known_answer_to_the_cent(sim):
    """Fully hand-computed RMD year, single filer, no other income:

    RMD       = 500,000 / 23.7           = 21,097.0464
    deduction = 16,100 + 2,050 (age 65+) = 18,150
    taxable   = 21,097.0464 - 18,150     =  2,947.0464
    federal   = 10% x 2,947.0464         =    294.7046   (all inside the 10% bracket)
    state     =  5% x 2,947.0464         =    147.3523
    tax                                   =    442.0570
    net       = 21,097.0464 - 442.0570   = 20,654.9895
    surplus   = net - 10,000             = 10,654.9895  -> reinvested in brokerage
    """
    rmd = 500_000.0 / 23.7
    plan = run_plan(sim, net_need=10_000.0, balances=(0.0, 0.0, 500_000.0, 0.0),
                    brokerage_basis=0.0, household_age=76.0, status="single",
                    ss_income=0.0, rmd_amount=rmd)
    assert plan["tax"] == pytest.approx(442.0570, abs=0.01)
    assert plan["surplus_reinvested"] == pytest.approx(10_654.9895, abs=0.01)
    assert plan["balances"][2] == pytest.approx(500_000.0 - rmd, abs=0.01)


def test_rmd_divisor_gated_by_start_age(sim):
    """No divisor before the configured start age (75 in the baseline); the IRS
    table value from then on; the 120+ floor far out."""
    assert sim.rmd_divisor(74) is None
    assert sim.rmd_divisor(75) == 24.6
    assert sim.rmd_divisor(90) == 12.2
    assert sim.rmd_divisor(130) == 2.0


# ---------------------------------------------------------------- basis handling
def test_dividends_raise_basis_before_sale(sim):
    """This year's reinvested distributions count as basis before the sale's gain
    fraction is computed: with basis already at the balance, a sale realises no
    gain at all, and the year's only gains are the dividends themselves."""
    plan = run_plan(sim, net_need=50_000.0, balances=(0.0, 200_000.0, 0.0, 0.0),
                    brokerage_basis=200_000.0, household_age=70.0, status="single")
    qdiv = 200_000.0 * 0.7 * sim.cfg.market.stock_dividend_yield
    assert plan["realized_gain"] == pytest.approx(0.0, abs=0.01)
    # gains reported to the tax engine that year = dividends only
    assert qdiv > 0


def test_underwater_account_keeps_its_basis(sim):
    """An account worth LESS than was paid for it must keep the excess basis.

    Balance $584,000 against $980,000 of basis. Selling $10,000 is 1.7123% of the
    account, so it carries away 1.7123% of the basis -- $16,780.82 -- leaving
    $963,219.18. The basis/balance RATIO is unchanged, which is what "pro-rata"
    means. Clamping the basis down to the balance instead would delete $396,000
    for good and tax the eventual recovery as if it were profit.
    """
    plan = run_plan(sim, net_need=10_000.0, balances=(0.0, 584_000.0, 0.0, 0.0),
                    brokerage_basis=980_000.0, household_age=70.0, status="single",
                    ss_income=0.0, stock_alloc=0.0, bond_alloc=0.0, rmd_amount=0.0)
    assert plan["take_brokerage"] == pytest.approx(10_000.0, abs=0.01)
    assert plan["realized_gain"] == pytest.approx(0.0)     # sold at a loss, not a gain
    assert plan["new_basis"] == pytest.approx(963_219.178, abs=0.01)
    # The ratio is preserved up to the solver's sub-cent overshoot, which is
    # reinvested as surplus into the balance and the basis alike.
    assert (plan["new_basis"] / plan["balances"][1]
            == pytest.approx(980_000.0 / 584_000.0, rel=1e-6))


def test_recovery_after_a_crash_is_not_taxed_as_phantom_gain(sim):
    """The consequence of the test above: after the market doubles back, tax is due
    only on the gain over the SURVIVING basis."""
    crashed = run_plan(sim, net_need=10_000.0, balances=(0.0, 584_000.0, 0.0, 0.0),
                       brokerage_basis=980_000.0, household_age=70.0, status="single",
                       ss_income=0.0, stock_alloc=0.0, bond_alloc=0.0, rmd_amount=0.0)
    balance, basis = crashed["balances"][1], crashed["new_basis"]
    recovered = run_plan(sim, net_need=10_000.0, balances=(0.0, balance * 2, 0.0, 0.0),
                         brokerage_basis=basis, household_age=70.0, status="single",
                         ss_income=0.0, stock_alloc=0.0, bond_alloc=0.0, rmd_amount=0.0)
    gain_share = 1 - basis / (balance * 2)
    assert recovered["realized_gain"] == pytest.approx(
        recovered["take_brokerage"] * gain_share, rel=1e-9)
    assert recovered["realized_gain"] < 2_000.0        # would be ~4x with a clamped basis


def test_ladder_conserves_basis_and_gain(sim):
    """Mechanical identity of every brokerage sale: the amount withdrawn is exactly
    the basis returned plus the gain realised, at any basis fraction."""
    balances = (0.0, 200_000.0, 0.0, 0.0)
    for basis_fraction in (0.0, 0.25, 0.5, 1.0):
        takes = sim._ladder_withdraw(50_000.0, balances, basis_fraction)
        assert (takes["basis_returned"] + takes["realized_gain"]
                == pytest.approx(takes["brokerage"], rel=1e-12))
    # underwater: gain floors at zero while basis stays strictly pro-rata
    takes = sim._ladder_withdraw(50_000.0, balances, 1.5)
    assert takes["realized_gain"] == 0.0
    assert takes["basis_returned"] == pytest.approx(75_000.0)


def test_ladder_drains_in_the_documented_order(sim):
    """cash -> brokerage -> traditional -> Roth, cheapest tax first."""
    balances = (10_000.0, 20_000.0, 30_000.0, 40_000.0)
    assert sim._ladder_withdraw(5_000.0, balances, 1.0)["cash"] == 5_000.0
    mid = sim._ladder_withdraw(45_000.0, balances, 1.0)
    assert (mid["cash"], mid["brokerage"], mid["traditional"], mid["roth"]) == \
        (10_000.0, 20_000.0, 15_000.0, 0.0)
    everything = sim._ladder_withdraw(200_000.0, balances, 1.0)
    assert everything["unfunded"] == pytest.approx(100_000.0)
    assert sum(everything[k] for k in ("cash", "brokerage", "traditional", "roth")) \
        == pytest.approx(100_000.0)


def test_rmd_table_matches_the_irs_uniform_lifetime_table():
    """Pin the whole table, not a spot check. These divisors are law (Pub. 590-B,
    post-2022); a transcription slip changes every forced withdrawal after 75."""
    from retirement_age_calculator import RMD_TABLE
    expected = {
        72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0,
        79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0,
        86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8,
        93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
        101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
        108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
        115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
    }
    assert RMD_TABLE == expected
    assert all(a >= b for a, b in zip(expected.values(), list(expected.values())[1:]))


def test_bisection_converges_tightly_on_a_huge_portfolio(sim):
    """The solver must land just above the need, not merely somewhere above it,
    even when the search interval spans tens of millions."""
    plan = run_plan(sim, net_need=250_000.0,
                    balances=(0.0, 20_000_000.0, 20_000_000.0, 10_000_000.0),
                    brokerage_basis=5_000_000.0, household_age=70.0,
                    status="married_filing_jointly", ss_income=0.0)
    net = plan["gross_total"] - plan["tax"]
    assert net >= 250_000.0 - 1e-6
    assert net - 250_000.0 < 0.01
