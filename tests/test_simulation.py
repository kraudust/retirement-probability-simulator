"""Whole-engine tests of simulate_life.

The deterministic scenarios are the strongest checks in the suite: with zero
volatility and fixed lifespans the engine's monthly loop reduces to arithmetic a
test can replicate in a few lines, so growth/withdrawal ordering, plan timing and
the success accounting are all pinned to the dollar. The stochastic tests cover
reproducibility, common random numbers, and the regression cases found in review.
"""

import copy

import numpy as np
import pytest

from retirement_age_calculator import RetirementSimulator
from tests.conftest import deterministic_cfg, trinity_cfg


# ---------------------------------------------------------------- closed form
def test_deterministic_final_balance_matches_recurrence(base_cfg):
    """30-year retirement at exactly 4% real, $50k constant spending, no taxes.

    (At 4% real a 5% initial withdrawal survives 30 years with roughly $337k to
    spare -- at 2% real it would rightly fail around year 25, which the engine
    detects; see test_depletion_before_death_is_failure for that side.)

    The engine's documented timing is: each month applies growth first; on plan
    anniversaries the year's spending is then withdrawn in one transaction. So
    with monthly factor g = 1.04^(1/12), each plan year is
        R <- R*g - S        (anniversary month: growth, then withdrawal)
        R <- R*g^11         (the other eleven months)
    Replicating that recurrence must reproduce the engine's final balance to
    within a dollar (the slack is bisection dust, < $0.005/year, reinvested).
    """
    cfg = deterministic_cfg(base_cfg, real_return=0.04, spend=50_000.0,
                            roth=1_000_000.0, retire_age=65, death_age=95)
    sim = RetirementSimulator(cfg)
    survived, _, iwr, final, traj = sim.simulate_life(65, random_seed=0,
                                                      record_trajectory=True)
    g = 1.04 ** (1 / 12)
    R = 1_000_000.0
    for _ in range(30):
        R = R * g - 50_000.0
        R *= g ** 11
    assert survived is True
    assert final == pytest.approx(R, abs=1.0)
    # first-year gross withdrawal is exactly the $50k need (taxes are zero),
    # over the portfolio measured after the first month's growth
    assert iwr == pytest.approx(50_000.0 / (1_000_000.0 * g), abs=1e-6)
    # one trajectory point per retirement year, anchored at the tested age;
    # the first records the post-plan, post-first-bill total
    assert len(traj) == 30
    assert traj[0] == pytest.approx(1_000_000.0 * g - 50_000.0 / 12, abs=1.0)


def test_exact_depletion_boundary_is_success(base_cfg):
    """$900k at 0% real spending $30k for exactly 30 years: the last dollar goes
    out with the last bill, death follows -- the money outlasted the person, so
    this is a SUCCESS with a ~$0 estate, not a failure."""
    cfg = deterministic_cfg(base_cfg, real_return=0.0, spend=30_000.0,
                            roth=900_000.0, retire_age=65, death_age=95)
    survived, _, _, final, _ = RetirementSimulator(cfg).simulate_life(65, random_seed=0)
    assert survived is True
    assert final == pytest.approx(0.0, abs=1.0)


def test_depletion_before_death_is_failure(base_cfg):
    """$1M at 0% real spending $100k funds exactly ten plan years; the eleventh
    plan finds nothing and the run fails."""
    cfg = deterministic_cfg(base_cfg, real_return=0.0, spend=100_000.0,
                            roth=1_000_000.0, retire_age=65, death_age=95)
    survived, min_portfolio, _, final, _ = RetirementSimulator(cfg).simulate_life(
        65, random_seed=0)
    assert survived is False
    assert final == pytest.approx(0.0, abs=1.0)
    assert min_portfolio == pytest.approx(0.0, abs=1.0)


def test_rmd_year_known_answer_through_the_engine(base_cfg):
    """One full RMD year wired through simulate_life, hand-computed to the cent.

    A single 75-year-old with $500k traditional (0% real growth) and $10k
    spending. At the first anniversary:
      RMD       = 500,000 / 24.6           = 20,325.2033   (divisor at 75)
      deduction = 16,100 + 2,050 (age 65+) = 18,150
      taxable   = 20,325.2033 - 18,150     =  2,175.2033
      tax       = 10% federal + 5% state   =    326.2805
      net       =                            19,998.9228
    Spending takes 10,000; the surplus 9,998.9228 is reinvested in the brokerage.
    Death at 76 ends the run after that single year, so the final estate is
      (500,000 - 20,325.2033) + 9,998.9228 = 489,673.7195.
    """
    cfg = deterministic_cfg(base_cfg, real_return=0.0, spend=10_000.0,
                            roth=0.0, retire_age=75, death_age=76)
    cfg.accounts.traditional = 500_000.0
    cfg.taxes.filing_status = "single"
    survived, _, _, final, _ = RetirementSimulator(cfg).simulate_life(75, random_seed=0)
    assert survived is True
    assert final == pytest.approx(489_673.7195, abs=0.5)


# ---------------------------------------------------------------- reproducibility
def test_same_seed_same_life(base_cfg):
    """Identical seeds must reproduce a lifetime bit for bit."""
    sim = RetirementSimulator(base_cfg)
    a = sim.simulate_life(50, random_seed=123)
    b = sim.simulate_life(50, random_seed=123)
    assert a[:4] == b[:4]


def test_different_seed_different_life(base_cfg):
    """Different seeds must give a different lifetime (different market luck)."""
    sim = RetirementSimulator(base_cfg)
    assert (sim.simulate_life(50, random_seed=123)[3]
            != sim.simulate_life(50, random_seed=124)[3])


def test_common_random_numbers_share_scenarios(base_cfg):
    """With CRN, the same seed at adjacent retirement ages is the SAME lifetime
    (same death, same market path) -- so on a portfolio too rich to fail, the
    final balances differ only by one extra year of contributions, not by luck."""
    cfg = copy.deepcopy(base_cfg)
    cfg.accounts.roth = 5e7
    sim = RetirementSimulator(cfg)
    at45 = sim.simulate_life(45, random_seed=999)[3]
    at46 = sim.simulate_life(46, random_seed=999)[3]
    assert abs(at45 - at46) / at45 < 0.25


# ---------------------------------------------------------------- regressions
def test_ss_covered_zero_portfolio_is_success(base_cfg):
    """A retiree whose Social Security fully covers spending succeeds with a $0
    portfolio. (Regression: an old sentinel flagged total==0 as failure.)"""
    cfg = copy.deepcopy(base_cfg)
    for f in ("roth", "traditional", "brokerage", "cash", "brokerage_cost_basis"):
        setattr(cfg.accounts, f, 0.0)
    for f in ("annual_roth", "annual_traditional", "annual_brokerage"):
        setattr(cfg.contributions, f, 0.0)
    cfg.simulation.current_age = 68
    cfg.simulation.min_retirement_age = 68
    cfg.simulation.max_retirement_age = 68
    cfg.spending.initial_annual_expenses = 5_000.0    # far below the SS benefit
    cfg.healthcare.pre_medicare_annual_premium = 0.0
    cfg.healthcare.medicare_annual_premium = 0.0
    cfg.life_events.ss_earnings_years_at_current_age = 35
    survived = RetirementSimulator(cfg).simulate_life(68, random_seed=3)[0]
    assert survived is True


def test_spouse_lifetimes_run_clean(base_cfg):
    """Smoke test across the survivor/filing-status branches: 50 spouse-enabled
    lifetimes must complete with well-formed results."""
    cfg = copy.deepcopy(base_cfg)
    cfg.spouse.enabled = True
    cfg.spouse.ss_annual_full_retirement_benefit = 20_000.0
    cfg.spouse.ss_earnings_years_at_current_age = 10
    cfg.spouse.ss_credits_at_current_age = 40
    sim = RetirementSimulator(cfg)
    for seed in range(50):
        survived, min_p, iwr, final, _ = sim.simulate_life(55, random_seed=seed)
        assert survived in (True, False)
        assert final >= 0.0


def test_survivor_forced_early_drawdown(base_cfg):
    """A primary dying around 45 while age 65 is being tested forces the widowed
    household into drawdown twenty years early; the run must handle it."""
    cfg = copy.deepcopy(base_cfg)
    cfg.spouse.enabled = True
    cfg.life_events.mortality_model = "normal"
    cfg.life_events.death_age_mean = 45
    cfg.life_events.death_age_std = 1.0
    cfg.life_events.death_age_min = 44
    cfg.life_events.death_age_max = 46
    survived = RetirementSimulator(cfg).simulate_life(65, random_seed=11)[0]
    assert survived in (True, False)


def test_death_before_retirement_is_success(base_cfg):
    """A life ending before the tested retirement age never draws down: the money
    outlasted the person, counted as success with the accumulated estate."""
    cfg = deterministic_cfg(base_cfg, real_return=0.0, spend=50_000.0,
                            roth=100_000.0, retire_age=65, death_age=95)
    cfg.simulation.current_age = 40
    cfg.simulation.min_retirement_age = 65
    cfg.simulation.max_retirement_age = 65
    cfg.life_events.death_age_mean = 50
    cfg.life_events.death_age_min = 50
    cfg.life_events.death_age_max = 50
    survived, _, iwr, final, _ = RetirementSimulator(cfg).simulate_life(65, random_seed=0)
    assert survived is True
    assert iwr is None                      # no retirement statistics exist
    assert final == pytest.approx(100_000.0, abs=1.0)   # 0% real, no contributions


# ------------------------------------------------- cost basis is nominal, not real
# Every balance in this model is real, but the brokerage cost basis is the historical
# dollar figure on the 1099 and the IRS never indexes it. Its REAL value must
# therefore decay with the price level, which is what makes inflation alone produce a
# taxable capital gain. Holding it flat under-taxes every brokerage sale.
def test_basis_decay_tracks_inflation_exactly(base_cfg):
    """With inflation volatility off, the basis deflator is closed-form: after n
    months the surviving real fraction of a frozen dollar is 1/(1+i)^(n/12)."""
    cfg = copy.deepcopy(base_cfg)
    cfg.market.inflation = 0.03
    cfg.market.inflation_volatility = 0.0
    _, _, _, decay = RetirementSimulator(cfg)._market_path(
        360, np.random.default_rng(0))
    assert decay.prod() == pytest.approx(1 / 1.03 ** 30, rel=1e-12)


def test_zero_inflation_leaves_basis_untouched(base_cfg):
    """The decay is driven by inflation alone: at 0% inflation a nominal dollar and
    a real dollar are the same thing, so the basis must not move."""
    cfg = copy.deepcopy(base_cfg)
    cfg.market.inflation = 0.0
    cfg.market.inflation_volatility = 0.0
    _, _, _, decay = RetirementSimulator(cfg)._market_path(
        360, np.random.default_rng(0))
    assert decay.prod() == pytest.approx(1.0, rel=1e-12)


def test_inflation_alone_creates_a_taxable_gain(base_cfg):
    """Known answer to the cent: a brokerage lot bought at par and never sold still
    owes tax 30 years later, purely because inflation eroded its basis.

    $500k brokerage, basis $500k, single filer, 0% real return, 3% inflation, age
    35 -> retire 65 -> die 66 (one plan year). No contributions, dividends, SS or
    healthcare, so the ONLY tax is on the inflation-manufactured gain.

    Mirroring the engine's growth-first convention, the basis has decayed for the
    361 months up to and including the plan month:
      basis = 500,000 / 1.03^(361/12)      = 205,486.5944  -> gain fraction 58.9027%
    A single filer with no ordinary income pays 0% federal LTCG (the gain is far
    inside the $49,450 0% band) and the flat 5% state rate on AGI over the $18,150
    standard deduction (16,100 plus the 2,050 age-65 addition), so the solver's
    gross G satisfies
      G - 0.05*(G*f - 18,150) = 60,000     -> G = 60,885.6644
    leaving an estate of 500,000 - G = 439,114.3356. With a flat real basis the gain
    fraction would be 0%, no tax would be due at all, and the estate would be
    $440,000 -- the $885.66 difference is the bug this pins.
    """
    cfg = deterministic_cfg(base_cfg, real_return=0.0, spend=60_000.0, roth=0.0,
                            retire_age=65, death_age=66)
    cfg.simulation.current_age = 35
    cfg.accounts.brokerage = 500_000.0
    cfg.accounts.brokerage_cost_basis = 500_000.0
    cfg.taxes.filing_status = "single"

    survived, _, _, final, _ = RetirementSimulator(cfg).simulate_life(65, random_seed=0)

    basis = 500_000.0 / 1.03 ** (((65 - 35) * 12 + 1) / 12)
    gain_fraction = 1 - basis / 500_000.0
    gross = (60_000.0 - 0.05 * 18_150.0) / (1 - 0.05 * gain_fraction)
    assert survived is True
    assert final == pytest.approx(500_000.0 - gross, abs=0.01)
    assert final == pytest.approx(439_114.3356, abs=0.01)
    assert final < 440_000.0        # a flat real basis would owe nothing


# ------------------------------------------------------- trajectory sample sizes
# compute_trajectory_percentiles takes column i's percentile over the runs that
# reported a value there, so a trajectory MUST be exactly as long as the retirement
# that life lived -- no shorter (a broke run would vanish from the chart) and no
# longer (a dead run would keep voting in columns it should have left). Getting the
# "longer" case wrong is invisible in a success rate but badly distorts the fan
# chart's low percentiles at old ages, where survivors are scarce.
def test_broke_run_records_zeros_through_its_own_death(base_cfg):
    """$1M at 0% real spending $100k: ten funded years, then broke at 75, death at
    95. The trajectory must carry all 30 retirement years -- ten real balances then
    twenty zeros -- because the household is alive and broke for those twenty."""
    cfg = deterministic_cfg(base_cfg, real_return=0.0, spend=100_000.0,
                            roth=1_000_000.0, retire_age=65, death_age=95)
    survived, _, _, _, traj = RetirementSimulator(cfg).simulate_life(
        65, random_seed=0, record_trajectory=True)
    assert survived is False
    assert len(traj) == 30                      # 95 - 65, not truncated at failure
    for i in range(10):                         # each year: growth, then the draw
        assert traj[i] == pytest.approx(1_000_000.0 - i * 100_000.0 - 100_000.0 / 12,
                                        abs=0.01)
    assert all(v == 0.0 for v in traj[10:])     # alive and broke, not absent


def test_trajectory_length_always_equals_years_lived(base_cfg):
    """The invariant behind every fan-chart percentile, over a stochastic mix of
    successes and failures: a trajectory is exactly one entry per retirement
    anniversary the person lived to see. Death is drawn first from the run's own
    generator, so the expected length is recomputable here from the seed alone."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.current_age = 60
    cfg.accounts.roth = 700_000.0               # thin enough to produce failures
    cfg.accounts.traditional = cfg.accounts.brokerage = cfg.accounts.cash = 0.0
    cfg.accounts.brokerage_cost_basis = 0.0
    cfg.spending.initial_annual_expenses = 65_000.0
    sim = RetirementSimulator(cfg)

    outcomes = set()
    for seed in sim._run_seeds(60, 300):
        death = sim.draw_death_age(cfg.life_events, 60, np.random.default_rng(seed))
        survived, _, _, _, traj = sim.simulate_life(60, random_seed=seed,
                                                    record_trajectory=True)
        outcomes.add(survived)
        living_months = max(int(round((death - 60) * 12)), 1)
        assert len(traj) == -(-living_months // 12)
    assert outcomes == {True, False}            # the sample really covers both


def test_percentile_columns_only_count_the_living(base_cfg):
    """End to end through compute_trajectory_percentiles: each column's reported n
    must equal the number of runs whose trajectory reached that column, and n must
    fall monotonically as the cohort dies off. A dead-and-broke run padded past its
    own death would break the second assertion long before the first."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.current_age = 60
    cfg.simulation.min_retirement_age = cfg.simulation.max_retirement_age = 60
    cfg.accounts.roth = 700_000.0
    cfg.accounts.traditional = cfg.accounts.brokerage = cfg.accounts.cash = 0.0
    cfg.accounts.brokerage_cost_basis = 0.0
    cfg.spending.initial_annual_expenses = 65_000.0
    sim = RetirementSimulator(cfg)

    n_samples = 200
    ages, pcts = sim.compute_trajectory_percentiles(60, n_samples=n_samples)
    counts = pcts["n"]
    expected = np.zeros_like(counts)
    for seed in sim._run_seeds(60, n_samples):
        traj = sim.simulate_life(60, random_seed=seed, record_trajectory=True)[4]
        expected[:min(len(traj), len(counts))] += 1

    assert ages[0] == 60
    assert list(counts) == list(expected)
    assert counts[0] == n_samples                      # everyone is alive at 60
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))


# ---------------------------------------------------------------- the pool
def test_multiprocessing_pool_reproducible(base_cfg):
    """One tiny run through the real Pool path: well-formed results, and exactly
    equal on a second call (fixed seed + common random numbers)."""
    cfg = trinity_cfg(base_cfg, spend=40_000.0)
    cfg.simulation.monte_carlo_runs = 24
    sim = RetirementSimulator(cfg)
    first = sim.retirement_probability(65)
    second = sim.retirement_probability(65)
    assert 0.0 <= first[0] <= 1.0
    assert first == second


def test_pool_workers_compute_the_same_answer_as_serial(base_cfg):
    """The workers run in child processes, so a coverage tool cannot see them --
    this asserts their BEHAVIOUR instead. Every aggregate from the real Pool must
    equal the same computation done serially over the same seeds, which can only
    hold if init_worker built an equivalent simulator and both workers unpacked
    their arguments and returned their tuples correctly."""
    import numpy as np
    cfg = trinity_cfg(base_cfg, spend=40_000.0)
    cfg.simulation.monte_carlo_runs = 32
    sim = RetirementSimulator(cfg)

    pooled = sim.retirement_probability(65)
    seeds = sim._run_seeds(65, cfg.simulation.monte_carlo_runs)
    serial = [sim.simulate_life(65, random_seed=s) for s in seeds]

    assert pooled[0] == pytest.approx(sum(1 for r in serial if r[0]) / len(serial))
    assert pooled[1] == pytest.approx(float(np.median([r[1] for r in serial])))
    assert pooled[2] == pytest.approx(
        float(np.median([r[2] for r in serial if r[2] is not None])))
    assert pooled[3] == pytest.approx(float(np.median([r[3] for r in serial])))

    # and the trajectory worker's separate return shape, likewise
    ages, pcts = sim.compute_trajectory_percentiles(65, n_samples=32)
    serial_traj = [sim.simulate_life(65, random_seed=s, record_trajectory=True)[4]
                   for s in sim._run_seeds(65, 32)]
    expected_n = np.zeros(len(ages), dtype=int)
    for t in serial_traj:
        expected_n[:min(len(t), len(ages))] += 1
    assert list(pcts["n"]) == list(expected_n)


def test_trajectory_percentiles_shape(base_cfg):
    """The fan-chart matrix: one column per retirement year up to death_age_max,
    every run contributing while alive (fixed 30-year horizon here)."""
    cfg = trinity_cfg(base_cfg, spend=40_000.0)
    ages, pcts = RetirementSimulator(cfg).compute_trajectory_percentiles(65, n_samples=24)
    assert list(ages[:2]) == [65, 66]
    assert len(ages) == 30
    assert pcts["n"][0] == 24
    assert all(k in pcts for k in (1, 10, 25, 50))
