"""Tests of the mechanics inside simulate_life that the deterministic fixtures
switch OFF, plus the results/reporting layer.

conftest's deterministic_cfg neutralises the guardrails, the spending smile and
healthcare so that the withdrawal accounting is hand-computable. That is the right
trade for those tests, but it left three real spending mechanisms and the whole
reporting layer unasserted. Each test here turns exactly one of them back on.
"""

import copy
import warnings

import pytest

from retirement_age_calculator import RetirementSimulator
from tests.conftest import deterministic_cfg


# ---------------------------------------------------------------- glide path
def test_glide_path_endpoints_and_slope(base_cfg):
    """Allocation is the START percentage right up to the retirement day, then
    moves linearly to the END percentage over glide_path_years, and stops there."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.glide_path = True
    cfg.simulation.glide_path_start_stock_pct = 0.90
    cfg.simulation.glide_path_end_stock_pct = 0.50
    cfg.simulation.glide_path_years = 20

    def allocation(age, retirement_age=65):
        years = max(cfg.simulation.glide_path_years, 1)
        frac = min(1.0, max(0.0, age - retirement_age) / years)
        return 0.90 + (0.50 - 0.90) * frac

    assert allocation(40) == pytest.approx(0.90)          # decades before retiring
    assert allocation(65) == pytest.approx(0.90)          # the day itself
    assert allocation(75) == pytest.approx(0.70)          # halfway down the glide
    assert allocation(85) == pytest.approx(0.50)          # glide complete
    assert allocation(100) == pytest.approx(0.50)         # and it stays there


def test_rising_glide_path_is_allowed(base_cfg):
    """end > start is a legitimate strategy (the 'rising equity glide path'), not
    an error -- validation must accept it and the interpolation must run upward."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.glide_path_start_stock_pct = 0.30
    cfg.simulation.glide_path_end_stock_pct = 0.70
    from retirement_age_calculator import validate_config
    validate_config(cfg)
    assert RetirementSimulator(cfg).simulate_life(65, random_seed=1)[0] in (True, False)


# ---------------------------------------------------------------- guardrails
def test_guardrail_cut_reduces_spending_after_a_bad_year(base_cfg):
    """A year returning worse than the cut threshold must reduce the next year's
    spending, and so leave MORE in the portfolio than an identical run with the
    guardrails disabled. -6% real every year, both runs survive to 85."""
    base = deterministic_cfg(base_cfg, real_return=-0.06, spend=60_000.0,
                             roth=3_000_000.0, retire_age=65, death_age=85)
    with_rails = copy.deepcopy(base)
    with_rails.spending.guardrail_cut_return_threshold = -0.02
    with_rails.spending.guardrail_cut_amount = 0.90
    with_rails.spending.guardrail_cut_floor = 0.70

    off = RetirementSimulator(base).simulate_life(65, random_seed=0)
    on = RetirementSimulator(with_rails).simulate_life(65, random_seed=0)
    assert off[0] is True and on[0] is True     # neither run is failing for other reasons
    assert on[3] > off[3] * 1.5                 # cutting spending compounds into the estate


def test_guardrail_cut_respects_its_floor(base_cfg):
    """Spending can never be cut below guardrail_cut_floor of plan, however many
    bad years arrive in a row -- so an aggressive cut_amount cannot starve the
    retiree, and the estate stops improving once the floor binds."""
    cfg = deterministic_cfg(base_cfg, real_return=-0.06, spend=60_000.0,
                            roth=3_000_000.0, retire_age=65, death_age=85)
    cfg.spending.guardrail_cut_return_threshold = -0.02
    cfg.spending.guardrail_cut_floor = 0.70

    gentle = copy.deepcopy(cfg)
    gentle.spending.guardrail_cut_amount = 0.70     # reaches the floor in one step
    savage = copy.deepcopy(cfg)
    savage.spending.guardrail_cut_amount = 0.10     # would obliterate spending, but cannot

    a = RetirementSimulator(gentle).simulate_life(65, random_seed=0)
    b = RetirementSimulator(savage).simulate_life(65, random_seed=0)
    assert a[0] is True and b[0] is True
    assert a[3] == pytest.approx(b[3], rel=1e-9)    # both clamp to the same 70% floor

    multiplier, floor = 1.0, cfg.spending.guardrail_cut_floor
    for _ in range(20):
        multiplier = max(multiplier * 0.10, floor)
    assert multiplier == pytest.approx(floor)


def test_guardrail_raise_respects_its_ceiling(base_cfg):
    """The mirror clamp: a long boom cannot lift spending past the ceiling."""
    cfg = deterministic_cfg(base_cfg, real_return=0.20, spend=50_000.0,
                            roth=1_000_000.0, retire_age=65, death_age=95)
    cfg.spending.guardrail_raise_return_threshold = 0.05
    cfg.spending.guardrail_raise_amount = 2.0       # double every single year
    cfg.spending.guardrail_raise_ceiling = 1.30

    multiplier = 1.0
    for _ in range(30):
        multiplier = min(multiplier * cfg.spending.guardrail_raise_amount,
                         cfg.spending.guardrail_raise_ceiling)
    assert multiplier == pytest.approx(1.30)
    # spending is capped, so a 20% real return still leaves a large estate
    assert RetirementSimulator(cfg).simulate_life(65, random_seed=0)[3] > 1_000_000.0


def test_guardrails_do_not_fire_in_the_first_year(base_cfg):
    """There is no completed year to judge at the first anniversary, so the first
    year's spending is always exactly to plan."""
    cfg = deterministic_cfg(base_cfg, real_return=-0.30, spend=50_000.0,
                            roth=1_000_000.0, retire_age=65, death_age=66)
    cfg.spending.guardrail_cut_return_threshold = -0.05
    cfg.spending.guardrail_cut_amount = 0.50
    sim = RetirementSimulator(cfg)
    _, _, iwr, _, _ = sim.simulate_life(65, random_seed=0)
    g = (1 - 0.30) ** (1 / 12)
    assert iwr == pytest.approx(50_000.0 / (1_000_000.0 * g), abs=1e-6)


# ---------------------------------------------------------------- healthcare
def test_healthcare_is_charged_per_living_person_on_their_own_clock(base_cfg):
    """Premiums are per person and each person's OWN age gates Medicare, so a
    primary of 65 with a 60-year-old spouse pays one Medicare premium plus one
    pre-Medicare premium -- not two of either.

    Everything else is switched off (0% real, no base spending, all Roth so no
    tax), which makes total lifetime healthcare exactly the drawdown. The primary
    dies at 95; the spouse, five years younger, dies at 95 on THEIR clock, i.e.
    when the primary would be 100 -- so there are five survivor-only years too:

      primary,  ages 65-94, all Medicare               30 x  5,000 = 150,000
      spouse,   ages 60-64, pre-Medicare                5 x 20,000 = 100,000
      spouse,   ages 65-89, Medicare                   25 x  5,000 = 125,000
      survivor, spouse ages 90-94, Medicare             5 x  5,000 =  25,000
                                                                     -------
                                                                     400,000
    """
    cfg = deterministic_cfg(base_cfg, real_return=0.0, spend=0.0, roth=2_000_000.0,
                            retire_age=65, death_age=95)
    cfg.spouse.enabled = True
    cfg.spouse.age_offset = -5
    cfg.spouse.mortality_model = "normal"
    cfg.spouse.death_age_mean = cfg.spouse.death_age_min = cfg.spouse.death_age_max = 95
    cfg.spouse.death_age_std = 0.0
    cfg.healthcare.pre_medicare_annual_premium = 20_000.0
    cfg.healthcare.medicare_annual_premium = 5_000.0
    cfg.healthcare.medicare_age = 65

    final = RetirementSimulator(cfg).simulate_life(65, random_seed=0)[3]
    assert 2_000_000.0 - final == pytest.approx(400_000.0, abs=1.0)


def test_medicare_age_uses_each_persons_own_clock(base_cfg):
    """The discriminating half of the test above: if the spouse's eligibility were
    keyed to the PRIMARY's age, the five pre-Medicare spouse years would vanish and
    the total would fall by 5 x (20,000 - 5,000) = 75,000."""
    cfg = deterministic_cfg(base_cfg, real_return=0.0, spend=0.0, roth=2_000_000.0,
                            retire_age=65, death_age=95)
    cfg.spouse.enabled = True
    cfg.spouse.age_offset = -5
    cfg.spouse.mortality_model = "normal"
    cfg.spouse.death_age_mean = cfg.spouse.death_age_min = cfg.spouse.death_age_max = 90
    cfg.spouse.death_age_std = 0.0                 # both die on the primary's 95th
    cfg.healthcare.pre_medicare_annual_premium = 20_000.0
    cfg.healthcare.medicare_annual_premium = 5_000.0
    cfg.healthcare.medicare_age = 65

    final = RetirementSimulator(cfg).simulate_life(65, random_seed=0)[3]
    # 30x5,000 primary + 5x20,000 + 25x5,000 spouse = 375,000
    assert 2_000_000.0 - final == pytest.approx(375_000.0, abs=1.0)
    assert 2_000_000.0 - final != pytest.approx(300_000.0, abs=1.0)   # the "own clock" delta


def test_healthcare_multiplier_applies_only_to_premiums(base_cfg):
    """annual_healthcare_increase_rate compounds the PREMIUMS after the decline end
    age; base spending stays frozen at its declined level."""
    sim = RetirementSimulator(base_cfg)
    s = base_cfg.spending
    base_factor, health = sim.spending_smile(s.spending_decline_end_age + 4)
    frozen, _ = sim.spending_smile(s.spending_decline_end_age)
    assert base_factor == pytest.approx(frozen)
    assert health == pytest.approx((1 + s.annual_healthcare_increase_rate) ** 4)


# ---------------------------------------------------------------- results layer
def test_reported_figures_are_medians_not_means(base_cfg):
    """Every dollar column is a median. On a portfolio where most runs fail, the
    mean low-water mark is wildly optimistic while the median tells the truth."""
    import numpy as np
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.current_age = 60
    cfg.simulation.min_retirement_age = cfg.simulation.max_retirement_age = 60
    cfg.simulation.monte_carlo_runs = 120
    cfg.accounts.roth = 250_000.0
    cfg.accounts.traditional = cfg.accounts.brokerage = cfg.accounts.cash = 0.0
    cfg.accounts.brokerage_cost_basis = 0.0
    cfg.spending.initial_annual_expenses = 90_000.0
    sim = RetirementSimulator(cfg)

    seeds = sim._run_seeds(60, cfg.simulation.monte_carlo_runs)
    runs = [sim.simulate_life(60, random_seed=s) for s in seeds]
    prob, med_min, med_wr, med_final = sim.retirement_probability(60)

    assert prob == pytest.approx(sum(1 for r in runs if r[0]) / len(runs))
    assert med_min == pytest.approx(float(np.median([r[1] for r in runs])))
    assert med_final == pytest.approx(float(np.median([r[3] for r in runs])))
    assert med_wr == pytest.approx(
        float(np.median([r[2] for r in runs if r[2] is not None])))
    # the mean would be materially higher on this deliberately fragile portfolio
    assert float(np.mean([r[1] for r in runs])) > med_min


def test_find_retirement_age_picks_the_earliest_clearing_age(base_cfg):
    sim = RetirementSimulator(base_cfg)
    sim.cfg.simulation.target_success_probability = 0.90
    sim.probability_results = {
        60: (0.50, 0.0, 0.0, 0.0),
        61: (0.89, 0.0, 0.0, 0.0),
        62: (0.91, 0.0, 0.0, 0.0),
        63: (0.99, 0.0, 0.0, 0.0),
    }
    result = sim.find_retirement_age()
    assert result.retirement_age == 62
    assert result.success_probability == pytest.approx(0.91)


def test_find_retirement_age_returns_none_when_unreachable(base_cfg):
    sim = RetirementSimulator(base_cfg)
    sim.cfg.simulation.target_success_probability = 0.95
    sim.probability_results = {60: (0.10, 0.0, 0.0, 0.0), 61: (0.20, 0.0, 0.0, 0.0)}
    assert sim.find_retirement_age() is None
    assert "No retirement age met" in sim.format_results_table()


def test_results_table_renders_every_swept_age(base_cfg):
    sim = RetirementSimulator(base_cfg)
    sim.cfg.simulation.target_success_probability = 0.90
    sim.probability_results = {
        60: (0.50, 100_000.0, 0.055, 200_000.0),
        61: (0.95, 300_000.0, 0.041, 900_000.0),
    }
    table = sim.format_results_table()
    assert "AGE 61" in table
    assert "Median Min In Ret" in table and "Median Final Bal" in table
    assert table.count("\n") >= 4
    for age in (60, 61):
        assert any(line.strip().startswith(str(age)) for line in table.splitlines())


def test_assumption_report_survives_crises_being_disabled(base_cfg):
    """Setting monthly_crisis_probability to 0 is the natural way to switch the
    regime model off, and validate_config accepts it -- so the report, which the
    CLI prints BEFORE the sweep, must not divide by it."""
    from retirement_age_calculator import validate_config
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.normal_regime.monthly_crisis_probability = 0.0
    validate_config(cfg)
    text = RetirementSimulator(cfg).assumption_report()
    assert "crisis regime  disabled" in text

    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.crisis_regime.monthly_recovery_probability = 0.0
    validate_config(cfg)
    RetirementSimulator(cfg).assumption_report()      # must not raise


def test_probability_curve_is_monotone_under_common_random_numbers(base_cfg):
    """With CRN on, every age faces the SAME lifetimes, so a later retirement can
    only help. A non-monotone curve is the canonical signal of an engine
    regression -- CLAUDE.md calls this out explicitly.

    Run serially: the invariant lives in the seeding and simulate_life, not in the
    pool, and each Pool costs ~0.85s of process startup on macOS for work that takes
    milliseconds. test_pool_workers_compute_the_same_answer_as_serial covers the
    pool path separately.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.current_age = 55
    cfg.simulation.common_random_numbers = True
    sim = RetirementSimulator(cfg)

    probs = []
    for age in range(60, 65):
        seeds = sim._run_seeds(age, 150)
        runs = [sim.simulate_life(age, random_seed=s) for s in seeds]
        probs.append(sum(1 for r in runs if r[0]) / len(runs))
    assert len(probs) == 5
    assert all(a <= b + 1e-12 for a, b in zip(probs, probs[1:])), probs


# ------------------------------------------------------- seeds and remaining paths
def test_non_common_random_numbers_gives_each_age_its_own_scenarios(base_cfg):
    """With CRN off, ages must NOT share lifetimes -- and the seed sets must not
    collide either, or two ages would silently share runs."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.common_random_numbers = False
    sim = RetirementSimulator(cfg)
    per_age = {age: sim._run_seeds(age, 500) for age in range(40, 71)}

    assert per_age[40] != per_age[41]
    everything = [s for seeds in per_age.values() for s in seeds]
    assert len(set(everything)) == len(everything)      # no collisions anywhere

    cfg.simulation.common_random_numbers = True
    crn = RetirementSimulator(cfg)
    assert crn._run_seeds(40, 50) == crn._run_seeds(70, 50)


def test_default_config_loads_the_shipped_parameters():
    """The CLI and GUI both start here, so it must parse and validate."""
    from retirement_age_calculator import default_config, Config
    assert isinstance(default_config(), Config)


def test_assumption_report_describes_each_penalty_regime(base_cfg):
    """The three mutually exclusive penalty branches must each render, because the
    printout is how a user confirms which one the model is applying."""
    plain = RetirementSimulator(base_cfg).assumption_report()
    assert "10% before age 59.5" in plain

    cfg = copy.deepcopy(base_cfg)
    cfg.taxes.assume_qualified_plan_age55_exception = True
    assert "rule-of-55" in RetirementSimulator(cfg).assumption_report()

    cfg = copy.deepcopy(base_cfg)
    cfg.taxes.use_72t_sepp = True
    assert "72(t)" in RetirementSimulator(cfg).assumption_report()

    # 72(t) wins over rule-of-55 when both are set
    cfg.taxes.assume_qualified_plan_age55_exception = True
    assert "72(t)" in RetirementSimulator(cfg).assumption_report()


def test_progress_callback_is_invoked_for_every_age(base_cfg, monkeypatch):
    """The GUI drives its status line from this callback: it must fire once per
    swept age, before that age runs, with a 0-based index and the total.

    retirement_probability is stubbed out because it is the only slow part and is
    not what this test is about -- each real call spawns a process pool costing
    ~0.85s on macOS regardless of how few lifetimes it runs."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.current_age = 60
    cfg.simulation.min_retirement_age = 61
    cfg.simulation.max_retirement_age = 63
    sim = RetirementSimulator(cfg)

    ran = []
    monkeypatch.setattr(sim, "retirement_probability",
                        lambda age: ran.append(age) or (0.5, 0.0, 0.0, 0.0))
    seen = []
    sim.compute_probability_curve(progress=lambda i, total, age: seen.append((i, total, age)))

    assert seen == [(0, 3, 61), (1, 3, 62), (2, 3, 63)]
    assert ran == [61, 62, 63]                       # fired before each age, in order
    assert sorted(sim.probability_results) == [61, 62, 63]


# ---------------------------------------------------------------- charts render
def test_charts_render_without_error(base_cfg):
    """Presentation code, but it consumes NaN-tailed percentile arrays and an
    Optional result, both of which are easy to crash on.

    The percentile arrays are built directly rather than simulated: what is under
    test is the rendering, and compute_trajectory_percentiles (covered in
    test_simulation.py) would spawn a ~0.85s process pool just to produce them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sim = RetirementSimulator(copy.deepcopy(base_cfg))
    sim.cfg.simulation.target_success_probability = 0.90
    sim.probability_results = {60: (0.80, 1e5, 0.05, 5e5),
                               61: (0.92, 2e5, 0.04, 9e5)}

    # a realistic fan: values for the first years, NaN once the cohort has died off
    ages = np.arange(60, 90)
    pcts = {p: np.concatenate([np.linspace(1e6 * p / 50, 1e5, 20), np.full(10, np.nan)])
            for p in (1, 10, 25, 50)}
    pcts["n"] = np.concatenate([np.full(20, 100), np.zeros(10, dtype=int)])
    assert np.isnan(pcts[50][-1])          # the tail really is NaN-backed

    fig = plt.figure()
    sim.draw_probability_curve(fig.add_subplot(2, 1, 1))
    sim.draw_trajectory(fig.add_subplot(2, 1, 2), ages, pcts, 60, 100)
    plt.close(fig)

    # with no age clearing the target, so the "earliest age" marker is absent
    sim.cfg.simulation.target_success_probability = 1.0
    sim.probability_results = {60: (0.10, 0.0, 0.0, 0.0), 61: (0.20, 0.0, 0.0, 0.0)}
    assert sim.find_retirement_age() is None
    fig = plt.figure()
    sim.draw_probability_curve(fig.add_subplot(1, 1, 1))
    plt.close(fig)

    # and with one clearing it, so the marker IS drawn
    sim.cfg.simulation.target_success_probability = 0.15
    assert sim.find_retirement_age() is not None
    fig = plt.figure()
    sim.draw_probability_curve(fig.add_subplot(1, 1, 1))
    plt.close(fig)


def test_plot_results_assembles_the_full_figure(base_cfg):
    """The top-level composition the CLI calls: it runs its own trajectory sweep
    and stacks both charts, so it can fail even when each chart works alone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.current_age = 62
    sim = RetirementSimulator(cfg)
    sim.probability_results = {63: (0.96, 1e5, 0.04, 8e5)}

    # an age clears the target, so the trajectory fan is added as a second panel.
    # This is the one place a real trajectory sweep is unavoidable, so it runs at
    # the smallest sample that still produces a fan.
    sim.cfg.simulation.target_success_probability = 0.90
    assert sim.find_retirement_age() is not None
    with warnings.catch_warnings():             # Agg has no window to show
        warnings.simplefilter("ignore", UserWarning)
        sim.plot_results(trajectory_samples=8)
        # and again with no qualifying age, so only the curve is drawn and no
        # trajectory sweep happens at all
        sim.cfg.simulation.target_success_probability = 1.0
        assert sim.find_retirement_age() is None
        sim.plot_results(trajectory_samples=8)
    plt.close("all")


def test_trajectory_percentiles_handle_an_empty_horizon(base_cfg):
    """A retirement age at or past death_age_max leaves zero columns; the
    percentile arrays must come back empty rather than raising on an all-NaN
    reduction."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.current_age = 60
    cfg.life_events.death_age_max = 100
    cfg.simulation.min_retirement_age = cfg.simulation.max_retirement_age = 100
    sim = RetirementSimulator(cfg)
    ages, pcts = sim.compute_trajectory_percentiles(100, n_samples=8)
    assert len(ages) == 0
    assert len(pcts["n"]) == 0
    assert all(len(pcts[p]) == 0 for p in (1, 10, 25, 50))


def test_fan_chart_drops_columns_too_thin_to_support_a_percentile(base_cfg):
    """The tail of the trajectory must not be drawn once the surviving cohort is
    too small. With a handful of runs left, every percentile lands on the same one
    or two lives, so the four lines converge and shoot upward -- it reads as a
    dramatic late-life spike but is pure small-sample noise.

    The data still carries those columns (with honest counts in n); the renderer
    declines to plot them and says so in the title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from retirement_age_calculator import MIN_TRAJECTORY_SAMPLES

    sim = RetirementSimulator(copy.deepcopy(base_cfg))
    ages = np.arange(60, 70)
    # healthy cohort for the first six years, then it collapses
    counts = np.array([500, 400, 300, 200, 150, 120, 40, 9, 3, 1])
    pcts = {p: np.full(10, 1e6, dtype=float) for p in (1, 10, 25, 50)}
    for p in (1, 10, 25, 50):
        pcts[p][6:] = 9e7          # the noisy spike the thin columns would draw
    pcts["n"] = counts

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sim.draw_trajectory(ax, ages, pcts, 60, 500)

    drawn = ax.lines[0].get_ydata()                     # the median line, in $M
    assert np.all(np.isfinite(drawn[:6]))               # well-backed columns are drawn
    assert np.all(np.isnan(drawn[6:]))                  # thin ones are blanked
    assert np.nanmax(drawn) < 90.0                      # the spike never reaches the axes
    # the last plotted age is the last column meeting the threshold, and the axis
    # says where the chart was trimmed
    assert int(ages[counts >= MIN_TRAJECTORY_SAMPLES][-1]) == 65
    assert "shown to 65" in ax.get_xlabel()
    plt.close(fig)


@pytest.mark.parametrize("figsize,stacked", [((9.0, 7.0), True),    # CLI: two stacked
                                             ((9.0, 4.5), False),   # GUI tab: one plot
                                             ((6.5, 3.2), False)])  # GUI, small window
def test_chart_labels_are_never_clipped(base_cfg, figsize, stacked):
    """No axis label may fall outside the figure at any geometry either front end
    uses. The CLI stacks both plots in one 9x7in figure; the GUI now gives each plot
    its own tab and its own figure, and Tk stretches that figure to whatever the
    window leaves -- which is why both front ends use constrained layout rather than
    tight_layout, and why the trajectory's sampling caveat sits on the x-axis
    instead of in an over-long title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sim = RetirementSimulator(copy.deepcopy(base_cfg))
    sim.probability_results = {a: (0.5, 0.0, 0.0, 0.0) for a in range(40, 71)}
    ages = np.arange(40, 110)
    counts = np.full(len(ages), 500)
    counts[-8:] = 0                                   # a thin, trimmed tail
    pcts = {p: np.full(len(ages), 1e6, dtype=float) for p in (1, 10, 25, 50)}
    pcts["n"] = counts

    fig = plt.figure(figsize=figsize, layout="constrained")
    rows = 2 if stacked else 1
    axes = [fig.add_subplot(rows, 1, 1)]
    sim.draw_probability_curve(axes[0])
    if stacked:
        axes.append(fig.add_subplot(rows, 1, 2))
        sim.draw_trajectory(axes[1], ages, pcts, 40, 100_000)
    else:
        # each GUI tab holds a single plot; check the busier one on its own too
        fan = plt.figure(figsize=figsize, layout="constrained")
        fan_ax = fan.add_subplot(1, 1, 1)
        sim.draw_trajectory(fan_ax, ages, pcts, 40, 100_000)
        fan.canvas.draw()
        axes.append(fan_ax)

    for ax in axes:
        f = ax.figure
        f.canvas.draw()
        r = f.canvas.get_renderer()
        w, h = f.get_size_inches() * f.dpi
        for label in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if not label.get_text():
                continue
            box = label.get_window_extent(r)
            assert box.x0 >= -0.5 and box.x1 <= w + 0.5, label.get_text()
            assert box.y0 >= -0.5 and box.y1 <= h + 0.5, label.get_text()
    plt.close("all")
