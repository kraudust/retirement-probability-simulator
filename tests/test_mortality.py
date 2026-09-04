"""Distributional tests of the mortality models.

The Gompertz curves are calibrated so survival CONDITIONAL ON REACHING 65 matches
SSA cohort life-table targets (2025 Trustees Report, intermediate assumptions).
The draws below use a fixed Generator, so the measured survival rates are exactly
repeatable -- the tolerances exist for future recalibration headroom, not noise.
"""

import copy

import numpy as np
import pytest

from retirement_age_calculator import RetirementSimulator

N_DRAWS = 40_000

# (sex, survival targets from 65 to [80, 90, 95, 100])
CALIBRATION_TARGETS = [
    ("male", (0.70, 0.33, 0.15, 0.05)),
    ("female", (0.78, 0.44, 0.24, 0.085)),
]


@pytest.fixture()
def sim(base_cfg):
    """Simulator only for its mortality methods; config content is irrelevant."""
    return RetirementSimulator(base_cfg)


@pytest.mark.parametrize("sex,targets", CALIBRATION_TARGETS)
def test_gompertz_matches_ssa_targets(sim, sex, targets):
    """Survival from 65 must sit within 3.5pp of each SSA cohort target."""
    rng = np.random.default_rng(7)
    draws = np.array([sim._actuarial_death_age(sex, 65.0, 120.0, rng)
                      for _ in range(N_DRAWS)])
    for age, target in zip((80, 90, 95, 100), targets):
        measured = float(np.mean(draws >= age))
        assert measured == pytest.approx(target, abs=0.035), \
            f"{sex} survival 65->{age}: {measured:.3f} vs target {target}"


def test_gompertz_median_lifespans(sim):
    """Median death from 65: ~86 for males, ~89 for females (the values quoted in
    the GOMPERTZ_PARAMS documentation)."""
    rng = np.random.default_rng(7)
    for sex, med in (("male", 86.0), ("female", 89.0)):
        draws = [sim._actuarial_death_age(sex, 65.0, 120.0, rng) for _ in range(N_DRAWS)]
        assert float(np.median(draws)) == pytest.approx(med, abs=1.0)


def test_actuarial_respects_max_age(sim, base_cfg):
    """The Gompertz draw is capped at the person's death_age_max."""
    rng = np.random.default_rng(3)
    draws = [sim.draw_death_age(base_cfg.life_events, 35, rng) for _ in range(2_000)]
    assert max(draws) <= base_cfg.life_events.death_age_max
    assert min(draws) >= 35.0


def test_normal_model_clipped(sim, base_cfg):
    """The clipped-normal model stays inside [death_age_min, death_age_max]."""
    le = copy.deepcopy(base_cfg.life_events)
    le.mortality_model = "normal"
    rng = np.random.default_rng(3)
    draws = [sim.draw_death_age(le, 35, rng) for _ in range(2_000)]
    assert min(draws) >= le.death_age_min
    assert max(draws) <= le.death_age_max


def test_zero_std_normal_is_deterministic(sim, base_cfg):
    """std=0 pins death to the mean exactly -- the deterministic scenarios rely
    on this to build fixed-horizon retirements."""
    le = copy.deepcopy(base_cfg.life_events)
    le.mortality_model = "normal"
    le.death_age_mean = 95
    le.death_age_std = 0.0
    le.death_age_min = 95
    le.death_age_max = 95
    rng = np.random.default_rng(3)
    assert all(sim.draw_death_age(le, 35, rng) == 95.0 for _ in range(10))
