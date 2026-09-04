"""Benchmark tests against the retirement-research literature.

The reference points are the Trinity study (Cooley, Hubbard & Walz 1998, updated
through 2014) and Bengen (1994): constant real withdrawals from a stock/bond
portfolio over a fixed 30-year horizon, no taxes or Social Security. Their
headline: 4% initial withdrawals survived 95-100% of HISTORICAL 30-year windows.

What "passing" means here -- and why it is a band, not 95% exactly:

This engine deliberately does NOT replay history. It draws independent fat-tailed
monthly returns with regime switching, which produces more distinct bad sequences
than the single historical path contains (the README documents ~1.3-1.5x the
historical frequency of large one-year losses, and no multi-year mean reversion).
An IID Monte Carlo engine that reported the same success rate as a historical
back-test would in fact be BROKEN -- the correct signature is a result a few
points BELOW Trinity at 4%, converging with it at 3% (where nearly nothing
fails), and tracking it closely at 5-6% (where failures are frequent under both).

The scenarios run on a pinned seed with common random numbers, so every measured
rate below is exactly repeatable: the band widths are review thresholds for
intentional model changes, not allowances for sampling noise. Values observed at
calibration (n=400, seed 12345): 3%: 99.2%, 4%/50-50: 93.8%, 4%/75-25: 92.5%,
5%: 82.8%, 6%: 65.2%, 4%/20y: 99.2%, 4%/40y: 90.5%, 4%/50y/75-25: 88.0%.
"""

import pytest

from tests.conftest import serial_success, trinity_cfg

N = 400


@pytest.fixture(scope="module")
def rates(request):
    """Run every benchmark scenario once and share the rates across tests."""
    # request the base config through the function-scoped fixture machinery once
    from retirement_age_calculator import load_config
    from tests.conftest import BASELINE_PATH
    base = load_config(BASELINE_PATH)
    return {
        "3pct_30y": serial_success(trinity_cfg(base, 30_000), N),
        "4pct_30y": serial_success(trinity_cfg(base, 40_000), N),
        "4pct_30y_75": serial_success(trinity_cfg(base, 40_000, stock_pct=0.75), N),
        "5pct_30y": serial_success(trinity_cfg(base, 50_000), N),
        "6pct_30y": serial_success(trinity_cfg(base, 60_000), N),
        "4pct_20y": serial_success(trinity_cfg(base, 40_000, years=20), N),
        "4pct_40y": serial_success(trinity_cfg(base, 40_000, years=40), N),
        "4pct_50y_75": serial_success(trinity_cfg(base, 40_000, years=50,
                                                  stock_pct=0.75), N),
    }


def test_three_percent_is_nearly_safe(rates):
    """3% over 30 years survives essentially always in every study; here too."""
    assert rates["3pct_30y"] >= 0.97


def test_four_percent_rule_lands_in_the_credible_band(rates):
    """Trinity: 95-100% historically. An IID fat-tail engine belongs a few points
    below -- inside [0.89, 0.98] -- and must NOT reach the historical ceiling."""
    assert 0.89 <= rates["4pct_30y"] <= 0.98
    assert rates["4pct_30y"] < 0.99          # the documented conservatism


def test_five_percent_matches_trinity_neighbourhood(rates):
    """Trinity 50/50 at 5%/30y: ~80%. Band [0.77, 0.89]."""
    assert 0.77 <= rates["5pct_30y"] <= 0.89


def test_six_percent_matches_trinity_neighbourhood(rates):
    """Trinity 50/50 at 6%/30y: ~62-70%. Band [0.57, 0.75]."""
    assert 0.57 <= rates["6pct_30y"] <= 0.75


def test_withdrawal_rate_ordering_is_strict(rates):
    """More spending can only hurt: success falls strictly across 3->4->5->6%."""
    assert (rates["3pct_30y"] > rates["4pct_30y"] > rates["5pct_30y"]
            > rates["6pct_30y"])


def test_horizon_ordering_is_strict(rates):
    """A longer retirement can only be harder at the same withdrawal rate."""
    assert rates["4pct_20y"] > rates["4pct_30y"] > rates["4pct_40y"]


def test_early_retirement_horizon_band(rates):
    """The FIRE case: 4% over 50 years, 75/25. Monte Carlo studies put this in
    the mid-to-high 80s -- distinctly below the same rate over 30 years."""
    assert 0.82 <= rates["4pct_50y_75"] <= 0.94
    assert rates["4pct_50y_75"] < rates["4pct_30y_75"]
