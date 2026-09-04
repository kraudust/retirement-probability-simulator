"""Tests of the config layer: schema binding, validation, dotted-path access,
and the YAML round trip.

Why this file exists: validate_config is the only thing standing between a
mistyped parameter and a quietly wrong retirement age, and most of its rules had
no test at all -- a rule that never fires in the suite is a rule nobody has
checked. Each test below asserts that a specific bad value is REJECTED, so a
future refactor that drops a check fails here instead of silently shipping.

The most important case is the bracket ordering: _bracket_tax walks the pairs in
order, so a shuffled table does not raise, it just returns the wrong tax.
"""

import copy

import pytest
import yaml

from retirement_age_calculator import (Config, ValidationError, config_from_dict,
                                       config_to_dict, get_field, load_config,
                                       save_config, set_field, validate_config)


def rejects(cfg, fragment):
    """Assert validate_config raises and names the offending field."""
    with pytest.raises(ValidationError) as exc:
        validate_config(cfg)
    assert fragment in str(exc.value)


# ---------------------------------------------------------------- the schema
def test_baseline_is_valid(base_cfg):
    """The frozen baseline must itself pass every rule."""
    validate_config(base_cfg)


def test_missing_key_names_the_field(base_cfg):
    """config_from_dict expands each section with **, so a missing or misspelled
    YAML key raises TypeError NAMING it -- the property the whole schema rests on."""
    raw = config_to_dict(base_cfg)
    del raw["market"]["inflation"]
    with pytest.raises(TypeError, match="inflation"):
        config_from_dict(raw)


def test_unknown_key_is_rejected(base_cfg):
    raw = config_to_dict(base_cfg)
    raw["market"]["inflaton"] = 0.03           # typo
    with pytest.raises(TypeError, match="inflaton"):
        config_from_dict(raw)


def test_validation_reports_every_problem_at_once(base_cfg):
    """The error lists all failures, not just the first -- fixing a config one
    exception at a time is miserable."""
    cfg = copy.deepcopy(base_cfg)
    cfg.accounts.roth = -1.0
    cfg.simulation.monte_carlo_runs = 0
    cfg.spending.initial_annual_expenses = -5.0
    with pytest.raises(ValidationError) as exc:
        validate_config(cfg)
    message = str(exc.value)
    assert "accounts.roth" in message
    assert "monte_carlo_runs" in message
    assert "initial_annual_expenses" in message


# ------------------------------------------------- tax tables (silent-wrong risk)
def test_unsorted_brackets_are_rejected(base_cfg):
    """THE case that motivated this file. _bracket_tax assumes ascending upper
    bounds; a shuffled 2026 single table taxes $60k at $16,520 instead of $7,912
    and raises nothing, so validation has to catch it."""
    cfg = copy.deepcopy(base_cfg)
    cfg.taxes.federal_brackets["single"] = [[50_400, 0.12], [12_400, 0.10], [None, 0.22]]
    rejects(cfg, "upper bounds must increase")


def test_bracket_rate_out_of_range_is_rejected(base_cfg):
    for bad in (-0.5, 3.0):
        cfg = copy.deepcopy(base_cfg)
        cfg.taxes.federal_brackets["single"][0] = [12_400, bad]
        rejects(cfg, "rates must be between 0 and 1")


def test_null_bound_only_on_the_top_bracket(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.taxes.federal_brackets["single"][0] = [None, 0.10]
    rejects(cfg, "null upper bound on the LAST bracket")

    cfg = copy.deepcopy(base_cfg)
    cfg.taxes.ltcg_brackets["single"] = [[49_450, 0.0], [545_500, 0.15], [900_000, 0.20]]
    rejects(cfg, "must end with a null-bounded top bracket")


def test_missing_status_row_is_rejected(base_cfg):
    """Every per-status table must cover the configured status AND single, because
    a surviving spouse switches to single mid-simulation."""
    for table in ("standard_deductions", "additional_standard_deductions_65plus",
                  "federal_brackets", "ltcg_brackets", "ss_provisional_thresholds",
                  "niit_thresholds"):
        cfg = copy.deepcopy(base_cfg)
        getattr(cfg.taxes, table).pop("single")
        rejects(cfg, f"taxes.{table} is missing an entry for 'single'")


def test_negative_deductions_are_rejected(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.taxes.standard_deductions["single"] = -1
    rejects(cfg, "standard_deductions[single] must be non-negative")

    cfg = copy.deepcopy(base_cfg)
    cfg.taxes.additional_standard_deductions_65plus["single"] = -1
    rejects(cfg, "additional_standard_deductions_65plus[single] must be non-negative")


# ---------------------------------------------------------------- market inputs
@pytest.mark.parametrize("field", ["stock_return", "bond_return", "cash_return",
                                   "inflation"])
def test_rates_at_or_below_minus_one_are_rejected(base_cfg, field):
    """real_return divides by (1 + inflation) and the risky legs take log1p of the
    real return, so -1 is a ZeroDivisionError/domain error deep in __init__ unless
    it is caught here."""
    cfg = copy.deepcopy(base_cfg)
    setattr(cfg.market, field, -1.0)
    rejects(cfg, f"market.{field} must be greater than -1")


def test_extreme_inflation_volatility_is_rejected(base_cfg):
    """Cash subtracts the inflation shock arithmetically, so a big enough sigma can
    push a month past -100% and drive the balance negative."""
    cfg = copy.deepcopy(base_cfg)
    cfg.market.inflation_volatility = 1.0
    rejects(cfg, "inflation_volatility above 0.25")


def test_correlation_outside_unit_interval_is_rejected(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.market.stock_bond_correlation = -1.5
    rejects(cfg, "stock_bond_correlation")


def test_degrees_of_freedom_must_give_finite_variance(base_cfg):
    """A Student-t needs df > 2 or the variance -- and so the volatility scaling --
    is undefined."""
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.return_distribution_degrees_of_freedom = 2
    rejects(cfg, "degrees_of_freedom")


# ---------------------------------------------------------------- ages and ranges
def test_implausible_ages_are_rejected(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.current_age = -10
    rejects(cfg, "current_age must be between 0 and 120")

    cfg = copy.deepcopy(base_cfg)
    cfg.life_events.death_age_max = 500
    rejects(cfg, "death_age_max must be above current_age and at most 120")

    cfg = copy.deepcopy(base_cfg)
    cfg.spouse.enabled = True
    cfg.spouse.age_offset = -80
    rejects(cfg, "age_offset must be within 50 years")


def test_retirement_range_must_be_ordered(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.min_retirement_age = cfg.simulation.current_age - 1
    rejects(cfg, "min_retirement_age must be >= current_age")

    cfg = copy.deepcopy(base_cfg)
    cfg.simulation.max_retirement_age = cfg.simulation.min_retirement_age - 1
    rejects(cfg, "max_retirement_age must be >= min_retirement_age")


def test_claim_age_outside_62_70_is_rejected(base_cfg):
    for bad in (61, 71):
        cfg = copy.deepcopy(base_cfg)
        cfg.life_events.ss_claim_age = bad
        rejects(cfg, "ss_claim_age must be between 62 and 70")


def test_spouse_rules_are_skipped_when_disabled(base_cfg):
    """A nonsense spouse block must not block a single-person run."""
    cfg = copy.deepcopy(base_cfg)
    cfg.spouse.enabled = False
    cfg.spouse.ss_claim_age = 12
    cfg.spouse.mortality_sex = "nonsense"
    validate_config(cfg)


def test_basis_above_balance_is_rejected(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.accounts.brokerage = 1_000.0
    cfg.accounts.brokerage_cost_basis = 2_000.0
    rejects(cfg, "brokerage_cost_basis")


def test_allocation_percentages_must_be_fractions(base_cfg):
    for field in ("glide_path_start_stock_pct", "glide_path_end_stock_pct",
                  "static_stock_allocation"):
        cfg = copy.deepcopy(base_cfg)
        setattr(cfg.simulation, field, 1.5)
        rejects(cfg, field)


# ---------------------------------------------------------------- dotted paths
def test_get_and_set_field_round_trip(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    assert get_field(cfg, "accounts.roth") == cfg.accounts.roth
    set_field(cfg, "accounts.roth", 12_345.0)
    assert cfg.accounts.roth == 12_345.0


def test_set_field_coerces_to_the_declared_type(base_cfg):
    """The GUI hands over strings from text entries; the dataclass type decides."""
    cfg = copy.deepcopy(base_cfg)
    set_field(cfg, "accounts.roth", "250000")
    assert isinstance(cfg.accounts.roth, float) and cfg.accounts.roth == 250_000.0
    set_field(cfg, "simulation.monte_carlo_runs", "500.7")
    assert cfg.simulation.monte_carlo_runs == 500        # int(float(...)), truncating
    set_field(cfg, "simulation.glide_path", False)
    assert cfg.simulation.glide_path is False
    # string-typed fields pass through untouched and are caught by validate_config
    set_field(cfg, "taxes.filing_status", "single")
    assert cfg.taxes.filing_status == "single"
    set_field(cfg, "life_events.mortality_sex", "female")
    assert cfg.life_events.mortality_sex == "female"
    validate_config(cfg)


def test_set_field_reaches_nested_sections(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    set_field(cfg, "simulation.crisis_regime.annual_return_drag", "-0.2")
    assert cfg.simulation.crisis_regime.annual_return_drag == -0.2
    assert get_field(cfg, "simulation.crisis_regime.annual_return_drag") == -0.2


def test_unknown_path_raises(base_cfg):
    for path in ("accounts.not_a_field", "simulation.normal_regime.nope"):
        with pytest.raises(AttributeError, match="no such config field"):
            get_field(base_cfg, path)


# ---------------------------------------------------------------- YAML round trip
def test_save_load_round_trip_is_lossless(base_cfg, tmp_path):
    """Every field must survive save -> load unchanged; a dropped key here would
    silently reset a parameter to whatever the YAML happened to hold."""
    path = tmp_path / "round_trip.yaml"
    save_config(base_cfg, str(path))
    reloaded = load_config(str(path))
    assert config_to_dict(reloaded) == config_to_dict(base_cfg)


def test_save_refuses_to_persist_an_invalid_config(base_cfg, tmp_path):
    cfg = copy.deepcopy(base_cfg)
    cfg.accounts.roth = -1.0
    path = tmp_path / "bad.yaml"
    with pytest.raises(ValidationError):
        save_config(cfg, str(path))
    assert not path.exists()


def test_shipped_params_file_matches_the_schema():
    """simulation_params.yaml is what the CLI and GUI actually load, and it is
    edited by hand -- it must stay in sync with the dataclasses."""
    from retirement_age_calculator import DEFAULT_CONFIG_PATH
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg, Config)


def test_baseline_and_shipped_params_have_the_same_keys():
    """The frozen test baseline must not drift from the shipped file's SHAPE, or
    the suite stops testing the config people actually run."""
    from retirement_age_calculator import DEFAULT_CONFIG_PATH
    with open(DEFAULT_CONFIG_PATH) as f:
        shipped = yaml.safe_load(f)
    with open("tests/baseline_config.yaml") as f:
        baseline = yaml.safe_load(f)

    def shape(node):
        if isinstance(node, dict):
            return {k: shape(v) for k, v in sorted(node.items())}
        return type(node).__name__ if not isinstance(node, list) else "list"

    assert shape(shipped) == shape(baseline)


# --------------------------------------------------- every remaining rule fires
# A validation rule that never fires in the suite is a rule nobody has checked.
# Each entry mutates exactly one field to an illegal value and names a fragment of
# the message it must produce, so every `errors.append` in validate_config is
# proven reachable and correctly worded.
def _set(section, field, value):
    def mutate(cfg):
        setattr(getattr(cfg, section), field, value)
    return mutate


def _nested(path, value):
    def mutate(cfg):
        obj = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
    return mutate


def _enable_spouse(inner):
    def mutate(cfg):
        cfg.spouse.enabled = True
        inner(cfg)
    return mutate


RULES = [
    # accounts and contributions
    ("contribution negative", _set("contributions", "annual_roth", -1.0),
     "contributions.annual_roth must be non-negative"),
    ("contribution growth <= -1",
     _set("contributions", "annual_contribution_growth_rate", -1.0),
     "annual_contribution_growth_rate must be > -1"),
    # simulation sweep
    ("target probability zero", _set("simulation", "target_success_probability", 0.0),
     "target_success_probability must be in (0, 1]"),
    ("target probability above 1",
     _set("simulation", "target_success_probability", 1.5),
     "target_success_probability must be in (0, 1]"),
    ("glide path years zero", _set("simulation", "glide_path_years", 0),
     "glide_path_years must be at least 1"),
    ("rmd start age too low", _set("simulation", "rmd_start_age", 60),
     "rmd_start_age must be between 72 and 100"),
    ("crisis probability above 1",
     _nested("simulation.normal_regime.monthly_crisis_probability", 1.5),
     "monthly_crisis_probability must be between 0 and 1"),
    ("recovery probability negative",
     _nested("simulation.crisis_regime.monthly_recovery_probability", -0.1),
     "monthly_recovery_probability must be between 0 and 1"),
    ("crisis drag <= -1",
     _nested("simulation.crisis_regime.annual_return_drag", -1.0),
     "annual_return_drag must be greater than -1"),
    ("normal boost <= -1", _nested("simulation.normal_regime.return_boost", -1.0),
     "return_boost must be greater than -1"),
    ("crisis vol multiplier negative",
     _nested("simulation.crisis_regime.volatility_multiplier", -1.0),
     "crisis_regime.volatility_multiplier must be non-negative"),
    ("normal vol multiplier negative",
     _nested("simulation.normal_regime.volatility_multiplier", -1.0),
     "normal_regime.volatility_multiplier must be non-negative"),
    # market
    ("stock volatility negative", _set("market", "stock_volatility", -0.1),
     "market.stock_volatility must be non-negative"),
    ("dividend yield above 20%", _set("market", "stock_dividend_yield", 0.5),
     "market.stock_dividend_yield must be between 0 and 0.2"),
    ("bond taxable yield negative", _set("market", "bond_taxable_yield", -0.01),
     "market.bond_taxable_yield must be between 0 and 0.2"),
    # mortality and Social Security
    ("unknown mortality model", _set("life_events", "mortality_model", "magic"),
     "life_events.mortality_model must be one of"),
    ("unknown mortality sex", _set("life_events", "mortality_sex", "other"),
     "life_events.mortality_sex must be one of"),
    ("death age min above max", _set("life_events", "death_age_min", 115),
     "life_events.death_age_min must be <= death_age_max"),
    ("credits above 40", _set("life_events", "ss_credits_at_current_age", 41),
     "ss_credits_at_current_age must be 0..40"),
    ("negative earnings years",
     _set("life_events", "ss_earnings_years_at_current_age", -1),
     "ss_earnings_years_at_current_age must be non-negative"),
    ("negative FRA benefit",
     _set("life_events", "ss_annual_full_retirement_benefit", -1.0),
     "ss_annual_full_retirement_benefit must be non-negative"),
    ("negative death age std", lambda c: (
        setattr(c.life_events, "mortality_model", "normal"),
        setattr(c.life_events, "death_age_std", -1.0)),
     "life_events.death_age_std must be non-negative"),
    ("benefit years required zero",
     _set("life_events", "ss_benefit_years_required", 0),
     "ss_benefit_years_required must be at least 1"),
    ("eligibility credits above 40",
     _set("life_events", "ss_retirement_eligibility_credits", 41),
     "ss_retirement_eligibility_credits must be 1..40"),
    ("survivor spending factor above 1",
     _enable_spouse(_set("spouse", "survivor_spending_factor", 5.0)),
     "survivor_spending_factor must be between 0 and 1"),
    ("survivor spending factor negative",
     _enable_spouse(_set("spouse", "survivor_spending_factor", -1.0)),
     "survivor_spending_factor must be between 0 and 1"),
    # taxes
    ("unknown filing status", _set("taxes", "filing_status", "martian"),
     "filing_status must be one of"),
    ("bracket table not a mapping", _set("taxes", "federal_brackets", []),
     "taxes.federal_brackets must map filing status -> values"),
    ("negative bracket bound", lambda c: c.taxes.federal_brackets.__setitem__(
        "single", [[-100, 0.10], [50_400, 0.12], [None, 0.22]]),
     "upper bounds must be positive"),
    ("ss thresholds wrong length", lambda c: c.taxes.ss_provisional_thresholds
        .__setitem__("single", [25_000]),
     "ss_provisional_thresholds[single] must be [lower, upper]"),
    ("ss thresholds reversed", lambda c: c.taxes.ss_provisional_thresholds
        .__setitem__("single", [60_000, 32_000]),
     "ss_provisional_thresholds[single] must be non-negative and increasing"),
    ("negative niit threshold",
     lambda c: c.taxes.niit_thresholds.__setitem__("single", -1),
     "niit_thresholds[single] must be non-negative"),
    ("penalty above 100%", _set("taxes", "early_withdrawal_penalty", 1.5),
     "early_withdrawal_penalty must be between 0 and 1"),
    ("implausible penalty free age", _set("taxes", "penalty_free_age", 300.0),
     "penalty_free_age must be a plausible age"),
    ("ss max taxable fraction above 1", _set("taxes", "ss_max_taxable_fraction", 1.5),
     "ss_max_taxable_fraction must be between 0 and 1"),
    ("niit rate above 20%", _set("taxes", "niit_rate", 0.5),
     "niit_rate must be between 0 and 0.2"),
    ("state tax rate above 20%", _set("taxes", "state_tax_rate", 0.5),
     "state_tax_rate must be between 0 and 0.2"),
    # spending
    ("decline rate above 1", _set("spending", "annual_spending_decline_rate", 5.0),
     "annual_spending_decline_rate must be in (0, 1]"),
    ("decline rate zero", _set("spending", "annual_spending_decline_rate", 0.0),
     "annual_spending_decline_rate must be in (0, 1]"),
    ("healthcare rate <= -1", _set("spending", "annual_healthcare_increase_rate", -5.0),
     "annual_healthcare_increase_rate must be > -1"),
    ("implausible decline start age",
     _set("spending", "spending_decline_start_age", -50),
     "spending.spending_decline_start_age must be a plausible age"),
    ("decline start after end", lambda c: (
        setattr(c.spending, "spending_decline_start_age", 90),
        setattr(c.spending, "spending_decline_end_age", 80)),
     "spending_decline_start_age must be <= spending_decline_end_age"),
    ("guardrail thresholds inverted", lambda c: (
        setattr(c.spending, "guardrail_cut_return_threshold", 0.20),
        setattr(c.spending, "guardrail_raise_return_threshold", -0.20)),
     "guardrail_cut_return_threshold must be below"),
    ("cut amount above 1", _set("spending", "guardrail_cut_amount", 1.5),
     "guardrail_cut_amount must be in (0, 1]"),
    ("cut floor zero", _set("spending", "guardrail_cut_floor", 0.0),
     "guardrail_cut_floor must be in (0, 1]"),
    ("raise amount below 1", _set("spending", "guardrail_raise_amount", 0.5),
     "guardrail_raise_amount must be >= 1"),
    ("raise ceiling below 1", _set("spending", "guardrail_raise_ceiling", 0.5),
     "guardrail_raise_ceiling must be >= 1"),
    # healthcare
    ("negative premium", _set("healthcare", "pre_medicare_annual_premium", -1.0),
     "healthcare.pre_medicare_annual_premium must be non-negative"),
    ("negative medicare premium", _set("healthcare", "medicare_annual_premium", -1.0),
     "healthcare.medicare_annual_premium must be non-negative"),
    ("implausible medicare age", _set("healthcare", "medicare_age", 20),
     "healthcare.medicare_age must be a plausible age"),
]


@pytest.mark.parametrize("mutate,fragment",
                         [(m, f) for _, m, f in RULES],
                         ids=[label for label, _, _ in RULES])
def test_every_validation_rule_fires(base_cfg, mutate, fragment):
    cfg = copy.deepcopy(base_cfg)
    mutate(cfg)
    rejects(cfg, fragment)
