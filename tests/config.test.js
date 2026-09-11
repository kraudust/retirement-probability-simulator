'use strict';
/**
 * Tests of the config layer: schema binding, validation, dotted-path access and
 * the YAML round trip.
 *
 * Why this file exists: validateConfig is the only thing standing between a
 * mistyped parameter and a quietly wrong retirement age, and a rule that never
 * fires in the suite is a rule nobody has checked. Each case below asserts that
 * a specific bad value is REJECTED, so a refactor that drops a check fails here
 * instead of silently shipping.
 *
 * The most important case is bracket ordering: the bracket walk assumes
 * ascending upper bounds, so a shuffled table does not raise -- it just returns
 * the wrong tax.
 */

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { RS, baseCfg, rejects } = require('./helpers.js');

const ROOT = path.join(__dirname, '..');
const rawBaseline = () => RS.parseYaml(
  fs.readFileSync(path.join(__dirname, 'baseline_config.yaml'), 'utf8'));
const MFJ = 'married_filing_jointly';

const throwsWith = (fn, fragment) => {
  let m = null;
  try { fn(); } catch (e) { m = e.message; }
  assert.ok(m !== null, `expected a throw mentioning "${fragment}"`);
  assert.ok(m.includes(fragment), `expected "${fragment}", got: ${m}`);
};

// ---------------------------------------------------------------- the schema
test('the frozen baseline passes every rule', () => {
  RS.validateConfig(baseCfg());
});

test('a missing key is reported by name', () => {
  const raw = rawBaseline();
  delete raw.market.inflation;
  throwsWith(() => RS.configFromDict(raw), 'market.inflation');
});

test('an unknown key is rejected by name', () => {
  const raw = rawBaseline();
  raw.market.inflaton = 0.03;                       // typo
  throwsWith(() => RS.configFromDict(raw), 'inflaton');
});

test('validation reports every problem at once, not just the first', () => {
  const cfg = baseCfg();
  cfg.accounts.roth = -1.0;
  cfg.simulation.monte_carlo_runs = 0;
  cfg.spending.initial_annual_expenses = -5.0;
  let m = null;
  try { RS.validateConfig(cfg); } catch (e) { m = e.message; }
  assert.ok(m !== null);
  for (const f of ['accounts.roth', 'monte_carlo_runs', 'initial_annual_expenses'])
    assert.ok(m.includes(f), `expected "${f}" in: ${m}`);
});

// ------------------------------------------- tax tables (silent-wrong risk)
test('unsorted brackets are rejected', () => {
  // THE case that motivated this file: a shuffled 2026 single table taxes $60k
  // at $16,520 instead of $7,912 and raises nothing.
  const cfg = baseCfg();
  cfg.taxes.federal_brackets.single = [[50400, 0.12], [12400, 0.10], [null, 0.22]];
  rejects(assert, cfg, 'upper bounds must increase');
});

test('a bracket rate outside [0, 1] is rejected', () => {
  for (const bad of [-0.5, 3.0]) {
    const cfg = baseCfg();
    cfg.taxes.federal_brackets.single[0] = [12400, bad];
    rejects(assert, cfg, 'rates must be between 0 and 1');
  }
});

test('a null upper bound is allowed only on the top bracket', () => {
  const a = baseCfg();
  a.taxes.federal_brackets.single[0] = [null, 0.10];
  rejects(assert, a, 'null upper bound on the LAST bracket');

  const b = baseCfg();
  b.taxes.ltcg_brackets.single = [[49450, 0.0], [545500, 0.15], [900000, 0.20]];
  rejects(assert, b, 'must end with a null-bounded top bracket');
});

test('every per-status table must cover the configured status AND single', () => {
  // a surviving spouse switches to single mid-simulation
  for (const table of ['standard_deductions', 'additional_standard_deductions_65plus',
                       'federal_brackets', 'ltcg_brackets', 'ss_provisional_thresholds',
                       'niit_thresholds']) {
    const cfg = baseCfg();
    delete cfg.taxes[table].single;
    rejects(assert, cfg, `taxes.${table} is missing an entry for 'single'`);
  }
});

test('negative deductions are rejected', () => {
  const a = baseCfg();
  a.taxes.standard_deductions.single = -1;
  rejects(assert, a, 'standard_deductions[single] must be non-negative');

  const b = baseCfg();
  b.taxes.additional_standard_deductions_65plus.single = -1;
  rejects(assert, b, 'additional_standard_deductions_65plus[single] must be non-negative');
});

test('a modelled spouse cannot file as a single person', () => {
  // two incomes against one person's deduction and brackets for every joint year
  for (const bad of ['single', 'head_of_household']) {
    const cfg = baseCfg();
    cfg.spouse.enabled = true;
    cfg.taxes.filing_status = bad;
    rejects(assert, cfg, 'a modelled couple must file as married_filing_jointly');
  }
  const ok = baseCfg();
  ok.spouse.enabled = true;
  ok.taxes.filing_status = MFJ;
  RS.validateConfig(ok);

  // the reverse IS legitimate: you may just not want a second lifespan simulated
  const singlePlanner = baseCfg();
  singlePlanner.spouse.enabled = false;
  singlePlanner.taxes.filing_status = MFJ;
  RS.validateConfig(singlePlanner);
});

// ---------------------------------------------------------------- market inputs
test('a rate at or below -1 is rejected', () => {
  // realReturn divides by (1 + inflation) and the risky legs take log1p of the
  // real return, so -1 is a division or domain error deep in the constructor
  for (const field of ['stock_return', 'bond_return', 'cash_return', 'inflation']) {
    const cfg = baseCfg();
    cfg.market[field] = -1.0;
    rejects(assert, cfg, `market.${field} must be greater than -1`);
  }
});

test('extreme inflation volatility is rejected', () => {
  const cfg = baseCfg();
  cfg.market.inflation_volatility = 1.0;
  rejects(assert, cfg, 'inflation_volatility above 0.25');
});

test('a correlation outside [-1, 1] is rejected', () => {
  const cfg = baseCfg();
  cfg.market.stock_bond_correlation = -1.5;
  rejects(assert, cfg, 'stock_bond_correlation');
});

test('a Student-t needs df > 2 for finite variance', () => {
  const cfg = baseCfg();
  cfg.simulation.return_distribution_degrees_of_freedom = 2;
  rejects(assert, cfg, 'degrees_of_freedom');
});

// ---------------------------------------------------------------- ages and ranges
test('implausible ages are rejected', () => {
  const a = baseCfg(); a.simulation.current_age = -10;
  rejects(assert, a, 'current_age must be between 0 and 120');

  const b = baseCfg(); b.life_events.death_age_max = 500;
  rejects(assert, b, 'death_age_max must be above current_age and at most 120');

  const c = baseCfg(); c.spouse.enabled = true; c.spouse.age_offset = -80;
  rejects(assert, c, 'age_offset must be within 50 years');
});

test('the retirement range must be ordered', () => {
  const a = baseCfg();
  a.simulation.min_retirement_age = a.simulation.current_age - 1;
  rejects(assert, a, 'min_retirement_age must be >= current_age');

  const b = baseCfg();
  b.simulation.max_retirement_age = b.simulation.min_retirement_age - 1;
  rejects(assert, b, 'max_retirement_age must be >= min_retirement_age');
});

test('a claim age outside 62-70 is rejected', () => {
  for (const bad of [61, 71]) {
    const cfg = baseCfg();
    cfg.life_events.ss_claim_age = bad;
    rejects(assert, cfg, 'ss_claim_age must be between 62 and 70');
  }
});

test('spouse rules are skipped when the spouse is disabled', () => {
  const cfg = baseCfg();
  cfg.spouse.enabled = false;
  cfg.spouse.ss_claim_age = 12;
  cfg.spouse.mortality_sex = 'nonsense';
  RS.validateConfig(cfg);            // a nonsense spouse block must not block a run
});

test('a negative cost basis is rejected, but an underwater one is fine', () => {
  const bad = baseCfg();
  bad.accounts.brokerage_cost_basis = -1.0;
  rejects(assert, bad, 'brokerage_cost_basis');

  // basis ABOVE the balance is legitimate -- the ladder handles underwater accounts
  const underwater = baseCfg();
  underwater.accounts.brokerage = 1000.0;
  underwater.accounts.brokerage_cost_basis = 2000.0;
  RS.validateConfig(underwater);
});

test('allocation percentages must be fractions', () => {
  for (const field of ['glide_path_start_stock_pct', 'glide_path_end_stock_pct',
                       'static_stock_allocation']) {
    const cfg = baseCfg();
    cfg.simulation[field] = 1.5;
    rejects(assert, cfg, field);
  }
});

// ---------------------------------------------------------------- dotted paths
test('getField and setField round-trip', () => {
  const cfg = baseCfg();
  assert.strictEqual(RS.getField(cfg, 'accounts.roth'), cfg.accounts.roth);
  RS.setField(cfg, 'accounts.roth', 12345.0);
  assert.strictEqual(cfg.accounts.roth, 12345.0);
});

test('setField coerces to the declared type', () => {
  // the UI hands over strings from text inputs; the schema decides the type
  const cfg = baseCfg();
  RS.setField(cfg, 'accounts.roth', '250000');
  assert.strictEqual(cfg.accounts.roth, 250000.0);
  assert.strictEqual(typeof cfg.accounts.roth, 'number');
  RS.setField(cfg, 'simulation.monte_carlo_runs', '500.7');
  assert.strictEqual(cfg.simulation.monte_carlo_runs, 500);     // truncating
  RS.setField(cfg, 'simulation.glide_path', false);
  assert.strictEqual(cfg.simulation.glide_path, false);
  // string-typed fields pass through and are caught by validateConfig
  RS.setField(cfg, 'taxes.filing_status', 'single');
  assert.strictEqual(cfg.taxes.filing_status, 'single');
});

test('dotted paths reach nested sections', () => {
  const cfg = baseCfg();
  RS.setField(cfg, 'simulation.crisis_regime.annual_return_drag', '-0.2');
  assert.strictEqual(cfg.simulation.crisis_regime.annual_return_drag, -0.2);
  assert.strictEqual(RS.getField(cfg, 'simulation.crisis_regime.annual_return_drag'), -0.2);
});

test('an unknown dotted path throws', () => {
  for (const p of ['accounts.not_a_field', 'simulation.normal_regime.nope'])
    assert.throws(() => RS.getField(baseCfg(), p));
});

// ---------------------------------------------------------------- YAML round trip
test('save -> load is lossless for every field', () => {
  // a dropped key here would silently reset a parameter
  const cfg = baseCfg();
  const text = RS.saveConfigText(cfg);
  const reloaded = RS.loadConfigText(text);
  assert.deepStrictEqual(reloaded, cfg);
});

test('the shipped parameter file matches the schema', () => {
  // simulation_params.yaml is what the app actually loads, and it is hand-edited
  const raw = RS.parseYaml(fs.readFileSync(path.join(ROOT, 'simulation_params.yaml'), 'utf8'));
  const cfg = RS.configFromDict(raw);
  RS.validateConfig(cfg);
});

test('the baseline and the shipped file have the same SHAPE', () => {
  // the frozen test baseline must not drift from the config the app ships
  const shipped = RS.parseYaml(
    fs.readFileSync(path.join(ROOT, 'simulation_params.yaml'), 'utf8'));
  const baseline = rawBaseline();
  const shape = node => {
    if (node !== null && typeof node === 'object' && !Array.isArray(node)) {
      const out = {};
      for (const k of Object.keys(node).sort()) out[k] = shape(node[k]);
      return out;
    }
    return Array.isArray(node) ? 'list' : typeof node;
  };
  assert.deepStrictEqual(shape(shipped), shape(baseline));
});
