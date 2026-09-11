'use strict';
/**
 * Shared fixtures for the JavaScript suite.
 *
 * The engine is read out of RetirementSimulator.html at test time rather than
 * copied into a separate file. That is deliberate: the tests then exercise the
 * exact bytes that ship, so an edit to the app cannot pass a suite running
 * against a stale copy.
 *
 * Hermeticity: every test starts from baseline_config.yaml, a FROZEN copy of the
 * parameter file, never the shipped simulation_params.yaml -- that one is edited
 * freely and personal numbers must not be able to break the suite.
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const APP = path.join(ROOT, 'RetirementSimulator.html');
const BASELINE = path.join(__dirname, 'baseline_config.yaml');

/** Extract and evaluate the <script id="engine-src"> block from the shipped app. */
function loadEngine() {
  const html = fs.readFileSync(APP, 'utf8');
  const m = html.match(/<script id="engine-src">([\s\S]*?)<\/script>/);
  if (!m) throw new Error('engine-src script block not found in RetirementSimulator.html');
  // Evaluated in THIS realm, not a vm context: a sandbox gets its own Array and
  // Object constructors, and assert.deepStrictEqual then rejects structurally
  // identical arrays for not being reference-equal on their prototypes.
  const mod = { exports: {} };
  new Function('module', 'exports', m[1])(mod, mod.exports);
  return mod.exports;
}

const RS = loadEngine();

/** A fresh, validated config from the frozen baseline. Mutate freely. */
function baseCfg() {
  return RS.configFromDict(RS.parseYaml(fs.readFileSync(BASELINE, 'utf8')));
}

/**
 * A scenario with NO randomness, so outcomes are computable by hand.
 * The whole monthly loop then reduces to a recurrence a test can replicate in a
 * few lines, so any change to growth/withdrawal ordering shows up as a
 * dollar-level mismatch.
 */
function deterministicCfg({ base = null, realReturn = 0.02, spend = 50000.0,
                            roth = 1000000.0, retireAge = 65, deathAge = 95 } = {}) {
  const cfg = RS.deepClone(base || baseCfg());
  cfg.accounts.roth = roth;
  cfg.accounts.traditional = 0.0;
  cfg.accounts.brokerage = 0.0;
  cfg.accounts.cash = 0.0;
  cfg.accounts.brokerage_cost_basis = 0.0;
  for (const f of ['annual_roth', 'annual_traditional', 'annual_brokerage', 'annual_cash'])
    cfg.contributions[f] = 0.0;

  cfg.simulation.current_age = retireAge;
  cfg.simulation.min_retirement_age = retireAge;
  cfg.simulation.max_retirement_age = retireAge;
  cfg.simulation.glide_path = false;
  cfg.simulation.static_stock_allocation = 1.0;
  cfg.simulation.random_seed = 1;
  // crises impossible -> the regime never leaves 'normal'
  cfg.simulation.normal_regime.monthly_crisis_probability = 0.0;
  cfg.simulation.normal_regime.return_boost = 0.0;

  // a nominal return that is EXACTLY realReturn after 3% inflation
  cfg.market.inflation = 0.03;
  cfg.market.stock_return = (1 + realReturn) * 1.03 - 1;
  cfg.market.bond_return = cfg.market.stock_return;
  cfg.market.stock_volatility = 0.0;
  cfg.market.bond_volatility = 0.0;
  cfg.market.inflation_volatility = 0.0;
  cfg.market.stock_dividend_yield = 0.0;
  cfg.market.bond_taxable_yield = 0.0;

  // death at an exact age: a "normal" draw with zero spread
  cfg.life_events.mortality_model = 'normal';
  cfg.life_events.death_age_mean = deathAge;
  cfg.life_events.death_age_std = 0.0;
  cfg.life_events.death_age_min = deathAge;
  cfg.life_events.death_age_max = deathAge;
  cfg.life_events.ss_annual_full_retirement_benefit = 0.0;
  cfg.spouse.enabled = false;

  cfg.healthcare.pre_medicare_annual_premium = 0.0;
  cfg.healthcare.medicare_annual_premium = 0.0;

  // constant real spending: smile and guardrails neutralised
  cfg.spending.initial_annual_expenses = spend;
  cfg.spending.spending_decline_start_age = 119;
  cfg.spending.spending_decline_end_age = 120;
  cfg.spending.annual_healthcare_increase_rate = 0.0;
  cfg.spending.guardrail_cut_return_threshold = -0.99;
  cfg.spending.guardrail_raise_return_threshold = 0.99;
  cfg.spending.guardrail_cut_amount = 1.0;
  cfg.spending.guardrail_raise_amount = 1.0;
  return cfg;
}

/** A Trinity-study-shaped benchmark: $1M, fixed horizon, constant real
 * spending, tax-free, Trinity-like capital markets. */
function trinityCfg({ base = null, spend, years = 30, stockPct = 0.5 } = {}) {
  const cfg = RS.deepClone(base || baseCfg());
  cfg.accounts.roth = 1000000.0;
  cfg.accounts.traditional = 0.0;
  cfg.accounts.brokerage = 0.0;
  cfg.accounts.cash = 0.0;
  cfg.accounts.brokerage_cost_basis = 0.0;
  for (const f of ['annual_roth', 'annual_traditional', 'annual_brokerage', 'annual_cash'])
    cfg.contributions[f] = 0.0;

  cfg.simulation.current_age = 65;
  cfg.simulation.min_retirement_age = 65;
  cfg.simulation.max_retirement_age = 65;
  cfg.simulation.glide_path = false;
  cfg.simulation.static_stock_allocation = stockPct;
  cfg.simulation.random_seed = 12345;
  cfg.simulation.common_random_numbers = true;

  cfg.life_events.mortality_model = 'normal';       // fixed horizon, like the studies
  cfg.life_events.death_age_mean = 65 + years;
  cfg.life_events.death_age_std = 0.0;
  cfg.life_events.death_age_min = 65 + years;
  cfg.life_events.death_age_max = 65 + years;
  cfg.life_events.ss_annual_full_retirement_benefit = 0.0;
  cfg.spouse.enabled = false;

  cfg.healthcare.pre_medicare_annual_premium = 0.0;
  cfg.healthcare.medicare_annual_premium = 0.0;

  cfg.market.stock_return = 0.103;   // ~7.09% real at 3% inflation
  cfg.market.bond_return = 0.055;    // ~2.43% real
  cfg.market.bond_volatility = 0.08;
  cfg.market.stock_bond_correlation = 0.1;
  cfg.market.stock_dividend_yield = 0.0;
  cfg.market.bond_taxable_yield = 0.0;

  cfg.spending.initial_annual_expenses = spend;
  cfg.spending.spending_decline_start_age = 119;
  cfg.spending.spending_decline_end_age = 120;
  cfg.spending.annual_healthcare_increase_rate = 0.0;
  cfg.spending.guardrail_cut_return_threshold = -0.99;
  cfg.spending.guardrail_raise_return_threshold = 0.99;
  cfg.spending.guardrail_cut_amount = 1.0;
  cfg.spending.guardrail_raise_amount = 1.0;
  return cfg;
}

/** Success rate over n lifetimes, run through the engine's own seed scheme. */
function serialSuccess(cfg, n = 400) {
  const sim = new RS.RetirementSimulator(cfg);
  const age = cfg.simulation.min_retirement_age;
  const seeds = sim.runSeeds(age, n);
  let ok = 0;
  for (const s of seeds) if (sim.simulateLife(age, s, false).survived) ok++;
  return ok / n;
}

/** assert.ok with an absolute or relative tolerance and a readable message. */
function approx(actual, expected, { abs = null, rel = null } = {}) {
  const tol = abs != null ? abs : Math.abs(expected) * (rel != null ? rel : 1e-9);
  const ok = Math.abs(actual - expected) <= tol;
  if (!ok) {
    throw new Error(`expected ${expected} +/- ${tol}, got ${actual} ` +
                    `(difference ${Math.abs(actual - expected)})`);
  }
  return true;
}

/** Assert that validateConfig rejects `cfg` and names `fragment`. */
function rejects(assert, cfg, fragment) {
  let message = null;
  try {
    RS.validateConfig(cfg);
  } catch (e) {
    message = e.message;
  }
  assert.ok(message !== null, `expected validateConfig to reject, naming "${fragment}"`);
  assert.ok(message.includes(fragment),
            `expected the error to mention "${fragment}", got: ${message}`);
}

module.exports = { RS, baseCfg, deterministicCfg, trinityCfg, serialSuccess, approx, rejects };
