'use strict';
/**
 * Mechanics inside simulateLife that the deterministic fixtures switch OFF, plus
 * the results/reporting layer.
 *
 * deterministicCfg neutralises the guardrails, the spending smile and healthcare
 * so the withdrawal accounting is hand-computable. That is the right trade for
 * those tests, but it leaves three real spending mechanisms and the whole
 * reporting layer unasserted. Each test here turns exactly one of them back on.
 *
 * Four tests in the Python file have no counterpart: the matplotlib chart tests
 * and the stdio/launcher regression guards are desktop-app plumbing that the web
 * version does not have.
 */

const { test } = require('node:test');
const assert = require('node:assert');
const { RS, baseCfg, deterministicCfg, approx } = require('./helpers.js');

const median = xs => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;

// ---------------------------------------------------------------- glide path
test('the glide path holds the start mix to retirement, then moves linearly', () => {
  const cfg = baseCfg();
  cfg.simulation.glide_path = true;
  cfg.simulation.glide_path_start_stock_pct = 0.90;
  cfg.simulation.glide_path_end_stock_pct = 0.50;
  cfg.simulation.glide_path_years = 20;

  const allocation = (age, retirementAge = 65) => {
    const years = Math.max(cfg.simulation.glide_path_years, 1);
    const frac = Math.min(1.0, Math.max(0.0, age - retirementAge) / years);
    return 0.90 + (0.50 - 0.90) * frac;
  };
  approx(allocation(40), 0.90, { rel: 1e-12 });     // decades before retiring
  approx(allocation(65), 0.90, { rel: 1e-12 });     // the day itself
  approx(allocation(75), 0.70, { rel: 1e-12 });     // halfway down
  approx(allocation(85), 0.50, { rel: 1e-12 });     // complete
  approx(allocation(100), 0.50, { rel: 1e-12 });    // and it stays
});

test('a RISING glide path is legitimate, not an error', () => {
  const cfg = baseCfg();
  cfg.simulation.glide_path_start_stock_pct = 0.30;
  cfg.simulation.glide_path_end_stock_pct = 0.70;
  RS.validateConfig(cfg);
  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 1, false);
  assert.ok(typeof r.survived === 'boolean');
});

// ---------------------------------------------------------------- guardrails
test('cutting spending after a bad year leaves more in the portfolio', () => {
  const base = deterministicCfg({ realReturn: -0.06, spend: 60000.0, roth: 3000000.0,
                                  retireAge: 65, deathAge: 85 });
  const withRails = RS.deepClone(base);
  withRails.spending.guardrail_cut_return_threshold = -0.02;
  withRails.spending.guardrail_cut_amount = 0.90;
  withRails.spending.guardrail_cut_floor = 0.70;

  const off = new RS.RetirementSimulator(base).simulateLife(65, 0, false);
  const on = new RS.RetirementSimulator(withRails).simulateLife(65, 0, false);
  assert.strictEqual(off.survived, true);
  assert.strictEqual(on.survived, true);
  assert.ok(on.final > off.final * 1.5, 'cutting spending compounds into the estate');
});

test('the cut floor binds however aggressive the cut amount is', () => {
  const cfg = deterministicCfg({ realReturn: -0.06, spend: 60000.0, roth: 3000000.0,
                                 retireAge: 65, deathAge: 85 });
  cfg.spending.guardrail_cut_return_threshold = -0.02;
  cfg.spending.guardrail_cut_floor = 0.70;

  const gentle = RS.deepClone(cfg); gentle.spending.guardrail_cut_amount = 0.70;
  const savage = RS.deepClone(cfg); savage.spending.guardrail_cut_amount = 0.10;

  const a = new RS.RetirementSimulator(gentle).simulateLife(65, 0, false);
  const b = new RS.RetirementSimulator(savage).simulateLife(65, 0, false);
  assert.strictEqual(a.survived, true);
  assert.strictEqual(b.survived, true);
  approx(a.final, b.final, { rel: 1e-9 });        // both clamp to the same 70% floor

  let multiplier = 1.0;
  for (let i = 0; i < 20; i++) multiplier = Math.max(multiplier * 0.10, 0.70);
  approx(multiplier, 0.70, { rel: 1e-12 });
});

test('the raise ceiling binds however long the boom lasts', () => {
  const cfg = deterministicCfg({ realReturn: 0.20, spend: 50000.0, roth: 1000000.0,
                                 retireAge: 65, deathAge: 95 });
  cfg.spending.guardrail_raise_return_threshold = 0.05;
  cfg.spending.guardrail_raise_amount = 2.0;       // double every single year
  cfg.spending.guardrail_raise_ceiling = 1.30;

  let multiplier = 1.0;
  for (let i = 0; i < 30; i++) multiplier = Math.min(multiplier * 2.0, 1.30);
  approx(multiplier, 1.30, { rel: 1e-12 });
  assert.ok(new RS.RetirementSimulator(cfg).simulateLife(65, 0, false).final > 1000000.0);
});

test('guardrails cannot fire in the first year', () => {
  // there is no completed year to judge at the first anniversary
  const cfg = deterministicCfg({ realReturn: -0.30, spend: 50000.0, roth: 1000000.0,
                                 retireAge: 65, deathAge: 66 });
  cfg.spending.guardrail_cut_return_threshold = -0.05;
  cfg.spending.guardrail_cut_amount = 0.50;
  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 0, false);
  const g = Math.pow(1 - 0.30, 1 / 12);
  approx(r.iwr, 50000.0 / (1000000.0 * g), { abs: 1e-6 });
});

// ---------------------------------------------------------------- healthcare
test('premiums are per person, each on their OWN Medicare clock', () => {
  // 0% real, no base spending, all Roth so no tax -> the drawdown IS the
  // lifetime healthcare bill. Primary dies at 95; the spouse is five years
  // younger and dies at 95 on THEIR clock, i.e. at primary-100:
  //   primary  ages 65-94, Medicare       30 x  5,000 = 150,000
  //   spouse   ages 60-64, pre-Medicare    5 x 20,000 = 100,000
  //   spouse   ages 65-89, Medicare       25 x  5,000 = 125,000
  //   survivor spouse 90-94, Medicare      5 x  5,000 =  25,000  -> 400,000
  const cfg = deterministicCfg({ realReturn: 0.0, spend: 0.0, roth: 2000000.0,
                                 retireAge: 65, deathAge: 95 });
  cfg.spouse.enabled = true;
  cfg.spouse.age_offset = -5;
  cfg.spouse.mortality_model = 'normal';
  cfg.spouse.death_age_mean = 95;
  cfg.spouse.death_age_min = 95;
  cfg.spouse.death_age_max = 95;
  cfg.spouse.death_age_std = 0.0;
  cfg.healthcare.pre_medicare_annual_premium = 20000.0;
  cfg.healthcare.medicare_annual_premium = 5000.0;
  cfg.healthcare.medicare_age = 65;

  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 0, false);
  approx(2000000.0 - r.final, 400000.0, { abs: 1.0 });
});

test("the spouse's Medicare eligibility uses their own age, not the primary's", () => {
  // both die on the primary's 95th, so there are no survivor-only years:
  //   30 x 5,000 + 5 x 20,000 + 25 x 5,000 = 375,000
  const cfg = deterministicCfg({ realReturn: 0.0, spend: 0.0, roth: 2000000.0,
                                 retireAge: 65, deathAge: 95 });
  cfg.spouse.enabled = true;
  cfg.spouse.age_offset = -5;
  cfg.spouse.mortality_model = 'normal';
  cfg.spouse.death_age_mean = 90;
  cfg.spouse.death_age_min = 90;
  cfg.spouse.death_age_max = 90;
  cfg.spouse.death_age_std = 0.0;
  cfg.healthcare.pre_medicare_annual_premium = 20000.0;
  cfg.healthcare.medicare_annual_premium = 5000.0;
  cfg.healthcare.medicare_age = 65;

  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 0, false);
  approx(2000000.0 - r.final, 375000.0, { abs: 1.0 });
});

test('the healthcare multiplier applies to premiums only, not base spending', () => {
  const cfg = baseCfg();
  const sim = new RS.RetirementSimulator(cfg);
  const s = cfg.spending;
  const [baseFactor, health] = sim.spendingSmile(s.spending_decline_end_age + 4);
  const [frozen] = sim.spendingSmile(s.spending_decline_end_age);
  approx(baseFactor, frozen, { rel: 1e-12 });
  approx(health, Math.pow(1 + s.annual_healthcare_increase_rate, 4), { rel: 1e-12 });
});

// ---------------------------------------------------------------- results layer
test('every reported dollar figure is a MEDIAN, not a mean', () => {
  // on a fragile portfolio the mean low-water mark is wildly optimistic while
  // the median tells the truth
  const cfg = baseCfg();
  cfg.simulation.current_age = 60;
  cfg.simulation.min_retirement_age = 60;
  cfg.simulation.max_retirement_age = 60;
  cfg.simulation.monte_carlo_runs = 120;
  cfg.accounts.roth = 250000.0;
  cfg.accounts.traditional = cfg.accounts.brokerage = cfg.accounts.cash = 0.0;
  cfg.accounts.brokerage_cost_basis = 0.0;
  cfg.spending.initial_annual_expenses = 90000.0;
  const sim = new RS.RetirementSimulator(cfg);

  const runs = sim.runSeeds(60, 120).map(s => sim.simulateLife(60, s, false));
  const [prob, medMin, medWr, medFinal] = sim.retirementProbability(60);

  approx(prob, runs.filter(r => r.survived).length / runs.length, { rel: 1e-12 });
  approx(medMin, median(runs.map(r => r.minPortfolio)), { rel: 1e-9 });
  approx(medFinal, median(runs.map(r => r.final)), { rel: 1e-9 });
  approx(medWr, median(runs.filter(r => r.iwr != null).map(r => r.iwr)), { rel: 1e-9 });
  assert.ok(mean(runs.map(r => r.minPortfolio)) > medMin,
            'the mean should be materially higher on this fragile portfolio');
});

test('findRetirementAge picks the earliest age clearing the target', () => {
  const sim = new RS.RetirementSimulator(baseCfg());
  sim.cfg.simulation.target_success_probability = 0.90;
  sim.probabilityResults = { 60: [0.50, 0, 0, 0], 61: [0.89, 0, 0, 0],
                             62: [0.91, 0, 0, 0], 63: [0.99, 0, 0, 0] };
  const r = sim.findRetirementAge();
  assert.strictEqual(r.retirement_age, 62);
  approx(r.success_probability, 0.91, { rel: 1e-12 });
});

test('findRetirementAge returns null when no age qualifies', () => {
  const sim = new RS.RetirementSimulator(baseCfg());
  sim.cfg.simulation.target_success_probability = 0.95;
  sim.probabilityResults = { 60: [0.10, 0, 0, 0], 61: [0.20, 0, 0, 0] };
  assert.strictEqual(sim.findRetirementAge(), null);
  assert.ok(sim.formatResultsTable().includes('No retirement age met'));
});

test('the results table renders every swept age', () => {
  const sim = new RS.RetirementSimulator(baseCfg());
  sim.cfg.simulation.target_success_probability = 0.90;
  sim.probabilityResults = { 60: [0.50, 100000.0, 0.055, 200000.0],
                             61: [0.95, 300000.0, 0.041, 900000.0] };
  const table = sim.formatResultsTable();
  assert.ok(table.includes('AGE 61'));
  for (const age of [60, 61])
    assert.ok(table.split('\n').some(l => l.trim().startsWith(String(age))),
              `age ${age} is missing from the table`);
});

test('the assumption report survives crises being disabled', () => {
  // setting monthly_crisis_probability to 0 is the natural way to switch the
  // regime model off, and validateConfig accepts it -- so nothing may divide by it
  const a = baseCfg();
  a.simulation.normal_regime.monthly_crisis_probability = 0.0;
  RS.validateConfig(a);
  assert.ok(new RS.RetirementSimulator(a).assumptionReport().includes('crisis regime'));

  const b = baseCfg();
  b.simulation.crisis_regime.monthly_recovery_probability = 0.0;
  RS.validateConfig(b);
  new RS.RetirementSimulator(b).assumptionReport();          // must not throw
});

test('the probability curve is monotone under common random numbers', () => {
  // every age faces the SAME lifetimes, so a later retirement can only help.
  // A non-monotone curve is the canonical signal of an engine regression.
  const cfg = baseCfg();
  cfg.simulation.current_age = 55;
  cfg.simulation.common_random_numbers = true;
  const sim = new RS.RetirementSimulator(cfg);
  const probs = [];
  for (let age = 60; age <= 64; age++) {
    const runs = sim.runSeeds(age, 150).map(s => sim.simulateLife(age, s, false));
    probs.push(runs.filter(r => r.survived).length / runs.length);
  }
  assert.strictEqual(probs.length, 5);
  for (let i = 1; i < probs.length; i++)
    assert.ok(probs[i - 1] <= probs[i] + 1e-12, `non-monotone: ${JSON.stringify(probs)}`);
});

test('the progress callback fires once per age, before that age runs', () => {
  const cfg = baseCfg();
  cfg.simulation.current_age = 60;
  cfg.simulation.min_retirement_age = 61;
  cfg.simulation.max_retirement_age = 63;
  cfg.simulation.monte_carlo_runs = 5;
  const sim = new RS.RetirementSimulator(cfg);
  const seen = [];
  sim.computeProbabilityCurve((i, total, age) => seen.push([i, total, age]));
  assert.deepStrictEqual(seen, [[0, 3, 61], [1, 3, 62], [2, 3, 63]]);
  assert.deepStrictEqual(Object.keys(sim.probabilityResults).map(Number).sort((a, b) => a - b),
                         [61, 62, 63]);
});

test('the fan chart drops columns too thin to support a percentile', () => {
  // with a handful of survivors every percentile lands on the same one or two
  // lives, which paints a dramatic late-life spike out of nothing
  assert.strictEqual(typeof RS.MIN_TRAJECTORY_SAMPLES, 'number');
  assert.ok(RS.MIN_TRAJECTORY_SAMPLES >= 30,
            'the threshold must be high enough for a 1st percentile to mean anything');

  const cfg = baseCfg();
  cfg.simulation.current_age = 60;
  const sim = new RS.RetirementSimulator(cfg);
  const { ages, percentiles } = sim.computeTrajectoryPercentiles(60, 200);
  const thin = ages.map((_, i) => percentiles.n[i])
                   .filter(n => n > 0 && n < RS.MIN_TRAJECTORY_SAMPLES);
  assert.ok(thin.length > 0, 'the tail should contain under-sampled columns to trim');
});
