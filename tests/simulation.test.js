'use strict';
/**
 * Whole-engine tests of simulateLife.
 *
 * The deterministic scenarios are the strongest checks in the suite: with zero
 * volatility and fixed lifespans the monthly loop reduces to arithmetic a test
 * can replicate in a few lines, so growth/withdrawal ordering, plan timing and
 * the success accounting are all pinned to the dollar.
 *
 * The two multiprocessing tests in the Python file have no counterpart here:
 * the web engine runs single-threaded, which is itself why it is faster.
 */

const { test } = require('node:test');
const assert = require('node:assert');
const { RS, baseCfg, deterministicCfg, approx } = require('./helpers.js');

// ---------------------------------------------------------------- closed form
test('30 years at exactly 4% real reproduces the hand-computed recurrence', () => {
  // growth first, then the annual withdrawal:
  //   R <- R*g - S   (anniversary month), then R <- R*g^11
  const cfg = deterministicCfg({ realReturn: 0.04, spend: 50000.0, roth: 1000000.0,
                                 retireAge: 65, deathAge: 95 });
  const sim = new RS.RetirementSimulator(cfg);
  const r = sim.simulateLife(65, 0, true);

  const g = Math.pow(1.04, 1 / 12);
  let R = 1000000.0;
  for (let i = 0; i < 30; i++) { R = R * g - 50000.0; R *= Math.pow(g, 11); }

  assert.strictEqual(r.survived, true);
  approx(r.final, R, { abs: 1.0 });
  approx(r.final, 336497.1822, { abs: 0.01 });      // the value Python asserts
  approx(r.iwr, 50000.0 / (1000000.0 * g), { abs: 1e-6 });
  assert.strictEqual(r.trajectory.length, 30);
  approx(r.trajectory[0], 1000000.0 * g - 50000.0 / 12, { abs: 1.0 });
});

test('the last dollar going out with the last bill is a SUCCESS', () => {
  const cfg = deterministicCfg({ realReturn: 0.0, spend: 30000.0, roth: 900000.0,
                                 retireAge: 65, deathAge: 95 });
  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 0, false);
  assert.strictEqual(r.survived, true);
  approx(r.final, 0.0, { abs: 1.0 });
});

test('running dry before death is a FAILURE', () => {
  const cfg = deterministicCfg({ realReturn: 0.0, spend: 100000.0, roth: 1000000.0,
                                 retireAge: 65, deathAge: 95 });
  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 0, false);
  assert.strictEqual(r.survived, false);
  approx(r.final, 0.0, { abs: 1.0 });
  approx(r.minPortfolio, 0.0, { abs: 1.0 });
});

test('one full RMD year through the engine, hand-computed to the cent', () => {
  // single 75-year-old, $500k traditional, 0% real growth, $10k spending
  //   RMD       = 500,000 / 24.6           = 20,325.2033
  //   deduction = 16,100 + 2,050 (age 65+) = 18,150
  //   tax       = 10% federal + 5% state   =    326.2805
  //   surplus   = 9,998.9228 reinvested; death at 76 ends the run
  //   estate = (500,000 - 20,325.2033) + 9,998.9228 = 489,673.7195
  const cfg = deterministicCfg({ realReturn: 0.0, spend: 10000.0, roth: 0.0,
                                 retireAge: 75, deathAge: 76 });
  cfg.accounts.traditional = 500000.0;
  cfg.taxes.filing_status = 'single';
  const r = new RS.RetirementSimulator(cfg).simulateLife(75, 0, false);
  assert.strictEqual(r.survived, true);
  approx(r.final, 489673.7195, { abs: 0.5 });
});

// ---------------------------------------------------------------- reproducibility
test('identical seeds reproduce a lifetime exactly', () => {
  const sim = new RS.RetirementSimulator(baseCfg());
  const a = sim.simulateLife(50, 123, false);
  const b = sim.simulateLife(50, 123, false);
  assert.deepStrictEqual([a.survived, a.minPortfolio, a.iwr, a.final],
                         [b.survived, b.minPortfolio, b.iwr, b.final]);
});

test('different seeds give a different lifetime', () => {
  const sim = new RS.RetirementSimulator(baseCfg());
  assert.notStrictEqual(sim.simulateLife(50, 123, false).final,
                        sim.simulateLife(50, 124, false).final);
});

test('common random numbers make adjacent ages share scenarios', () => {
  const cfg = baseCfg();
  cfg.accounts.roth = 5e7;                    // too rich to fail either way
  const sim = new RS.RetirementSimulator(cfg);
  const at45 = sim.simulateLife(45, 999, false).final;
  const at46 = sim.simulateLife(46, 999, false).final;
  assert.ok(Math.abs(at45 - at46) / at45 < 0.25,
            'same lifetime, so the gap is one year of contributions, not luck');
});

test('non-CRN seeds differ by age and never collide', () => {
  const cfg = baseCfg();
  cfg.simulation.common_random_numbers = false;
  const sim = new RS.RetirementSimulator(cfg);
  const all = [];
  for (let age = 40; age <= 70; age++) all.push(...sim.runSeeds(age, 500));
  assert.notDeepStrictEqual(sim.runSeeds(40, 10), sim.runSeeds(41, 10));
  assert.strictEqual(new Set(all).size, all.length, 'seed collision across ages');

  cfg.simulation.common_random_numbers = true;
  const crn = new RS.RetirementSimulator(cfg);
  assert.deepStrictEqual(crn.runSeeds(40, 50), crn.runSeeds(70, 50));
});

// ---------------------------------------------------------------- regressions
test('a retiree fully covered by Social Security succeeds with a $0 portfolio', () => {
  const cfg = baseCfg();
  for (const f of ['roth', 'traditional', 'brokerage', 'cash', 'brokerage_cost_basis'])
    cfg.accounts[f] = 0.0;
  for (const f of ['annual_roth', 'annual_traditional', 'annual_brokerage'])
    cfg.contributions[f] = 0.0;
  cfg.simulation.current_age = 68;
  cfg.simulation.min_retirement_age = 68;
  cfg.simulation.max_retirement_age = 68;
  cfg.spending.initial_annual_expenses = 5000.0;       // far below the benefit
  cfg.healthcare.pre_medicare_annual_premium = 0.0;
  cfg.healthcare.medicare_annual_premium = 0.0;
  cfg.life_events.ss_earnings_years_at_current_age = 35;
  assert.strictEqual(new RS.RetirementSimulator(cfg).simulateLife(68, 3, false).survived,
                     true);
});

test('spouse-enabled lifetimes run clean across the survivor branches', () => {
  const cfg = baseCfg();
  cfg.spouse.enabled = true;
  cfg.spouse.ss_annual_full_retirement_benefit = 20000.0;
  cfg.spouse.ss_earnings_years_at_current_age = 10;
  cfg.spouse.ss_credits_at_current_age = 40;
  const sim = new RS.RetirementSimulator(cfg);
  for (let seed = 0; seed < 50; seed++) {
    const r = sim.simulateLife(55, seed, false);
    assert.ok(typeof r.survived === 'boolean');
    assert.ok(r.final >= 0.0);
  }
});

test('a primary dying young forces the widowed household into early drawdown', () => {
  const cfg = baseCfg();
  cfg.spouse.enabled = true;
  cfg.life_events.mortality_model = 'normal';
  cfg.life_events.death_age_mean = 45;
  cfg.life_events.death_age_std = 1.0;
  cfg.life_events.death_age_min = 44;
  cfg.life_events.death_age_max = 46;
  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 11, false);
  assert.ok(typeof r.survived === 'boolean');
});

test('a life ending before the tested retirement age is a success', () => {
  const cfg = deterministicCfg({ realReturn: 0.0, spend: 50000.0, roth: 100000.0,
                                 retireAge: 65, deathAge: 95 });
  cfg.simulation.current_age = 40;
  cfg.simulation.min_retirement_age = 65;
  cfg.simulation.max_retirement_age = 65;
  cfg.life_events.death_age_mean = 50;
  cfg.life_events.death_age_min = 50;
  cfg.life_events.death_age_max = 50;
  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 0, false);
  assert.strictEqual(r.survived, true);
  assert.strictEqual(r.iwr, null);                 // no retirement statistics exist
  approx(r.final, 100000.0, { abs: 1.0 });         // 0% real, no contributions
});

// ------------------------------------------- cost basis is nominal, not real
test('the basis deflator tracks inflation exactly', () => {
  const cfg = baseCfg();
  cfg.market.inflation = 0.03;
  cfg.market.inflation_volatility = 0.0;
  const { basisDecay } = new RS.RetirementSimulator(cfg).marketPath(360, new RS.Rng(0));
  const prod = basisDecay.reduce((a, b) => a * b, 1.0);
  approx(prod, 1 / Math.pow(1.03, 30), { rel: 1e-12 });
});

test('at 0% inflation the basis does not move', () => {
  const cfg = baseCfg();
  cfg.market.inflation = 0.0;
  cfg.market.inflation_volatility = 0.0;
  const { basisDecay } = new RS.RetirementSimulator(cfg).marketPath(360, new RS.Rng(0));
  approx(basisDecay.reduce((a, b) => a * b, 1.0), 1.0, { rel: 1e-12 });
});

test('inflation alone creates a taxable gain: known answer to the cent', () => {
  // $500k brokerage bought at par, 0% real, 3% inflation, age 35 -> retire 65 ->
  // die 66. Basis decays for the 361 months up to and including the plan month:
  //   basis = 500,000 / 1.03^(361/12) = 205,486.5944 -> gain fraction 58.9027%
  // single filer, 0% federal LTCG, 5% state over the 18,150 deduction:
  //   G - 0.05*(G*f - 18,150) = 60,000 -> G = 60,885.6644
  // estate = 500,000 - G = 439,114.3356 (440,000 with a flat real basis)
  const cfg = deterministicCfg({ realReturn: 0.0, spend: 60000.0, roth: 0.0,
                                 retireAge: 65, deathAge: 66 });
  cfg.simulation.current_age = 35;
  cfg.accounts.brokerage = 500000.0;
  cfg.accounts.brokerage_cost_basis = 500000.0;
  cfg.taxes.filing_status = 'single';
  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 0, false);

  const basis = 500000.0 / Math.pow(1.03, ((65 - 35) * 12 + 1) / 12);
  const gainFraction = 1 - basis / 500000.0;
  const gross = (60000.0 - 0.05 * 18150.0) / (1 - 0.05 * gainFraction);
  assert.strictEqual(r.survived, true);
  approx(r.final, 500000.0 - gross, { abs: 0.01 });
  approx(r.final, 439114.3356, { abs: 0.01 });
  assert.ok(r.final < 440000.0, 'a flat real basis would owe nothing');
});

// ------------------------------------------------------- trajectory sample sizes
test('a broke run records zeros through its OWN death, not to death_age_max', () => {
  const cfg = deterministicCfg({ realReturn: 0.0, spend: 100000.0, roth: 1000000.0,
                                 retireAge: 65, deathAge: 95 });
  const r = new RS.RetirementSimulator(cfg).simulateLife(65, 0, true);
  assert.strictEqual(r.survived, false);
  assert.strictEqual(r.trajectory.length, 30);          // 95 - 65, not truncated
  for (let i = 0; i < 10; i++)
    approx(r.trajectory[i], 1000000.0 - i * 100000.0 - 100000.0 / 12, { abs: 0.01 });
  assert.ok(r.trajectory.slice(10).every(v => v === 0.0), 'alive and broke, not absent');
});

test('a trajectory is always exactly one entry per year lived in retirement', () => {
  const cfg = baseCfg();
  cfg.simulation.current_age = 60;
  cfg.accounts.roth = 700000.0;                   // thin enough to produce failures
  cfg.accounts.traditional = cfg.accounts.brokerage = cfg.accounts.cash = 0.0;
  cfg.accounts.brokerage_cost_basis = 0.0;
  cfg.spending.initial_annual_expenses = 65000.0;
  const sim = new RS.RetirementSimulator(cfg);

  const outcomes = new Set();
  for (const seed of sim.runSeeds(60, 300)) {
    // death is drawn FIRST from the run's own generator, so it is recomputable
    const death = sim.drawDeathAge(cfg.life_events, 60, new RS.Rng(seed));
    const r = sim.simulateLife(60, seed, true);
    outcomes.add(r.survived);
    const livingMonths = Math.max(Math.round((death - 60) * 12), 1);
    assert.strictEqual(r.trajectory.length, Math.ceil(livingMonths / 12));
  }
  assert.strictEqual(outcomes.size, 2, 'the sample must cover both outcomes');
});

test('percentile columns count only the living, and n falls monotonically', () => {
  const cfg = baseCfg();
  cfg.simulation.current_age = 60;
  cfg.simulation.min_retirement_age = 60;
  cfg.simulation.max_retirement_age = 60;
  cfg.accounts.roth = 700000.0;
  cfg.accounts.traditional = cfg.accounts.brokerage = cfg.accounts.cash = 0.0;
  cfg.accounts.brokerage_cost_basis = 0.0;
  cfg.spending.initial_annual_expenses = 65000.0;
  const sim = new RS.RetirementSimulator(cfg);

  const n = 200;
  const { ages, percentiles } = sim.computeTrajectoryPercentiles(60, n);
  const counts = percentiles.n;
  const expected = new Array(counts.length).fill(0);
  for (const seed of sim.runSeeds(60, n)) {
    const traj = sim.simulateLife(60, seed, true).trajectory;
    for (let i = 0; i < Math.min(traj.length, counts.length); i++) expected[i]++;
  }
  assert.strictEqual(ages[0], 60);
  assert.deepStrictEqual(Array.from(counts), expected);
  assert.strictEqual(counts[0], n);                  // everyone is alive at 60
  for (let i = 0; i + 1 < counts.length; i++) assert.ok(counts[i] >= counts[i + 1]);
});

test('trajectory percentiles have the documented shape', () => {
  const cfg = baseCfg();
  cfg.simulation.current_age = 60;
  const sim = new RS.RetirementSimulator(cfg);
  const { ages, percentiles } = sim.computeTrajectoryPercentiles(65, 40);
  assert.strictEqual(ages.length, percentiles.n.length);
  for (const p of [1, 10, 25, 50])
    assert.strictEqual(percentiles[p].length, ages.length);
  // the fan is ordered at every column that has samples
  for (let i = 0; i < ages.length; i++) {
    if (!percentiles.n[i]) continue;
    assert.ok(percentiles[1][i] <= percentiles[10][i] + 1e-9);
    assert.ok(percentiles[10][i] <= percentiles[25][i] + 1e-9);
    assert.ok(percentiles[25][i] <= percentiles[50][i] + 1e-9);
  }
});
