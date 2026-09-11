'use strict';
/**
 * Tests of the capital-market model: rate conversions, the regime chain, the
 * Student-t scaling, correlation, and the numbers assumptionReport prints.
 *
 * Every other fixture deliberately switches the market model OFF (zero
 * volatility, crises disabled) so outcomes are hand-computable. That is right
 * for the accounting, but it leaves the market model itself unasserted -- a
 * broken t-scaling, a dropped correlation term or a biased regime chain would
 * fail nothing. These are statistical assertions over large samples: the
 * tolerances are the point, so they port across a different generator unchanged.
 */

const { test } = require('node:test');
const assert = require('node:assert');
const { RS, baseCfg, approx } = require('./helpers.js');

const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const std = xs => { const m = mean(xs); return Math.sqrt(mean(xs.map(x => (x - m) ** 2))); };
const log1p = Math.log1p;

/** n independent monthly log-return paths from the engine's own generator. */
function paths(sim, months, n, seed = 0) {
  const rng = new RS.Rng(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const p = sim.marketPath(months, new RS.Rng(Math.floor(rng.random() * 2 ** 31)));
    out.push([p.stockGrowth.map(log1p), p.bondGrowth.map(log1p), p.cashGrowth]);
  }
  return out;
}

// ---------------------------------------------------------------- rate conversion
test('real_return divides rather than subtracts', () => {
  const cfg = baseCfg();
  cfg.market.inflation = 0.03;
  const sim = new RS.RetirementSimulator(cfg);
  // 8% nominal at 3% inflation is 4.854% real, NOT 5%
  approx(sim.realReturn(0.08), 0.04854368932, { abs: 1e-9 });
  approx(sim.realReturn(0.03), 0.0, { abs: 1e-15 });
});

test('monthly rates compound back to the annual rate', () => {
  for (const annual of [0.04, 0.08, -0.02, 0.5]) {
    const monthly = RS.RetirementSimulator.monthlyRate(annual);
    approx(Math.pow(1 + monthly, 12) - 1, annual, { abs: 1e-12 });
    assert.ok(monthly < annual / 12 + 1e-12, 'must be below the naive r/12');
  }
});

// ---------------------------------------------------------------- the regime chain
test('the crisis fraction is the chain stationary distribution', () => {
  const cfg = baseCfg();
  const sim = new RS.RetirementSimulator(cfg);
  const pEnter = cfg.simulation.normal_regime.monthly_crisis_probability;
  const pExit = cfg.simulation.crisis_regime.monthly_recovery_probability;
  approx(sim.crisisFraction, pEnter / (pEnter + pExit), { rel: 1e-12 });
});

test('drag compensation delivers the configured return over a real horizon', () => {
  // The regression guard for the chain's starting state: seeding it in 'normal'
  // gave every life ~4% too much wealth, because a finite path then spends less
  // time in crisis than the stationary fraction the compensation assumes.
  const cfg = baseCfg();
  cfg.market.stock_volatility = 0.0;          // isolate the regime chain
  cfg.market.inflation_volatility = 0.0;
  const sim = new RS.RetirementSimulator(cfg);
  const target = log1p(sim.realReturn(cfg.market.stock_return));

  const months = 240;
  const realised = paths(sim, months, 4000, 3)
    .map(p => p[0].reduce((a, b) => a + b, 0) / (months / 12));
  approx(mean(realised), target, { abs: 0.0035 });
});

test('with compensation OFF, crises pull the return well below the input', () => {
  const cfg = baseCfg();
  cfg.market.stock_volatility = 0.0;
  cfg.market.inflation_volatility = 0.0;
  cfg.simulation.compensate_crisis_drag = false;
  const sim = new RS.RetirementSimulator(cfg);
  const target = log1p(sim.realReturn(cfg.market.stock_return));
  const months = 240;
  const realised = paths(sim, months, 1500, 4)
    .map(p => p[0].reduce((a, b) => a + b, 0) / (months / 12));
  assert.ok(mean(realised) < target - 0.015, 'the flag must actually do something');
});

test('crisis spells last the configured length', () => {
  // mean spell = 1/monthly_recovery_probability months (~18 at 0.055)
  const cfg = baseCfg();
  cfg.market.stock_volatility = 0.0;      // no shocks, so the log return IS the drift
  cfg.market.inflation_volatility = 0.0;
  const sim = new RS.RetirementSimulator(cfg);
  const stock = paths(sim, 400000, 1, 5)[0][0];
  // reduce, not Math.min(...stock): spreading a 400k array blows the call stack
  const lo = stock.reduce((a, b) => (b < a ? b : a), Infinity);
  const inCrisis = stock.map(v => Math.abs(v - lo) < 1e-12);

  const spells = [];
  let run = 0;
  for (const f of inCrisis) { if (f) run++; else if (run) { spells.push(run); run = 0; } }
  const expected = 1 / cfg.simulation.crisis_regime.monthly_recovery_probability;
  approx(mean(spells), expected, { rel: 0.06 });
  approx(inCrisis.filter(Boolean).length / inCrisis.length, sim.crisisFraction,
         { abs: 0.01 });
});

// ---------------------------------------------------------------- shock distribution
test('realised stock volatility matches the configured calm figure', () => {
  // the Student-t draws are rescaled by sqrt((df-2)/df) for unit variance, so
  // stock_volatility means what it says. Drop that and df=6 inflates it by 22%.
  const cfg = baseCfg();
  cfg.simulation.normal_regime.monthly_crisis_probability = 0.0;
  cfg.market.inflation_volatility = 0.0;
  const stock = paths(new RS.RetirementSimulator(cfg), 400000, 1, 6)[0][0];
  approx(std(stock) * Math.sqrt(12), cfg.market.stock_volatility, { rel: 0.03 });
});

test('the tails are actually fat', () => {
  // excess kurtosis of a t(6) is 6/(df-4) = 3; a normal would be ~0
  const cfg = baseCfg();
  cfg.simulation.normal_regime.monthly_crisis_probability = 0.0;
  cfg.market.inflation_volatility = 0.0;
  const stock = paths(new RS.RetirementSimulator(cfg), 400000, 1, 7)[0][0];
  const m = mean(stock), s = std(stock);
  const kurt = mean(stock.map(x => ((x - m) / s) ** 4)) - 3;
  assert.ok(kurt > 1.5, `excess kurtosis ${kurt} is too thin-tailed`);
});

test('the stock/bond correlation holds within a regime', () => {
  // bond shocks are built from STANDARDISED draws, so the configured correlation
  // survives the crisis volatility multiplier
  const cfg = baseCfg();
  cfg.simulation.normal_regime.monthly_crisis_probability = 0.0;
  cfg.market.inflation_volatility = 0.0;
  for (const corr of [-0.3, 0.0, 0.5]) {
    cfg.market.stock_bond_correlation = corr;
    const [stock, bond] = paths(new RS.RetirementSimulator(cfg), 300000, 1, 8)[0];
    const ms = mean(stock), mb = mean(bond);
    const cov = mean(stock.map((v, i) => (v - ms) * (bond[i] - mb)));
    approx(cov / (std(stock) * std(bond)), corr, { abs: 0.015 });
    approx(std(bond) * Math.sqrt(12), cfg.market.bond_volatility, { rel: 0.03 });
  }
});

test('inflation shocks erode bonds and cash but not stocks', () => {
  const cfg = baseCfg();
  cfg.simulation.normal_regime.monthly_crisis_probability = 0.0;
  cfg.market.bond_volatility = 0.0;
  const quiet = RS.deepClone(cfg); quiet.market.inflation_volatility = 0.0;
  const noisy = RS.deepClone(cfg); noisy.market.inflation_volatility = 0.04;

  const [qs, qb, qc] = paths(new RS.RetirementSimulator(quiet), 150000, 1, 9)[0];
  const [ns, nb, nc] = paths(new RS.RetirementSimulator(noisy), 150000, 1, 9)[0];
  approx(std(qb), 0.0, { abs: 1e-12 });
  assert.ok(std(nb) > 0.005, 'inflation must move bonds');
  assert.ok(std(nc) > std(qc), 'inflation must move cash');
  approx(std(ns), std(qs), { rel: 0.03 });          // stocks untouched
});

// ---------------------------------------------------------------- spending smile
test('the spending smile clamps and compounds at the right boundaries', () => {
  const cfg = baseCfg();
  const sim = new RS.RetirementSimulator(cfg);
  const s = cfg.spending;

  assert.deepStrictEqual(sim.spendingSmile(s.spending_decline_start_age - 5), [1.0, 1.0]);
  assert.deepStrictEqual(sim.spendingSmile(s.spending_decline_start_age), [1.0, 1.0]);

  const [midBase, midHealth] = sim.spendingSmile(s.spending_decline_start_age + 5);
  approx(midBase, Math.pow(s.annual_spending_decline_rate, 5), { rel: 1e-12 });
  approx(midHealth, 1.0, { rel: 1e-12 });

  const years = s.spending_decline_end_age - s.spending_decline_start_age;
  const [endBase] = sim.spendingSmile(s.spending_decline_end_age);
  approx(endBase, Math.pow(s.annual_spending_decline_rate, years), { rel: 1e-12 });

  // past the end: base frozen, healthcare compounding
  const [lateBase, lateHealth] = sim.spendingSmile(s.spending_decline_end_age + 10);
  approx(lateBase, endBase, { rel: 1e-12 });
  approx(lateHealth, Math.pow(1 + s.annual_healthcare_increase_rate, 10), { rel: 1e-12 });
});

// ------------------------------------------------------- what the report promises
test('the report\'s "after regimes" return is the one delivered', () => {
  const cfg = baseCfg();
  cfg.market.stock_volatility = 0.0;
  cfg.market.inflation_volatility = 0.0;
  const sim = new RS.RetirementSimulator(cfg);

  const line = sim.assumptionReport().split('\n').find(l => l.includes('stock return'));
  const printed = Number(line.split('->').pop().trim().replace(/[^0-9.\-]/g, '')) / 100;

  const months = 240;
  const realised = paths(sim, months, 4000, 11)
    .map(p => p[0].reduce((a, b) => a + b, 0) / (months / 12));
  approx(Math.expm1(mean(realised)), printed, { abs: 0.0035 });
});

test('the report\'s effective volatility is the one delivered', () => {
  // must include BOTH the within-regime variance mixture and the dispersion of
  // the regime means, with regime persistence inside a year
  const cfg = baseCfg();
  const sim = new RS.RetirementSimulator(cfg);
  const line = sim.assumptionReport().split('\n').find(l => l.includes('stock volatility'));
  const printed = Number(line.split('->')[1].split('%')[0].trim()) / 100;

  const quiet = RS.deepClone(cfg);
  quiet.market.inflation_volatility = 0.0;
  const stock = paths(new RS.RetirementSimulator(quiet), 240000, 1, 12)[0][0];
  // annual log returns, so regime persistence inside the year is captured
  const annual = [];
  for (let i = 0; i + 12 <= stock.length; i += 12)
    annual.push(stock.slice(i, i + 12).reduce((a, b) => a + b, 0));
  approx(std(annual), printed, { abs: 0.012 });
});

test('the report mentions every value the model transforms', () => {
  const text = new RS.RetirementSimulator(baseCfg()).assumptionReport();
  for (const fragment of ['nominal', 'real', 'after regimes', 'crisis regime',
                          'effective', 'Social Security', 'healthcare',
                          'early withdrawal penalty', 'RMD', 'filing status'])
    assert.ok(text.includes(fragment), `assumption report is missing "${fragment}"`);
});
