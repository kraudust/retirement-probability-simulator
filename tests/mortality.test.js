'use strict';
/**
 * Distributional tests of the mortality models.
 *
 * The Gompertz curves are calibrated so survival CONDITIONAL ON REACHING 65
 * matches SSA cohort life-table targets (2025 Trustees Report, intermediate
 * assumptions). The targets below are the same ones the Python suite asserts --
 * they come from the life tables, not from either engine's generator, so the two
 * suites check the identical calibration despite using different RNGs.
 */

const { test } = require('node:test');
const assert = require('node:assert');
const { RS, baseCfg, approx } = require('./helpers.js');

const R = RS.RetirementSimulator;
const N_DRAWS = 40000;

// (sex, survival targets from 65 to [80, 90, 95, 100])
const CALIBRATION_TARGETS = [
  ['male', [0.70, 0.33, 0.15, 0.05]],
  ['female', [0.78, 0.44, 0.24, 0.085]],
];

const median = xs => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

for (const [sex, targets] of CALIBRATION_TARGETS) {
  test(`Gompertz survival from 65 matches the SSA targets (${sex})`, () => {
    const rng = new RS.Rng(7);
    const draws = Array.from({ length: N_DRAWS },
                             () => R.actuarialDeathAge(sex, 65.0, 120.0, rng));
    [80, 90, 95, 100].forEach((age, i) => {
      const measured = draws.filter(d => d >= age).length / draws.length;
      approx(measured, targets[i], { abs: 0.035 });
    });
  });
}

test('median death from 65 is ~86 male, ~89 female', () => {
  for (const [sex, med] of [['male', 86.0], ['female', 89.0]]) {
    const rng = new RS.Rng(7);
    const draws = Array.from({ length: N_DRAWS },
                             () => R.actuarialDeathAge(sex, 65.0, 120.0, rng));
    approx(median(draws), med, { abs: 1.0 });
  }
});

test('the Gompertz draw is capped at death_age_max and floored at today', () => {
  const cfg = baseCfg();
  const sim = new RS.RetirementSimulator(cfg);
  const rng = new RS.Rng(3);
  const draws = Array.from({ length: 2000 },
                           () => sim.drawDeathAge(cfg.life_events, 35, rng));
  assert.ok(Math.max(...draws) <= cfg.life_events.death_age_max);
  assert.ok(Math.min(...draws) >= 35.0);
});

test('a fractional age gap must not overshoot the cap', () => {
  // death_age_max also sizes the trajectory chart, so an overshoot would index
  // past the end of it.
  const rng = new RS.Rng(11);
  for (const [start, cap] of [[65.0, 70.5], [35.5, 80.0], [65.0, 70.0]]) {
    const draws = Array.from({ length: 3000 },
                             () => R.actuarialDeathAge('male', start, cap, rng));
    assert.ok(Math.max(...draws) <= cap + 1e-12, `overshot ${cap}`);
    assert.ok(Math.min(...draws) >= start);
  }
});

test('the clipped-normal model stays inside [min, max]', () => {
  const cfg = baseCfg();
  const sim = new RS.RetirementSimulator(cfg);
  const le = RS.deepClone(cfg.life_events);
  le.mortality_model = 'normal';
  const rng = new RS.Rng(3);
  const draws = Array.from({ length: 2000 }, () => sim.drawDeathAge(le, 35, rng));
  assert.ok(Math.min(...draws) >= le.death_age_min);
  assert.ok(Math.max(...draws) <= le.death_age_max);
});

test('std = 0 pins death to the mean exactly', () => {
  // the deterministic scenarios rely on this to build fixed-horizon retirements
  const cfg = baseCfg();
  const sim = new RS.RetirementSimulator(cfg);
  const le = RS.deepClone(cfg.life_events);
  le.mortality_model = 'normal';
  le.death_age_mean = 95;
  le.death_age_std = 0.0;
  le.death_age_min = 95;
  le.death_age_max = 95;
  const rng = new RS.Rng(3);
  for (let i = 0; i < 10; i++)
    assert.strictEqual(sim.drawDeathAge(le, 35, rng), 95.0);
});
