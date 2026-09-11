'use strict';
/**
 * Method-level tests of the Social Security model against SSA's published rules.
 *
 * Claim-age factors: ssa.gov/oact/quickcalc/early_late.html
 * PIA formula and 2026 bend points: ssa.gov/oact/cola/piaformula.html
 * Eligibility credits: ssa.gov/benefits/retirement/planner/credits.html
 */

const { test } = require('node:test');
const assert = require('node:assert');
const { RS, baseCfg, approx } = require('./helpers.js');

const R = RS.RetirementSimulator;
const sim = () => new RS.RetirementSimulator(baseCfg());

// ---------------------------------------------------------------- claim factors
test('own-record claim factors at 62 / 67 / 70', () => {
  // 62 -> 1 - (36 x 5/9 + 24 x 5/12)/100 = 0.70
  // 70 -> 1 + 36 x (2/3)/100 = 1.24 (8%/yr SIMPLE, not compounded)
  approx(R.ssBenefitFactor(62), 0.70, { abs: 5e-7 });
  approx(R.ssBenefitFactor(67), 1.00, { abs: 5e-7 });
  approx(R.ssBenefitFactor(70), 1.24, { abs: 5e-7 });
});

test('claims are only possible 62-70; out-of-range ages clamp', () => {
  assert.strictEqual(R.ssBenefitFactor(55), R.ssBenefitFactor(62));
  assert.strictEqual(R.ssBenefitFactor(80), R.ssBenefitFactor(70));
});

test('own-record factors at EVERY claim age 62-70', () => {
  const expected = { 62: 0.700000, 63: 0.750000, 64: 0.800000, 65: 0.866667,
                     66: 0.933333, 67: 1.000000, 68: 1.080000, 69: 1.160000,
                     70: 1.240000 };
  for (const [age, f] of Object.entries(expected))
    approx(R.ssBenefitFactor(Number(age)), f, { abs: 5e-7 });
});

test('spousal factors at every claim age: steeper cut, no delayed credits', () => {
  const expected = { 62: 0.650000, 63: 0.700000, 64: 0.750000, 65: 0.833333,
                     66: 0.916667, 67: 1.000000, 68: 1.000000, 69: 1.000000,
                     70: 1.000000 };
  for (const [age, f] of Object.entries(expected))
    approx(R.ssSpousalFactor(Number(age)), f, { abs: 5e-7 });
});

test('the full_retirement_age argument actually moves the schedule', () => {
  approx(R.ssBenefitFactor(66, 66), 1.0, { abs: 5e-7 });
  approx(R.ssBenefitFactor(62, 66), 0.75, { abs: 5e-7 });   // 25% cut, not 30%
  approx(R.ssSpousalFactor(66, 66), 1.0, { abs: 5e-7 });
});

// ---------------------------------------------------------------- PIA formula
test('the AIME<->PIA inversion is exact in all three segments', () => {
  for (const pia of [500.0, 2000.0, 4000.0])
    approx(R.piaFromAime(R.aimeFromPia(pia)), pia, { rel: 1e-12 });
});

test('the formula at the 2026 bend points themselves', () => {
  // PIA(1,286) = 0.90 x 1,286 = 1,157.40
  // PIA(7,749) = 1,157.40 + 0.32 x 6,463 = 3,225.56
  approx(R.piaFromAime(1286.0), 1157.40, { abs: 1e-6 });
  approx(R.piaFromAime(7749.0), 3225.56, { abs: 1e-6 });
});

test('a 60% career pays MORE than 60% of the full benefit', () => {
  const s = sim();
  const cfg = baseCfg();
  const le = RS.deepClone(cfg.life_events);
  le.ss_earnings_years_at_current_age = 21;
  const partial = s.ssFraBenefit(le, 0);
  const full = s.ssFraBenefit(cfg.life_events, 35);
  approx(full, 40000.0, { abs: 1.0 });
  assert.ok(partial / full > 21 / 35 + 0.02,
            `progressivity should beat linear: got ${partial / full}`);
});

test('the 40-credit eligibility cliff', () => {
  const s = sim();
  const le = RS.deepClone(baseCfg().life_events);
  le.ss_credits_at_current_age = 8;
  assert.strictEqual(s.ssFraBenefit(le, 0), 0.0);
  assert.ok(s.ssFraBenefit(le, 10) > 0.0);
});

// ---------------------------------------------------------------- household
test('a stay-at-home spouse collects exactly 50% of the primary PIA at FRA', () => {
  const cfg = baseCfg();
  cfg.spouse.enabled = true;
  const s = new RS.RetirementSimulator(cfg);
  const primaryPia = s.ssFraBenefit(cfg.life_events, 65 - cfg.simulation.current_age);
  approx(s.spouseSsIncome(65), 0.5 * primaryPia, { abs: 0.01 });
});

test('an early death caps the work record', () => {
  const cfg = baseCfg();
  cfg.spouse.enabled = true;
  const s = new RS.RetirementSimulator(cfg);
  const full = s.spouseSsIncome(65);
  const diedYoung = s.spouseSsIncome(65, 10.0, 10.0);
  assert.ok(diedYoung < full - 1.0);
  const defaultYears = 65 - cfg.simulation.current_age;
  approx(s.spouseSsIncome(65, defaultYears, defaultYears), full, { rel: 1e-12 });
});

test('the primary benefit is non-decreasing in retirement age', () => {
  const s = sim();
  const incomes = [40, 50, 60, 70].map(a => s.primarySsIncome(a));
  for (let i = 1; i < incomes.length; i++)
    assert.ok(incomes[i - 1] <= incomes[i] + 1e-9);
  approx(incomes[incomes.length - 1], 40000.0, { abs: 1.0 });
});

// ---------------------------------------------------------------- survivors
test('the survivor factor follows life, not intent', () => {
  approx(R.ssSurvivorFactor(70, 63), 1.00, { abs: 1e-9 });   // no DRCs earned
  approx(R.ssSurvivorFactor(70, 68), 1.08, { abs: 1e-9 });   // one year of DRCs
  approx(R.ssSurvivorFactor(70, 71), 1.24, { abs: 1e-9 });   // filed and lived
  approx(R.ssSurvivorFactor(67, 63), 1.00, { abs: 1e-9 });   // died before FRA
  approx(R.ssSurvivorFactor(70, 75), 1.24, { abs: 1e-9 });   // DRCs stop at 70
});

test("the widow's limit floors an early claim at 82.5% of PIA", () => {
  approx(R.ssSurvivorFactor(62, 70), 0.825, { abs: 1e-9 });
  approx(R.ssSurvivorFactor(63, 70), 0.825, { abs: 1e-9 });
  approx(R.ssSurvivorFactor(65, 70), 0.866667, { abs: 5e-7 });   // above the floor
});

test("the survivor's OWN early-claim reduction", () => {
  // 28.5% at 60 prorated over the 84 months to FRA
  approx(R.ssSurvivorClaimFactor(62), 1 - 0.285 * 60 / 84, { abs: 1e-9 });
  approx(R.ssSurvivorClaimFactor(65), 1 - 0.285 * 24 / 84, { abs: 1e-9 });
  approx(R.ssSurvivorClaimFactor(67), 1.0, { abs: 1e-9 });
  approx(R.ssSurvivorClaimFactor(70), 1.0, { abs: 1e-9 });
});

test('both survivor reductions combine correctly', () => {
  const at62 = R.ssSurvivorClaimFactor(62);
  // deceased claimed at 62 (floored to 0.825); survivor claims at 62 -> their own
  // 0.796 is below the cap, so that is what they get
  approx(R.ssSurvivorFactor(62, 70, 67, 62), at62, { abs: 1e-9 });
  // survivor at FRA: the cap binds
  approx(R.ssSurvivorFactor(62, 70, 67, 67), 0.825, { abs: 1e-9 });
  // deceased filed at 70 and lived: 1.24 x the survivor's own reduction
  approx(R.ssSurvivorFactor(70, 71, 67, 62), 1.24 * at62, { abs: 1e-9 });
  // died before filing at 63: the plain PIA, reduced for the survivor's age
  approx(R.ssSurvivorFactor(70, 63, 67, 62), at62, { abs: 1e-9 });
});

test('survivor income end to end, in dollars', () => {
  // The deceased died at 63 having PLANNED to claim at 70. They earned no delayed
  // credits, so the survivor gets the PIA ($50,000), not 1.24 x PIA ($62,000).
  const cfg = baseCfg();
  cfg.simulation.current_age = 60;
  cfg.spouse.enabled = true;
  cfg.spouse.age_offset = 0;
  cfg.spouse.ss_annual_full_retirement_benefit = 50000.0;
  cfg.spouse.ss_earnings_years_at_current_age = 35;
  cfg.spouse.ss_credits_at_current_age = 40;
  cfg.spouse.ss_claim_age = 70;
  cfg.life_events.ss_annual_full_retirement_benefit = 0.0;
  cfg.life_events.ss_earnings_years_at_current_age = 0;
  cfg.life_events.ss_credits_at_current_age = 0;
  const s = new RS.RetirementSimulator(cfg);

  const spouseFra = s.ssFraBenefit(cfg.spouse, 3.0);
  approx(spouseFra, 50000.0, { abs: 1.0 });
  approx(spouseFra * R.ssSurvivorFactor(70, 63.0), 50000.0, { abs: 1.0 });
  approx(spouseFra * R.ssBenefitFactor(70), 62000.0, { abs: 1.0 });
});

test('a spousal benefit is never inheritable', () => {
  const cfg = baseCfg();
  cfg.spouse.enabled = true;
  const s = new RS.RetirementSimulator(cfg);
  const years = 65 - cfg.simulation.current_age;
  assert.ok(s.spouseSsIncome(65) > 0.0);                 // spousal while alive
  assert.strictEqual(s.ssFraBenefit(cfg.spouse, years), 0.0);   // nothing to inherit
});
