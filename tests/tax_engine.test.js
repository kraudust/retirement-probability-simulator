'use strict';
/**
 * Method-level tests of TaxCalculator against hand-computed 2026 tax law.
 * Ported from tests/test_tax_engine.py -- every expected value is derived from
 * the law (IRS Rev. Proc. 2025-32; IRC 86 for Social Security; IRC 1411 for
 * NIIT; IRC 63(f) for the age-65 addition), not from what either engine prints,
 * so the two suites assert the identical numbers.
 */

const { test } = require('node:test');
const assert = require('node:assert');
const { RS, baseCfg, approx } = require('./helpers.js');

const tax = () => new RS.TaxCalculator(baseCfg().taxes);
const MFJ = 'married_filing_jointly';

// ---------------------------------------------------------------- ordinary income
test('MFJ, $100,000 ordinary income, both spouses 70', () => {
  // deduction = 32,200 + 2 x 1,650 (IRC 63(f), both 65+) = 35,500
  // taxable = 100,000 - 35,500 = 64,500
  // federal = 24,800 x 10% + 39,700 x 12% = 2,480 + 4,764 = 7,244
  // state   = 5% x 64,500 = 3,225
  approx(tax().totalTax(100000, 0, 0, 0, MFJ, 70, 0, 70), 10469.0, { abs: 1e-6 });
});

test('the widow tax penalty: the same $100,000 filed single at 70', () => {
  // deduction = 16,100 + 2,050 = 18,150; taxable = 81,850
  // federal = 12,400x10 + 38,000x12 + 31,450x22 = 1,240 + 4,560 + 6,919 = 12,719
  // state   = 5% x 81,850 = 4,092.50
  approx(tax().totalTax(100000, 0, 0, 0, 'single', 70), 16811.5, { abs: 1e-6 });
});

test('taxable income landing exactly on a bracket top is all taxed at that rate', () => {
  // ordinary = 24,800 + 35,500 so taxable is exactly 24,800 -> federal 2,480
  approx(tax().taxBreakdown(60300, 0, 0, 0, MFJ, 70, 0, 70).federal_ordinary,
         2480.0, { abs: 1e-6 });
});

test('income below the standard deduction owes nothing', () => {
  approx(tax().totalTax(30000, 0, 0, 0, MFJ, 70, 0, 70), 0.0, { abs: 1e-9 });
});

// ------------------------------------------- the age-65 addition (IRC 63(f))
test('63(f) is claimed once per filer 65+, twice only on a joint return', () => {
  const t = tax();
  approx(t.totalStandardDeduction(MFJ, 64, 64), 32200.0, { abs: 1e-9 });
  approx(t.totalStandardDeduction(MFJ, 65, 64), 33850.0, { abs: 1e-9 });
  approx(t.totalStandardDeduction(MFJ, 70, 70), 35500.0, { abs: 1e-9 });
  // a lone survivor on a joint-rate return claims it once, not twice
  approx(t.totalStandardDeduction(MFJ, 70, null), 33850.0, { abs: 1e-9 });
  approx(t.totalStandardDeduction('single', 64), 16100.0, { abs: 1e-9 });
  approx(t.totalStandardDeduction('single', 65), 18150.0, { abs: 1e-9 });
  // the base accessor stays the under-65 figure
  approx(t.standardDeduction('single'), 16100.0, { abs: 1e-9 });
});

test('the age-65 addition switches on at 65, not 64', () => {
  // at 64: taxable 43,900 -> 12,400x10 + 31,500x12 = 5,020; state 2,195
  // at 65: taxable 41,850 -> 12,400x10 + 29,450x12 = 4,774; state 2,092.50
  approx(tax().totalTax(60000, 0, 0, 0, 'single', 64), 7215.0, { abs: 1e-6 });
  approx(tax().totalTax(60000, 0, 0, 0, 'single', 65), 6866.5, { abs: 1e-6 });
});

// ---------------------------------------------------------------- capital gains
test('MFJ $20k ordinary + $60k gains fits entirely in the 0% LTCG band', () => {
  // taxable_total = 80,000 - 35,500 = 44,500 < 98,900
  approx(tax().taxBreakdown(20000, 60000, 0, 60000, MFJ, 70, 0, 70).federal_ltcg,
         0.0, { abs: 1e-9 });
});

test('gains stack on top of ordinary income when choosing the LTCG rate', () => {
  // taxable_ordinary 54,500; taxable_total 114,500
  // only the slice above the 98,900 band top is taxed: 15,600 x 15% = 2,340
  approx(tax().taxBreakdown(90000, 60000, 0, 60000, MFJ, 70, 0, 70).federal_ltcg,
         2340.0, { abs: 1e-6 });
});

test("the single filer's 0% band is about half the MFJ band", () => {
  const t = tax();
  // 60k gains, no ordinary: taxable_total 41,850 -> still 0%
  approx(t.taxBreakdown(0, 60000, 0, 60000, 'single', 70).federal_ltcg, 0.0, { abs: 1e-9 });
  // 80k gains: taxable_total 61,850 -> (61,850 - 49,450) x 15% = 1,860
  approx(t.taxBreakdown(0, 80000, 0, 80000, 'single', 70).federal_ltcg, 1860.0, { abs: 1e-6 });
});

test('the 37% ordinary and 20% LTCG top bands', () => {
  // single, $900,000 ordinary: taxable = 881,850
  //   12,400x10 + 38,000x12 + 55,300x22 + 96,075x24 + 54,450x32 + 384,375x35
  //   + 241,250x37 = 282,241.75
  approx(tax().taxBreakdown(900000, 0, 0, 0, 'single', 70).federal_ordinary,
         282241.75, { abs: 1e-4 });
  // single, no ordinary, $700k gains -> taxable_total 681,850
  //   49,450 at 0% + 496,050 x 15% + 136,350 x 20% = 74,407.50 + 27,270 = 101,677.50
  approx(tax().taxBreakdown(0, 700000, 0, 700000, 'single', 70).federal_ltcg,
         101677.5, { abs: 1e-4 });
});

// ---------------------------------------------------------------- Social Security
test('$40k SS alone is untaxed: provisional 20,000 < 25,000', () => {
  approx(tax().taxableSocialSecurity(40000, 0, 'single'), 0.0, { abs: 1e-9 });
});

test('the MIDDLE tier, limited by half the excess over the first threshold', () => {
  // provisional = 18,000 + 10,000 = 28,000, inside 25,000..34,000
  // taxable = min(0.5 x 3,000, 0.5 x 20,000) = 1,500
  approx(tax().taxableSocialSecurity(20000, 18000, 'single'), 1500.0, { abs: 1e-6 });
});

test('the middle tier, limited instead by half the benefit', () => {
  const t = tax();
  approx(t.taxableSocialSecurity(4000, 26000, 'single'), 1500.0, { abs: 1e-6 });
  approx(t.taxableSocialSecurity(4000, 31000, 'single'), 2000.0, { abs: 1e-6 });
});

test('no jump discontinuity at either IRC 86 kink', () => {
  const t = tax();
  const below = t.taxableSocialSecurity(40000, 13999.99, 'single');
  const at = t.taxableSocialSecurity(40000, 14000.0, 'single');    // provisional 34,000
  const above = t.taxableSocialSecurity(40000, 14000.01, 'single');
  approx(at, 4500.0, { abs: 1e-6 });                               // the tier-1 amount
  approx(below, at, { abs: 0.02 });
  approx(above, at, { abs: 0.02 });
});

test('above the second threshold', () => {
  // provisional 40,000 > 34,000; tier1 = 4,500
  // taxable = min(0.85 x 6,000 + 4,500, 0.85 x 40,000) = 9,600
  approx(tax().taxableSocialSecurity(40000, 20000, 'single'), 9600.0, { abs: 1e-6 });
});

test('the taxable share caps at exactly 85% of the benefit', () => {
  approx(tax().taxableSocialSecurity(40000, 500000, 'single'), 34000.0, { abs: 1e-6 });
});

test('the MFJ thresholds leave the same case untaxed', () => {
  approx(tax().taxableSocialSecurity(40000, 10000, MFJ), 0.0, { abs: 1e-9 });
});

test('state tax is NOT levied on taxable Social Security', () => {
  // most states exempt SS entirely; the flat rate applies to ordinary + gains only
  const bd = tax().taxBreakdown(40000, 0, 40000, 0, 'single', 70);
  assert.ok(bd.taxable_social_security > 0, 'this case should have taxable SS');
  approx(bd.state, 0.05 * Math.max(0, 40000 - 18150), { abs: 1e-6 });
});

// ---------------------------------------------------------------- NIIT
test('NIIT is 3.8% of the LESSER of investment income and the AGI excess', () => {
  // single, 150k ordinary + 100k gains: AGI 250k
  // base = min(250,000 - 200,000, 100,000) = 50,000 -> 1,900
  approx(tax().taxBreakdown(150000, 100000, 0, 100000, 'single', 70).niit,
         1900.0, { abs: 1e-6 });
});

test('no NIIT below the threshold, however large investment income is', () => {
  approx(tax().taxBreakdown(0, 100000, 0, 100000, 'single', 70).niit, 0.0, { abs: 1e-9 });
});

// ---------------------------------------------------------------- early penalty
test('10% of the traditional distribution before 59.5', () => {
  approx(tax().taxBreakdown(10000, 0, 0, 0, 'single', 50, 10000).early_penalty,
         1000.0, { abs: 1e-6 });
});

test('no penalty from 59.5 onward, and the boundary is exact', () => {
  const t = tax();
  approx(t.taxBreakdown(10000, 0, 0, 0, 'single', 60, 10000).early_penalty, 0.0, { abs: 1e-9 });
  approx(t.taxBreakdown(10000, 0, 0, 0, 'single', 59.49, 10000).early_penalty,
         1000.0, { abs: 1e-6 });
  approx(t.taxBreakdown(10000, 0, 0, 0, 'single', 59.5, 10000).early_penalty, 0.0, { abs: 1e-9 });
});

test('a 72(t) SEPP / Roth-ladder plan disables the penalty at any age', () => {
  const cfg = baseCfg();
  cfg.taxes.use_72t_sepp = true;
  const bd = new RS.TaxCalculator(cfg.taxes).taxBreakdown(10000, 0, 0, 0, 'single', 45, 10000);
  approx(bd.early_penalty, 0.0, { abs: 1e-9 });
});

test('rule-of-55 needs BOTH age 55+ and separation at 55+', () => {
  const cfg = baseCfg();
  cfg.taxes.assume_qualified_plan_age55_exception = true;
  const calc = new RS.TaxCalculator(cfg.taxes);
  // separated at 56, now 56: the exception applies
  approx(calc.taxBreakdown(10000, 0, 0, 0, 'single', 56, 10000, null, 56).early_penalty,
         0.0, { abs: 1e-9 });
  // below 55: no exception regardless
  approx(calc.taxBreakdown(10000, 0, 0, 0, 'single', 53, 10000, null, 53).early_penalty,
         1000.0, { abs: 1e-6 });
  // THE BUG THIS PINS: retired at 45, now 56. The law requires separation from
  // service in or after the year you turn 55, so the penalty still applies.
  approx(calc.taxBreakdown(10000, 0, 0, 0, 'single', 56, 10000, null, 45).early_penalty,
         1000.0, { abs: 1e-6 });
});

// ---------------------------------------------------------------- consistency
test('the itemised breakdown reconciles with totalTax exactly', () => {
  const t = tax();
  const bd = t.taxBreakdown(80000, 30000, 25000, 35000, 'single', 50, 40000);
  const parts = bd.federal_ordinary + bd.federal_ltcg + bd.state + bd.niit + bd.early_penalty;
  approx(bd.total, parts, { abs: 1e-9 });
  approx(t.totalTax(80000, 30000, 25000, 35000, 'single', 50, 40000), bd.total, { abs: 1e-9 });
});

test('MFJ becomes single after the first death; single filers stay single', () => {
  const sim = new RS.RetirementSimulator(baseCfg());
  assert.strictEqual(sim.householdFilingStatus(true, true), MFJ);
  assert.strictEqual(sim.householdFilingStatus(true, false), 'single');
  assert.strictEqual(sim.householdFilingStatus(false, true), 'single');
});
