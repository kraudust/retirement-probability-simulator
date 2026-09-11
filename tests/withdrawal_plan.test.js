'use strict';
/**
 * Tests of the annual withdrawal-plan solver: exactness, money conservation,
 * ladder order, RMD mechanics, and its known-answer tax integration.
 *
 * The solver's contract: after-tax proceeds cover the year's net need to within
 * a cent, no dollar is created or destroyed, and every tax comes from the actual
 * withdrawal composition.
 */

const { test } = require('node:test');
const assert = require('node:assert');
const { RS, baseCfg, approx } = require('./helpers.js');

const sim = () => new RS.RetirementSimulator(baseCfg());

/** annualWithdrawalPlan with readable keyword defaults, mirroring run_plan(). */
function runPlan(s, kw = {}) {
  const a = Object.assign({
    netNeed: 80000.0,
    balances: [20000.0, 150000.0, 400000.0, 100000.0],
    brokerageBasis: 90000.0,
    householdAge: 55.0,
    status: 'married_filing_jointly',
    ssIncome: 0.0,
    stockAlloc: 0.7,
    bondAlloc: 0.3,
    rmdAmount: 0.0,
    spouseAge: null,
    separationAge: null,
  }, kw);
  return s.annualWithdrawalPlan(a.netNeed, a.balances, a.brokerageBasis, a.householdAge,
                                a.status, a.ssIncome, a.stockAlloc, a.bondAlloc,
                                a.rmdAmount, a.spouseAge, a.separationAge);
}

// ---------------------------------------------------------------- core contract
test('money conservation: outflow equals gross minus reinvested surplus', () => {
  const balances = [20000.0, 150000.0, 400000.0, 100000.0];
  const plan = runPlan(sim(), { balances });
  const outflow = balances.reduce((a, b) => a + b, 0)
                - plan.balances.reduce((a, b) => a + b, 0);
  approx(outflow, plan.gross_total - plan.surplus_reinvested, { abs: 0.01 });
});

test('after-tax proceeds reach the need and overshoot by less than a dollar', () => {
  const plan = runPlan(sim());
  const net = plan.gross_total - plan.tax;
  assert.ok(net >= 80000.0 - 0.01, `net ${net} fell short`);
  assert.ok(net < 80000.0 + 1.0, `net ${net} overshot`);
});

test('the cheapest rung (already-taxed cash) empties first', () => {
  approx(runPlan(sim()).take_cash, 20000.0, { abs: 1e-6 });
});

test('net proceeds rise with the gross withdrawal, so the bisection is valid', () => {
  const s = sim();
  const balances = [5000.0, 10000.0, 400000.0, 100000.0];
  const nets = [20000.0, 50000.0, 90000.0].map(need => {
    const p = runPlan(s, { netNeed: need, balances, brokerageBasis: 8000.0,
                           householdAge: 50.0 });
    return p.gross_total - p.tax;
  });
  assert.deepStrictEqual(nets, [...nets].sort((a, b) => a - b));
});

test('liquidating everything still short returns null, not a wrong plan', () => {
  assert.strictEqual(
    runPlan(sim(), { balances: [1000.0, 0.0, 0.0, 0.0], brokerageBasis: 0.0 }), null);
});

// ---------------------------------------------------------------- tax integration
test('a modest MFJ year inside the 0% LTCG band is essentially tax-free', () => {
  assert.ok(runPlan(sim()).tax < 500.0);
});

test('the early penalty reaches the plan when the ladder hits traditional', () => {
  const plan = runPlan(sim(), { balances: [5000.0, 10000.0, 400000.0, 100000.0],
                                brokerageBasis: 8000.0, householdAge: 50.0 });
  assert.ok(plan.take_traditional > 0);
  assert.ok(plan.tax > 0.09 * plan.take_traditional);
});

test('zero spending need still pays tax on dividends and interest', () => {
  const plan = runPlan(sim(), { netNeed: 0.0, balances: [50000.0, 3000000.0, 0.0, 0.0],
                                brokerageBasis: 500000.0, householdAge: 70.0,
                                status: 'single' });
  assert.ok(plan.tax > 1000.0);
  approx(plan.gross_total - plan.tax, 0.0, { abs: 1.0 });
});

// ---------------------------------------------------------------- RMD mechanics
test('a forced RMD that covers the year triggers no extra withdrawal', () => {
  const rmd = 500000.0 / 23.7;                    // age 76 divisor
  const plan = runPlan(sim(), { netNeed: 10000.0, balances: [0.0, 0.0, 500000.0, 0.0],
                                brokerageBasis: 0.0, householdAge: 76.0,
                                status: 'single', ssIncome: 30000.0, rmdAmount: rmd });
  approx(plan.gross_total, rmd, { abs: 0.01 });
});

test('after-tax RMD money beyond the need is reinvested with full basis', () => {
  const rmd = 500000.0 / 23.7;
  const plan = runPlan(sim(), { netNeed: 10000.0, balances: [0.0, 0.0, 500000.0, 0.0],
                                brokerageBasis: 0.0, householdAge: 76.0,
                                status: 'single', ssIncome: 30000.0, rmdAmount: rmd });
  assert.ok(plan.surplus_reinvested > 0);
  approx(plan.balances[1], plan.surplus_reinvested, { abs: 0.01 });
  approx(plan.new_basis, plan.surplus_reinvested, { abs: 0.01 });
});

test('a fully hand-computed RMD year, single filer, to the cent', () => {
  // RMD       = 500,000 / 23.7           = 21,097.0464
  // deduction = 16,100 + 2,050 (age 65+) = 18,150
  // taxable   = 2,947.0464
  // federal 10% = 294.7046 ; state 5% = 147.3523 ; tax = 442.0570
  // net = 20,654.9895 ; surplus = 10,654.9895
  const rmd = 500000.0 / 23.7;
  const plan = runPlan(sim(), { netNeed: 10000.0, balances: [0.0, 0.0, 500000.0, 0.0],
                                brokerageBasis: 0.0, householdAge: 76.0,
                                status: 'single', ssIncome: 0.0, rmdAmount: rmd });
  approx(plan.tax, 442.0570, { abs: 0.01 });
  approx(plan.surplus_reinvested, 10654.9895, { abs: 0.01 });
  approx(plan.balances[2], 500000.0 - rmd, { abs: 0.01 });
});

test('no RMD divisor before the start age; the IRS table after', () => {
  const s = sim();
  assert.strictEqual(s.rmdDivisor(74), null);
  assert.strictEqual(s.rmdDivisor(75), 24.6);
  assert.strictEqual(s.rmdDivisor(90), 12.2);
  assert.strictEqual(s.rmdDivisor(130), 2.0);      // the 120+ floor
});

test('RMD_TABLE matches the IRS Uniform Lifetime Table', () => {
  // Pub. 590-B, post-2022. A transcription slip changes every forced withdrawal.
  const expected = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4,
    88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
    96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2,
    104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4,
    112: 3.3, 113: 3.1, 114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3,
    120: 2.0,
  };
  for (const [age, div] of Object.entries(expected))
    assert.strictEqual(RS.RMD_TABLE[age], div, `divisor at ${age}`);
  assert.strictEqual(Object.keys(RS.RMD_TABLE).length, Object.keys(expected).length);
});

// ---------------------------------------------------------------- basis handling
test("this year's reinvested distributions count as basis before the sale", () => {
  const s = sim();
  const plan = runPlan(s, { netNeed: 50000.0, balances: [0.0, 200000.0, 0.0, 0.0],
                            brokerageBasis: 200000.0, householdAge: 70.0,
                            status: 'single' });
  approx(plan.realized_gain, 0.0, { abs: 0.01 });
  assert.ok(200000.0 * 0.7 * s.cfg.market.stock_dividend_yield > 0);
});

test('an underwater account keeps its excess basis, pro-rata', () => {
  // $584,000 against $980,000 of basis. Selling $10,000 is 1.7123% of the
  // account, carrying away 1.7123% of the basis -> $963,219.18 left.
  const plan = runPlan(sim(), { netNeed: 10000.0, balances: [0.0, 584000.0, 0.0, 0.0],
                                brokerageBasis: 980000.0, householdAge: 70.0,
                                status: 'single', stockAlloc: 0.0, bondAlloc: 0.0 });
  approx(plan.take_brokerage, 10000.0, { abs: 0.01 });
  approx(plan.realized_gain, 0.0, { abs: 1e-9 });      // sold at a loss, not a gain
  approx(plan.new_basis, 963219.178, { abs: 0.01 });
  // the ratio survives up to the solver's sub-cent overshoot
  approx(plan.new_basis / plan.balances[1], 980000.0 / 584000.0, { rel: 1e-6 });
});

test('the recovery after a crash is not taxed as phantom gain', () => {
  const s = sim();
  const crashed = runPlan(s, { netNeed: 10000.0, balances: [0.0, 584000.0, 0.0, 0.0],
                               brokerageBasis: 980000.0, householdAge: 70.0,
                               status: 'single', stockAlloc: 0.0, bondAlloc: 0.0 });
  const balance = crashed.balances[1], basis = crashed.new_basis;
  const recovered = runPlan(s, { netNeed: 10000.0, balances: [0.0, balance * 2, 0.0, 0.0],
                                 brokerageBasis: basis, householdAge: 70.0,
                                 status: 'single', stockAlloc: 0.0, bondAlloc: 0.0 });
  const gainShare = 1 - basis / (balance * 2);
  approx(recovered.realized_gain, recovered.take_brokerage * gainShare, { rel: 1e-9 });
  assert.ok(recovered.realized_gain < 2000.0);   // ~4x that with a clamped basis
});

test('every sale conserves basis and gain', () => {
  const balances = [0.0, 200000.0, 0.0, 0.0];
  for (const bf of [0.0, 0.25, 0.5, 1.0]) {
    const t = RS.RetirementSimulator.ladderWithdraw(50000.0, balances, bf);
    approx(t.basis_returned + t.realized_gain, t.brokerage, { rel: 1e-12 });
  }
  // underwater: gain floors at zero while basis stays strictly pro-rata
  const t = RS.RetirementSimulator.ladderWithdraw(50000.0, balances, 1.5);
  assert.strictEqual(t.realized_gain, 0.0);
  approx(t.basis_returned, 75000.0, { abs: 1e-9 });
});

test('the ladder drains cash -> brokerage -> traditional -> Roth', () => {
  const L = RS.RetirementSimulator.ladderWithdraw;
  const balances = [10000.0, 20000.0, 30000.0, 40000.0];
  assert.strictEqual(L(5000.0, balances, 1.0).cash, 5000.0);
  const mid = L(45000.0, balances, 1.0);
  assert.deepStrictEqual([mid.cash, mid.brokerage, mid.traditional, mid.roth],
                         [10000.0, 20000.0, 15000.0, 0.0]);
  const all = L(200000.0, balances, 1.0);
  approx(all.unfunded, 100000.0, { abs: 1e-9 });
  approx(all.cash + all.brokerage + all.traditional + all.roth, 100000.0, { abs: 1e-9 });
});

test('the bisection converges tightly even on a huge portfolio', () => {
  const plan = runPlan(sim(), { netNeed: 250000.0,
                                balances: [0.0, 20000000.0, 20000000.0, 10000000.0],
                                brokerageBasis: 5000000.0, householdAge: 70.0 });
  const net = plan.gross_total - plan.tax;
  assert.ok(net >= 250000.0 - 1e-6);
  assert.ok(net - 250000.0 < 0.01);
});
