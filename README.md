# Retirement Probability Simulator

Monte Carlo simulator that answers one question: **at what age can I retire and still
have my money outlast me?**

It sweeps every candidate retirement age, simulates thousands of randomised lifetimes
for each one, and reports the fraction in which the money lasted. It also draws the
distribution of portfolio values across retirement, so you can see not just *whether*
you survive but *with how much*.

## Run it

Open **`RetirementSimulator.html`** in any modern browser — double-click the file.
No install, no server, no build step. Nothing you type ever leaves your machine.

The sweep runs locally using Web Workers, one per CPU core.

Fill in the **Common Settings** tab, look through **Advanced Settings** (filing
status and state tax rate are the ones most likely to be wrong for you), then
click **Run Simulation**. Every input has a **?** button explaining what the
number means, how the simulation uses it, and how to pick a value.

You can save your inputs to a `simulation_params.yaml` file and load it again
later, which is the easy way to compare "what if I retire two years earlier"
against a baseline.

Read the **effective assumptions** block in the results. Several configured
values are transformed by the model's own mechanics — crises raise the effective
volatility above the calm-market number you entered, and the PIA formula
rescales the Social Security benefit — and that block shows what the simulation
is actually using.

## Tests

```bash
node --test "tests/*.test.js"     # ~7 seconds, no dependencies
```

Node's test runner is built in, so there is nothing to install. The suite loads
the engine straight out of `RetirementSimulator.html`, which means it always
tests the exact bytes that ship — it cannot pass against a stale copy.

154 tests pin the engine four ways:
- **Method-level known answers**: every tax quantity hand-computed from 2026 law
  (bracket math, LTCG stacking and the 0% band, the IRC 63(f) age-65 addition, SS
  provisional income, NIIT, penalties and their exceptions), SSA claim and
  survivor factors, PIA bend-point math, mortality against SSA cohort targets,
  and withdrawal-solver invariants (money conservation to the cent, exact need
  coverage, RMD mechanics).
- **Closed-form scenarios**: with volatility zeroed and lifespans fixed the whole
  engine reduces to arithmetic — final balances asserted to the cent, including
  the inflation-eroded cost basis and a full RMD year.
- **Statistical assertions** over large samples for the parts the deterministic
  fixtures switch OFF: Student-t scaling, stock/bond correlation, the regime
  chain's stationary start, and whether the assumption report's printed numbers
  are the ones actually delivered.
- **Benchmark bands** against the Trinity study, sized to at least three standard
  errors so they flag a real model change rather than sampling noise.

**A test that only checks self-consistency is not a test of correctness.** The
worst bugs found in this codebase lived in functions at 100% line coverage: the
tests executed the lines but never compared a value to an independently derived
truth. Derive expected numbers from the law or the formula by hand, never from
what the engine currently prints.

## Everything is in today's dollars

You enter **nominal** market returns and inflation is divided out once, up front.
Every balance, expense and contribution is **real** (inflation-adjusted). A portfolio
reading $2,000,000 at age 70 means $2,000,000 of *today's* purchasing power.

The consequence that trips people up: **any growth rate you set for a quantity already
in today's dollars is growth ABOVE inflation.** To keep contributions flat in
purchasing power, set `annual_contribution_growth_rate: 0.0`. Setting it to `0.03`
means a 3% raise every year *on top of* inflation — 1.8x your real contribution after
20 years.

There is exactly one deliberate exception: `brokerage_cost_basis`. Every *balance*
floats with the price level and so holds its real value, but a cost basis is a frozen
historical number that the IRS never indexes, so its real value decays with inflation
every month. Enter it in today's dollars like everything else; the decay starts today.

## Reproducible, comparable runs

The tool exists to compare decisions — "what does retiring with a paid-off house do?",
"what does one more working year buy?" — so the noise between two runs matters as much
as the accuracy of one.

- **`random_seed`**: any value ≥ 0 makes runs exactly repeatable; `-1` draws fresh
  entropy each time.
- **`common_random_numbers: true`**: every candidate retirement age is tested against
  the *same* set of random lifetimes (same deaths, same market paths). Differences
  between ages — or between two configs run with the same seed — then reflect the
  decision, not sampling luck. The success curve comes out monotone instead of
  wobbling around your target line.

## What the model includes

**Markets**
- Monthly returns with fat tails (Student-t), so large crashes occur at realistic rates
- Two-state regime switching between calm and crisis markets, calibrated to historical
  bear-market frequency (~every 5.6 years) and duration (~18 months)
- Optional `compensate_crisis_drag`, which raises calm-market returns so the long-run
  average equals the `stock_return` you configured. Without it, recurring crises
  silently subtract several points a year from your assumption
- Stock/bond correlation applied to standardised shocks, so realised bond volatility
  and correlation match the configured values exactly, in both regimes
- Stochastic inflation, which erodes the real return of bonds and cash (they pay a
  fixed nominal amount, so an inflation surprise comes straight off the top)

**Accounts and taxes**
- Four buckets: Roth, traditional, taxable brokerage, cash
- Withdrawal ladder cash → brokerage → traditional → Roth, cheapest tax first
- **Taxes are solved exactly, one tax year at a time**: a bisection finds the gross
  withdrawal whose after-tax proceeds cover the year's spending, with the bill
  computed from the actual withdrawal composition — no marginal-rate approximation.
  Bracket crossings, the Social Security "tax torpedo" and the 0% LTCG band land
  where the real 1040 puts them
- **Filing status is modelled, and it changes**: 2026 brackets/deductions per status,
  and when the first spouse dies the survivor files as *single* — half the deduction,
  compressed brackets (the widow's tax penalty)
- The **IRC §63(f) age-65 standard deduction addition**, claimed once per filer past
  65 and twice on a joint return where both are (2026: $2,050 unmarried, $1,650 per
  spouse). It applies in nearly every year a retirement model simulates — leaving it
  out overstates tax by roughly $350–600/yr
- Long-term capital gains taxed separately and *stacked* on ordinary income, so the
  0% LTCG band is captured — a retiree with modest ordinary income can realise
  substantial gains tax-free
- Brokerage **cost basis is tracked**: sales remove basis pro-rata, reinvested
  dividends and new contributions add it, and **inflation erodes it** — basis is the
  historical dollar figure on your 1099 and the IRS never indexes it, so inflation
  alone manufactures a taxable "gain". A $10k lot held 30 years at 8% nominal / 3%
  inflation is 90.1% taxable gain, not the 75.9% you would get by comparing today's
  dollars to today's dollars
- **Annual tax drag on the taxable account**: dividends (qualified rates) and bond
  and cash interest (ordinary rates) are taxed every year even when nothing is sold —
  the thing that makes a brokerage less efficient than an IRA
- The 3.8% **Net Investment Income Tax** (IRC 1411) above the AGI thresholds
- Social Security taxation via the IRS provisional-income rule (0% to 85% taxable)
- **10% early withdrawal penalty** on traditional accounts before 59.5, with a
  `use_72t_sepp` flag for a Rule 72(t)/Roth-ladder plan and an optional
  **rule-of-55** exception for employer-plan money
- Required Minimum Distributions from `rmd_start_age` (SECURE 2.0: 73 if born
  1951-1959, **75 if born 1960+**) using the IRS Uniform Lifetime Table on the
  prior year-end balance; an RMD bigger than the year's spending is reinvested in
  the brokerage, not left idle

**Social Security**
- Benefits rebuilt through the **progressive PIA bend-point formula** (2026 bend
  points): retiring early leaves zero-income years in the 35-year average, but the
  90/32/15 formula means a short career costs *less* than a linear proration —
  matching how SSA actually computes benefits
- **40-credit eligibility rule**: retire before ~10 working years and the model pays
  nothing, because SSA would pay nothing
- Exact claim-age factors: early-claim reductions (5/9% then 5/12% per month) and
  delayed credits at **8% per year simple** (factor 1.24 at 70, not 1.08³)
- Optional **spouse**: own-record benefit vs the 50% spousal benefit (with its own,
  steeper early-claim schedule and no delayed credits), survivor step-up to the
  larger benefit on the first death (with SSA's survivor-age reduction when the
  survivor's claim age is below 67, and the 82.5%-of-PIA widow(er)'s limit), and
  reduced household spending for the survivor

**Life and spending**
- Mortality per person, two models: `ssa_inspired` — a Gompertz hazard calibrated to
  SSA cohort life tables (male median death ~86 from 65, female ~89, realistic tails
  both directions) — or `normal`, a clipped normal for "plan to exactly age X"
  experiments
- **Healthcare costs** added on top of expenses, with the expensive pre-Medicare
  (ACA marketplace) years modelled separately. This is the biggest cost most early
  retirement plans forget
- The retirement "spending smile": real spending drifts down through your 70s; the
  late-life upturn comes through the healthcare premiums growing at
  `annual_healthcare_increase_rate` (medical inflation above CPI)
- Guardrails that cut or raise spending in response to the portfolio's own annual
  return — measured exactly, since withdrawals happen on anniversaries
- Glide path from stocks to bonds (either direction), anchored at your retirement date

## Reading the output

| Column | Meaning |
|---|---|
| Success | Fraction of simulated lifetimes where the money lasted |
| Median Min In Ret | **Median** lowest portfolio value *during retirement* (accumulation excluded) |
| Median Initial W/D | First-year **gross** withdrawal (spending shortfall + the taxes to fund it) ÷ portfolio. The "4% rule" benchmark |
| Median Final Bal | **Median** balance at death |

Every dollar column is a **median**, never a mean. Retirement outcomes are extremely
right-skewed — a few paths compound to enormous values — and the low-water mark is
worse still, because failed runs pile up at exactly $0. At a 47% success rate the
*mean* low point reads $871k while the median is $0; only the median describes the
outcome you should plan around.

The trajectory chart shows portfolio percentiles across retirement. Later ages rest
on fewer samples, because only the runs still alive at that age contribute — the
title notes this, and `compute_trajectory_percentiles` returns the backing count per
year as `percentiles["n"]`.

## Known limitations

Worth understanding before you act on a number:

- **Returns are independent month to month.** Real markets mean-revert somewhat at
  multi-year horizons, which thins the left tail. At the default settings the model's
  frequency of large one-year losses matches history (a real return below -20% in
  about 8% of years); what it lacks is the multi-year mean reversion, so bad decades
  are more likely than in the record. That is conservative, and it is why this will
  give a lower success rate than historical-sequence tools like cFIREsim.
- **The configured stock return is a geometric (compound) return.** The generator's
  median compounded growth matches `stock_return`; its arithmetic average is higher
  by about half the variance (roughly 10% nominal for an 8% input at the default
  volatility). Enter a CAGR, not an arithmetic average.
- **Sequence-of-returns risk is approximated**, not replayed from history. The regime
  model reproduces crash frequency, duration and volatility, but a specific historical
  path (1966, 2000) may still be worse than anything it generates.
- **Each year's spending is withdrawn up front** into a zero-return spending bucket
  and paid out monthly — a realistic cash-bucket strategy, but it costs about half a
  year of growth on annual spending versus withdrawing monthly (mildly conservative).
- **SS provisional-income and NIIT thresholds are held constant in real terms.** By
  statute they are fixed in *nominal* dollars (so in reality they shrink every year);
  holding them real assumes Congress eventually indexes them — optimistic late in a
  long retirement.
- **The healthcare figure is a flat real cost** (plus its own inflation rate late in
  life). No ACA subsidy cliff, no IRMAA surcharges, no long-term-care event.
- **All accounts hold the same stock/bond mix.** Asset location (bonds in the IRA,
  stocks in taxable) would beat the modelled outcome slightly — conservative.
- **Accumulation-phase dividends are assumed taxed from salary**, not from the
  account; the annual dividend/interest tax drag applies in retirement.
- **State tax is a flat rate** on federal AGI less the standard deduction and less
  Social Security (most states exempt it). No brackets, no other retirement-income
  exclusions.
- **Roth withdrawals are treated as tax-free at any age.** Before 59.5 only
  contributions come out free; earnings are taxed and penalised, and the model tracks
  no Roth basis. This matters only when the cash, brokerage and traditional rungs
  are exhausted before 59.5 -- rare with a normal account mix.
- **Pre-Medicare premiums are flat in real terms, and neither age-rating nor
  subsidies are modelled.** Two real effects are missing and they pull opposite
  ways. ACA premiums are legally age-rated — the same plan costs 1.44x the base
  rate at 45, 2.23x at 55 and 3.00x at 64 — so an early retiree's premium roughly
  doubles by 65 before any medical inflation, which makes a flat figure
  optimistic. Against that, the premium tax credit can cut the bill sharply at low
  taxable income, and the model never applies it, which makes a flat figure
  pessimistic. Which dominates depends on the income your own withdrawals create;
  note the subsidy cliff returned for 2026, so above 400% of the federal poverty
  level the credit is zero. Enter the AVERAGE net premium you expect across the
  years before 65 rather than today's quote.
- **Medical inflation above CPI applies only after `spending_decline_end_age`**,
  so the configured trend does not compound through the pre-Medicare years.
- Stocks are treated as inflation-neutral in the long run; inflation surprises are
  applied only to bonds and cash.
- The 72(t) flag waives the penalty but does not enforce a SEPP schedule; the
  withdrawal ladder is fixed, not optimised per-year (no Roth-conversion tax
  arbitrage in low-income years).

## Files

| File | Purpose |
|---|---|
| `RetirementSimulator.html` | The whole application: engine, UI and help, no dependencies |
| `simulation_params.yaml` | Example parameters, heavily commented |
| `tests/` | The test suite, plus a frozen `baseline_config.yaml` it runs against |

Inside the HTML, the engine lives in `<script id="engine-src">` and the UI in the
script block after it. The tests extract that first block and run it under Node,
so the engine stays independently testable without a browser.

The engine owns the config schema (`configFromDict` / `validateConfig` /
`getField` / `setField`), so adding a parameter means editing `SCHEMA`, the YAML,
and one `FIELD_PATHS` entry plus a widget. Per-filing-status tax tables
(brackets, deductions, NIIT thresholds) are YAML-only. `validateConfig` runs on
every load, save and run, and lists every problem it finds.

## Not financial advice

This is a model. Its output is only as good as its assumptions, and the sections above
should make clear how much the answer moves when those assumptions change.
