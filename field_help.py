"""Long-form help for every input in the GUI, shown by the "?" button.

The gray text beside each box is a one-line reminder. This is the fuller answer:
what the number means, how the simulation uses it, and how to pick a value if you
are not sure. Keys match FIELD_PATHS / CHECKBOX_PATHS in retirement_gui.py.

Two rules apply to every dollar amount in the app and are worth stating once:

  * Everything is in TODAY'S dollars. Enter what things cost now; the model
    handles inflation for you and reports results in today's money too, so a
    balance of $2,000,000 at age 70 means "$2,000,000 of what a dollar buys
    today".
  * Investment returns are the exception -- enter those the normal way, before
    subtracting inflation (the "nominal" number you see quoted everywhere).
"""

FIELD_HELP = {
    # ---------------------------------------------------------------- you
    "current_age": """Your age today.

Everything starts here: the years between this age and your retirement age are
the years you keep saving, and the simulation begins on your next birthday.""",

    "min_retirement_age": """The earliest retirement age to test.

The app does not pick one retirement age -- it tests every age from this one up
to the maximum below, and shows you the odds for each. Set this to the earliest
age you would seriously consider, even if it looks optimistic. Seeing it fail is
useful information.""",

    "max_retirement_age": """The latest retirement age to test.

Testing more ages takes longer -- roughly two seconds per age -- so there is no
point going past the age where you are clearly safe.""",

    "target_success_probability": """How sure you want to be, as a decimal.

0.95 means "I want a 95% chance the money lasts". The headline answer is the
earliest age that clears this bar.

There is no universally correct value. 0.95 is the convention in retirement
research. Below about 0.80 you are taking a real risk of running out; above
0.99 you will likely die with a very large unspent balance, which is its own
kind of mistake.""",

    # ---------------------------------------------------------------- accounts
    "roth": """Current balance of all your Roth accounts (Roth IRA, Roth 401k).

Roth money has already been taxed, so withdrawals are completely tax-free. The
model spends this LAST, on purpose -- leaving it untouched the longest lets it
grow tax-free for as long as possible.""",

    "traditional": """Current balance of all pre-tax retirement accounts:
traditional 401k, traditional IRA, 403b, and similar.

You have never paid tax on this money, so every dollar withdrawn counts as
ordinary income. Two consequences the model handles for you: withdrawals before
age 59.5 usually carry an extra 10% penalty, and from age 75 the IRS forces
minimum withdrawals whether you want the money or not.""",

    "brokerage": """Current balance of ordinary taxable investment accounts --
the kind with no retirement rules attached.

When you sell, you owe tax only on the profit, usually at the lower capital
gains rate. You are also taxed each year on dividends and interest even if you
sell nothing, which the model accounts for.""",

    "cash": """Savings, checking and money market balances.

Treated as a separate safe bucket: it earns the cash return rather than the
stock/bond mix, and it is spent FIRST because a dollar of cash is a dollar you
can spend without triggering any tax.

Note that cash sitting outside your investment mix quietly lowers your real
stock exposure. If cash is 20% of your money, a "90% stocks" setting is really
about 72% stocks overall.""",

    "brokerage_basis": """What you originally PAID for your taxable brokerage
holdings -- not what they are worth now.

Only the growth above this figure is taxed when you sell. If your brokerage
account is worth $25,000 and you paid $20,000 for it, then $5,000 is taxable
profit and the other $20,000 is just your own money coming back.

If you do not know it, look for "cost basis" on your brokerage statement. A
rough guess is fine: 60-80% of the balance for an account you funded recently,
less for one you have held for decades.

One thing that surprises people: the IRS never adjusts this figure for
inflation, so as prices rise, more of your balance counts as "profit" even if
you gained nothing in real terms. The model reproduces that.""",

    # ---------------------------------------------------------------- saving
    "annual_roth": """How much you put into Roth accounts each year, in today's
dollars.

Include your own contributions only. Enter what you save in a typical year, not
your best year.""",

    "annual_traditional": """How much goes into pre-tax accounts each year --
your own 401k/IRA contributions PLUS any employer match.

The match is real money that compounds for you, so leaving it out understates
your savings meaningfully.""",

    "annual_brokerage": """How much you add to ordinary taxable investment
accounts each year, in today's dollars.""",

    "annual_cash": """How much you add to plain savings each year, beyond your
investment accounts. Leave at 0 if you are not deliberately building cash.""",

    "contrib_growth_rate": """How fast your saving grows each year ABOVE
inflation, as a decimal.

Leave this at 0. That does NOT mean your contributions stay flat -- it means
they keep pace with inflation, so you would actually be writing bigger checks
every year (about 3% bigger) while the sacrifice stays the same.

0.03 would mean a 3% raise EVERY year on top of inflation, for your whole
career. That compounds to nearly double your real saving after 20 years, which
is far more optimistic than it sounds. Long-run real wage growth is under 1%.""",

    # ---------------------------------------------------------------- spending
    "annual_expenses": """What you expect to spend per year in retirement, in
today's dollars, EXCLUDING health insurance premiums.

Leave health premiums out -- they are entered separately below, because they
change sharply at 65 when Medicare starts.

This is the single most important number in the whole app. A useful starting
point is what you spend now, minus what you currently save, minus costs that
stop at retirement (commuting, a mortgage that will be paid off).""",

    "decline_start": """The age at which spending starts to drift down.

Real retirees tend to spend less as they age -- less travel, fewer big
purchases. Researchers call the whole pattern the "retirement spending smile":
down through your 70s, then up again late in life through healthcare.""",

    "decline_rate": """How fast spending falls each year during the decline
window, as a multiplier.

0.99 means you spend 1% less each year than the year before. 1.0 turns the
decline off entirely, which is the conservative choice.""",

    "decline_end": """The age at which the spending decline stops.

After this, base spending stays flat at its reduced level and it is healthcare
costs that start rising instead.""",

    "healthcare_rate": """How fast health premiums grow each year after the
decline ends, above general inflation.

Medical costs have historically outrun everything else. 0.015 (1.5% a year
above inflation) is a reasonable, moderate assumption.""",

    # ---------------------------------------------------------------- healthcare
    "pre_medicare_premium": """Yearly health insurance cost PER PERSON before
Medicare starts, in today's dollars.

This is the number that makes early retirement expensive. Before 65 you buy
your own coverage on the open market with no employer paying most of it.
$12,000 a year is typical for one unsubsidised adult including out-of-pocket
costs -- though if your taxable income in retirement is low, subsidies can cut
this a lot.""",

    "medicare_premium": """Yearly health cost PER PERSON once Medicare starts,
in today's dollars.

Medicare is not free. Roughly $2,500 a year covers Part B, Part D and a
supplement policy.""",

    "medicare_age": """The age Medicare begins. This is 65 under current law;
there is no reason to change it unless you are testing a what-if.""",

    # ---------------------------------------------------------------- social security
    "ss_claim_age": """The age you start taking Social Security.

Claiming early permanently shrinks the monthly check; waiting permanently grows
it. At 62 you get 70% of your full benefit; at 67 (full retirement age) you get
100%; at 70 you get 124%.

Waiting is often the better deal if you expect to live into your 80s, and it
doubles as insurance against living a very long time. But claiming early can
make sense if you retire well before 67 and would otherwise have to sell
investments in a downturn to bridge the gap.""",

    "ss_benefit": """Your estimated yearly Social Security at full retirement
age (67), assuming a full 35-year career.

Get the real number from ssa.gov/myaccount -- it takes a few minutes and is far
better than a guess. Enter the ANNUAL figure (the site shows monthly, so
multiply by 12).

Enter the full-career number even if you plan to retire early. The app works
out the reduction for a shorter career itself, using the real Social Security
formula.""",

    "ss_years_worked": """How many years you have already worked and paid
Social Security taxes.

Social Security averages your best 35 years, and any year you did not work
counts as a zero in that average. So retiring early costs you twice: fewer
years of earnings, and zeros dragging down the average.""",

    "ss_credits": """Social Security work credits you have already earned. Your
ssa.gov statement shows this.

You earn up to 4 per year, and you need 40 -- about ten years of work -- to
qualify for ANY retirement benefit at all. If you would retire before reaching
40, the model correctly pays you nothing.""",

    # ---------------------------------------------------------------- market
    "stock_return": """Expected yearly stock return BEFORE subtracting
inflation.

Enter the normal, quoted kind of number. US stocks have returned about 10% a
year historically; most forecasters expect less going forward, so 7-8% is a
reasonable planning figure and lower is more cautious.

The app subtracts inflation for you. At 8% with 3% inflation it uses about
4.85% of real growth.

This is a long-run average, not a promise. Individual years vary enormously,
which is the entire point of running thousands of simulations.""",

    "bond_return": """Expected yearly bond return before inflation. Roughly
today's yield on intermediate-term bonds is a fair estimate.""",

    "cash_return": """Expected yearly interest on savings and money market
balances, before inflation.

Cash usually barely keeps up with inflation after tax, which is why holding a
lot of it for decades has a real cost.""",

    "stock_volatility": """How much stock returns bounce around, as a decimal.
This is the CALM-market figure.

0.15 is right for a broad stock index. The app adds market crises on top, which
pushes the effective all-in figure to about 19% -- matching the real historical
range for US stocks.

Bigger number means a wider spread of outcomes, both good and bad.""",

    "bond_volatility": """How much bond returns bounce around. Around 0.05 for
intermediate-term bonds -- much steadier than stocks, which is the point of
owning them.""",

    "inflation": """Expected long-run yearly inflation.

Long-run US inflation has been about 3%; the Federal Reserve targets 2%.

This is what converts your entered returns into real growth, so it matters a
lot. A higher number here makes every investment return worth less.""",

    "inflation_vol": """How uncertain inflation is, as a yearly standard
deviation.

This matters because bonds and cash promise a FIXED number of dollars. If
inflation comes in higher than expected, those dollars buy less and the loss
comes straight off your real return. Stocks are treated as roughly
inflation-proof over the long run. 0.015 matches post-war US experience.""",

    "stock_dividend_yield": """The share of stock returns paid out as dividends
each year, rather than showing up as price growth.

This does not change your total return -- it changes your TAX. Dividends are
taxed every year even if you never sell. About 1.5% matches a broad US index
fund. Only affects taxable brokerage money.""",

    "bond_taxable_yield": """The share of bond returns paid out as interest each
year.

Same idea as dividends, but interest is taxed at higher ordinary rates. Set it
roughly equal to your bond return, since nearly all of a bond's return is
interest.""",

    "stock_bond_corr": """How stocks and bonds move together, from -1 to +1.

Negative means bonds tend to rise when stocks fall, which is the diversification
people hope for. But be careful: this was reliably negative from 2000-2021 and
strongly POSITIVE in 2022 when both fell hard together. Near zero is the honest
long-run assumption; anything below -0.2 assumes a dependable hedge that may not
show up when you need it.""",

    # ---------------------------------------------------------------- allocation
    "glide_path": """When ticked, your stock percentage changes gradually
through retirement instead of staying fixed.

The traditional advice is to get more conservative with age. Some researchers
argue the opposite -- starting conservative at retirement and rising -- protects
better against a crash in your first few years. Both are valid; the app lets you
test either.""",

    "glide_start": """The share in stocks on the day you retire, as a decimal
(0.90 = 90% stocks).

Remember this applies to your invested accounts only. Cash sits outside the mix,
so a large cash balance means your true stock exposure is lower than this.""",

    "glide_end": """The share in stocks once the glide is complete.

Setting this HIGHER than the start is allowed and is a real strategy -- see the
glide path help above.""",

    "glide_years": """How many years the shift from the starting mix to the
ending mix takes.""",

    "static_stock": """The fixed share in stocks, used only when the glide path
is switched off.""",

    # ---------------------------------------------------------------- crises
    "t_df": """Controls how extreme the worst months can get.

Real markets crash harder and more often than a simple bell curve predicts. A
LOWER number here means fatter tails -- more severe crashes. 6 reproduces the
historical frequency of big one-year losses. Above 30 essentially removes fat
tails, which is unrealistically calm.""",

    "crisis_prob": """The chance each month of tipping into a market crisis.

0.015 works out to roughly one crisis every 5.6 years, matching how often real
bear markets arrive. Set to 0 to switch crises off entirely.""",

    "recovery_prob": """The chance each month of coming OUT of a crisis. This
sets how LONG crises last.

0.055 gives an average of about 18 months, matching the historical average bear
market. Higher values produce brief dips and would understate the drawn-out
downturns that actually break retirements.""",

    "crisis_drag": """How much worse yearly returns are during a crisis.

Rough guide: -0.08 is a correction, -0.12 a typical bear market, -0.20 a
2008-style crash.""",

    "crisis_vol": """How much more violently markets swing during a crisis. 2.0
means twice the normal volatility.""",

    "calm_boost": """An optional bonus to returns during calm periods, on top of
your stock return. Normally 0.""",

    "calm_vol": """Volatility multiplier during calm periods. Normally 1.0,
meaning your stock volatility applies as entered.""",

    "compensate_drag": """When ticked, calm-market returns are raised just
enough that the LONG-RUN average equals the stock return you entered.

Leave this on unless you know you want otherwise. With it off, recurring crises
quietly drag your average return well below what you asked for, and your entered
8% might actually deliver 2%.""",

    # ---------------------------------------------------------------- guardrails
    "guard_cut_thresh": """If your portfolio returns worse than this over the
past year, you cut back spending next year.

This is what real retirees do -- nobody keeps spending identically through a
crash. Modelling that flexibility is what makes higher withdrawal rates
survivable.""",

    "guard_cut_amt": """How much you cut spending after a bad year. 0.90 means
spending 10% less.""",

    "guard_cut_floor": """The most you would ever cut, as a share of plan. 0.70
means you will never go below 70% of planned spending, no matter how many bad
years arrive.

Set this to the level below which your budget genuinely cannot flex.""",

    "guard_raise_thresh": """If your portfolio returns better than this over the
past year, you allow yourself to spend a bit more.""",

    "guard_raise_amt": """How much you increase spending after a good year. 1.05
means 5% more.""",

    "guard_raise_ceil": """The most you would ever raise spending to, as a share
of plan. 1.30 means never more than 130% of the original plan.""",

    # ---------------------------------------------------------------- taxes
    "filing_status": """Your tax filing status: single, married_filing_jointly,
head_of_household, or qualifying_surviving_spouse.

Important: this is separate from the "model a spouse" tick box. Turning spouse
modelling off does NOT make you a single filer. This box controls the tax
brackets; that box controls whether a second person's lifespan and benefit are
simulated.""",

    "state_tax": """Your state income tax rate as a decimal. Use 0.0 for states
with no income tax (Texas, Florida, Washington, Nevada, Tennessee and others).

This is a flat approximation -- the app does not model state brackets or state
retirement-income exclusions, which many states offer. If your state has them,
your real tax will be lower than modelled.""",

    "early_penalty": """The extra tax on pre-tax retirement withdrawals taken
too early. 0.10 (10%) is current law.""",

    "penalty_age": """The age the early withdrawal penalty stops applying. 59.5
under current law.""",

    "niit_rate": """A 3.8% surtax on investment income for high earners. Rarely
applies at modest retirement spending, but it is real law and cheap to model
exactly.""",

    "use_72t": """Tick if you plan to use a Rule 72(t) withdrawal schedule or a
Roth conversion ladder -- both legally avoid the 10% early penalty.

Both take genuine planning and commitment to set up, so leaving this unticked is
the cautious choice. Withdrawals are still taxed as income either way.""",

    "age55_exception": """Tick only if the money in your pre-tax bucket is
genuinely in an employer 401k/403b that you will leave in the plan.

The "rule of 55" lets you take from an employer plan penalty-free if you leave
that job in or after the year you turn 55. IRAs never qualify -- if you have
rolled everything into an IRA, leave this unticked.""",

    "rmd_start_age": """The age the IRS forces you to start withdrawing from
pre-tax accounts, whether you need the money or not.

Under current law: 73 if you were born 1951-1959, 75 if born 1960 or later.""",

    # ---------------------------------------------------------------- lifespan
    "mortality_model": """How the app decides how long you live.

"ssa_inspired" draws a realistic range of lifespans from actuarial data -- some
runs end in your 60s, a few past 100. This is the honest choice, because living
a long time is itself a financial risk.

"normal" lets you dictate the range by hand with the four boxes below, useful
for a "what if I plan to exactly 95" test.""",

    "mortality_sex": """male or female, used to pick the right mortality table.
Women live meaningfully longer on average, so this changes the answer.""",

    "death_mean": """Average age at death. Used ONLY when the model is set to
"normal" -- ignored entirely by "ssa_inspired".""",

    "death_std": """How much lifespans vary around that average, in years. Used
only by the "normal" model.""",

    "death_min": """The youngest age at death to consider. Used only by the
"normal" model.""",

    "death_max": """The oldest possible age. This applies to BOTH models, and it
also sets how far the portfolio chart runs.""",

    # ---------------------------------------------------------------- spouse
    "spouse_enabled": """Tick to model a second person.

This changes several things at once: household spending drops after the first
death, the survivor inherits the larger of the two Social Security benefits, and
the survivor starts filing taxes as single -- half the standard deduction and
tighter brackets. That last effect is a real and often-missed hit to a widowed
household's income.""",

    "spouse_age_offset": """Your spouse's age relative to yours. -2 means they
are two years younger; +3 means three years older.""",

    "spouse_ss_claim_age": """The age your spouse starts Social Security. Same
trade-off as your own claim age.""",

    "spouse_ss_benefit": """Your spouse's own yearly benefit at full retirement
age, from their ssa.gov account.

Set to 0 for a partner with little or no earnings record. They will still
collect a SPOUSAL benefit worth up to half of yours -- the app applies that
automatically, so 0 here does not mean they get nothing.""",

    "spouse_ss_years_worked": """Years your spouse has already paid into Social
Security. 0 is fine for a partner who has not worked.""",

    "spouse_ss_credits": """Your spouse's Social Security credits (0-40). Only
affects their own benefit, not the spousal benefit.""",

    "survivor_spending": """What the surviving partner spends, as a share of
what the couple spent together.

Not 0.5. Rent, utilities, insurance and property taxes barely change when one
person dies. 0.75 is a realistic figure.""",

    "spouse_mortality_model": """How your spouse's lifespan is drawn. Same
choices as your own.""",

    "spouse_mortality_sex": """male or female, used to pick your spouse's
mortality table.

Women live meaningfully longer on average, so for a mixed-sex couple this
usually means the wife outlives the husband by several years -- and those extra
years are spent filing taxes as a single person on a reduced Social Security
benefit. Getting this right matters more than it looks.""",

    "spouse_death_mean": """Your spouse's average age at death. Used only by the
"normal" model.""",

    "spouse_death_std": """How much your spouse's lifespan varies. Used only by
the "normal" model.""",

    "spouse_death_min": """The youngest age at death to consider for your
spouse.

Used only when your spouse's mortality model is set to "normal" -- the
"ssa_inspired" model ignores it and draws from actuarial data instead.""",

    "spouse_death_max": """Oldest possible age for your spouse. Applies to both
models.""",

    # ---------------------------------------------------------------- run
    "monte_carlo_runs": """How many separate futures to simulate for each
retirement age.

More runs give a steadier answer but take longer. 1,000 is fine while you are
experimenting; 5,000 is good for a decision you will act on. Below about 500 the
number wobbles enough between runs to be misleading.""",

    "random_seed": """Controls the random numbers.

Any number 0 or above makes runs exactly repeatable -- run it twice, get the
same answer. Use -1 for fresh randomness each time.

Keep a fixed seed when comparing two choices, so the difference you see is the
choice and not luck.""",

    "common_random_numbers": """When ticked, every retirement age is tested
against the SAME set of simulated futures.

Leave this on. It means the difference between retiring at 57 and 58 reflects
that one extra year, not random luck, and it makes the results curve smooth
instead of jagged exactly where you read it.""",
}
