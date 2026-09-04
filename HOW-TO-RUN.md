# Retirement Simulator — how to open it

Nothing is installed and nothing is sent anywhere. The app runs entirely on your
own computer, and the numbers you type never leave it.

Download the file for your computer, then follow the steps below **once**. After
that it opens by double-clicking like any other app.

| Your computer | Download |
|---|---|
| Mac | `RetirementSimulator-macOS.zip` |
| Windows | `RetirementSimulator-Windows.zip` |

---

## Mac

1. Double-click the downloaded `.zip`. You'll get **Retirement Simulator**.
2. Drag it into your **Applications** folder.
3. Double-click it. macOS will refuse to open it — **this is expected.** Follow
   the steps below once, and it will open normally forever after.

### "Apple could not verify this app is free of malware"

You will see this, and nothing is wrong.

Apple shows it for any app that hasn't been through their paid developer program
($99/year), which this one hasn't — it's a hobby project shared between friends,
not something sold in the App Store. The message means "Apple hasn't checked
this," not "this looks dangerous."

The dialog only offers **Move to Trash** or **Done**. Neither opens the app.
There is no hidden "Open" button, and right-clicking the app won't help either
on current versions of macOS. You have to grant the exception in Settings:

1. Click **Done** to dismiss the warning.
2. Open the **Apple menu → System Settings**, then click **Privacy & Security**
   in the sidebar. You may need to scroll down to find it.
3. Scroll to the **Security** section. You'll see a line naming the app that was
   blocked, with an **Open Anyway** button.
4. Click **Open Anyway**.
5. Enter your Mac login password and click **OK**.

**Do this within about an hour of trying to open the app** — the **Open Anyway**
button only appears for a while after the failed attempt. If it isn't there,
double-click the app again to trigger the warning, then go straight back to
Privacy & Security.

That's it. The app is now saved as a permanent exception, and from here on it
opens with a normal double-click like anything else.

---

## Windows

1. Right-click the downloaded `.zip` → **Extract All**.
2. Open the extracted folder and double-click **RetirementSimulator.exe**.
3. Optional: right-click it → **Pin to Start** so you can find it later.

### "Windows protected your PC"

Same story as the Mac warning. Click **More info**, then **Run anyway**.

Windows shows this for programs without a paid code-signing certificate. It's a
statement about paperwork, not about the program.

Keep the whole extracted folder together — the `.exe` needs the files next to it.
Moving the `.exe` out on its own will stop it working.

---

## Using it

1. Fill in the **Common Settings** tab: your age, what you've saved, what you
   put away each year, and what you spend.
2. Look through the **Advanced Settings** tab too (see below).
3. Click **Run Simulation**.
4. Wait a minute or two. The progress bar shows which retirement age it's working
   on — it tests each age in turn, so it advances in steps every couple of
   seconds rather than sliding smoothly.
5. Results appear below, with two tabs of charts:
   - **Success by Age** — your odds of not running out of money, for each
     possible retirement age.
   - **Portfolio Trajectory** — how your balance is likely to evolve.

### If you don't know what a box means, click the "?"

Every single input has a **?** button next to it. Click it and you'll get a
proper explanation: what the number means, how the simulation uses it, and how
to choose a value if you're unsure. There are no inputs you're expected to
already understand.

### Don't skip Advanced Settings

The defaults are sensible, and you can get a useful answer without touching that
tab. But several settings there change the answer a lot, and the defaults can't
know anything about you:

- **Taxes** — your **filing status** and **state tax rate**. The default assumes
  a married couple in a state with a 5% income tax. If you're single, or live
  somewhere with no state income tax (Texas, Florida, Washington, Nevada,
  Tennessee and others), fix this first. It's the setting most likely to be
  wrong for you.
- **Spouse** — off by default. If you're planning as a couple, turn on **Model
  Spouse** and fill in their details. This matters more than most people expect,
  because it captures what happens to the survivor: household spending drops,
  but so does Social Security, and the survivor starts filing taxes as a single
  person.
- **Lifespan** — the default assumes a man. Set **mortality sex** correctly;
  women live several years longer on average and those years have to be paid
  for.
- **Market assumptions** — expected returns, inflation and volatility. If you
  think 8% stock returns is optimistic, lower it and see how much the answer
  moves. That's one of the more valuable things you can do with this tool.
- **Investment mix** — how much you hold in stocks versus bonds, and whether
  that shifts as you age.

A good habit: run it once with the defaults, then change one thing and run it
again. The *difference* between two runs is far more trustworthy than any single
number, and it's what the app is really built for.

**Every dollar figure is in today's money.** A balance of $2,000,000 at age 70
means "$2,000,000 of what a dollar buys today," so you can compare it directly to
what you spend now. Enter your expected investment returns the normal way (before
subtracting inflation); the app takes inflation out for you.

You can save your inputs to a file and load them again later with the **Save**
and **Load** buttons, which is handy for comparing "what if I retire two years
earlier" against a baseline.

---

## Please read this part

This is a planning tool built by a friend, not financial advice, and not a
prediction. It runs thousands of randomised futures to show a *range* of
outcomes, and the range is only as good as the assumptions you type in. It does
not know about your specific tax situation, and it deliberately leaves out things
like long-term care costs.

Use it to compare choices against each other — "how much does one more working
year buy me?" — rather than to trust any single number.

---

## Something went wrong?

Tell me what you clicked and what it said, and send a screenshot if you can. If
the app won't start at all, that's worth knowing about — it means the build is
broken, not that you did anything wrong.
