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
3. **The first time only:** right-click (or Control-click) the app and choose
   **Open**, then click **Open** again in the box that appears.

### "Apple could not verify this app is free of malware"

You will almost certainly see this the first time. Nothing is wrong.

Apple shows this for any app that hasn't been through their paid developer
program ($99/year), which this one hasn't — it's a hobby project shared between
friends, not something sold in the App Store. The warning means "Apple hasn't
checked this," not "this is dangerous."

The right-click → **Open** in step 3 is what gets past it. If you double-click
normally the first time, macOS will refuse and you'll have to right-click → Open
anyway.

If macOS is stubborn and won't offer **Open** at all, go to
**System Settings → Privacy & Security**, scroll down, and click
**Open Anyway** next to the app's name.

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
2. Click **Run Simulation**.
3. Wait. The progress bar shows which retirement age it's working on — it tests
   each age one at a time, so it moves in steps rather than smoothly.
4. Results appear below, with two tabs of charts:
   - **Success by Age** — your odds of not running out of money, for each
     possible retirement age.
   - **Portfolio Trajectory** — how your balance is likely to evolve.

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
