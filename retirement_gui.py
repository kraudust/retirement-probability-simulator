"""customtkinter desktop front end -- a thin view over the simulation engine.

The engine (retirement_age_calculator) owns the config schema and all modeling
logic. This file only maps widgets onto config fields, runs the simulation on a
worker thread, and displays the shared results table and charts.
"""

import copy
import os
import platform
import threading
from multiprocessing import freeze_support
from tkinter import filedialog

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from retirement_age_calculator import (
    RetirementSimulator,
    TRAJECTORY_SAMPLES,
    ValidationError,
    default_config,
    get_field,
    load_config,
    save_config,
    set_field,
    validate_config,
)

# Map from GUI widget keys to a field path on Config. This is the only mapping in the
# GUI: load, save, widget population and config building all resolve through it against
# the real dataclasses, so a bad path raises AttributeError instead of silently
# dropping the field the way two separate hand-maintained maps used to.
# (Deeper tax detail -- brackets, deductions, NIIT thresholds -- is per-filing-status
# tables and lives in the YAML only.)
FIELD_PATHS = {
    # Common - your details
    "current_age": "simulation.current_age",
    "min_retirement_age": "simulation.min_retirement_age",
    "max_retirement_age": "simulation.max_retirement_age",
    "target_success_probability": "simulation.target_success_probability",
    "monte_carlo_runs": "simulation.monte_carlo_runs",
    "random_seed": "simulation.random_seed",
    "rmd_start_age": "simulation.rmd_start_age",
    # Common - accounts
    "roth": "accounts.roth",
    "traditional": "accounts.traditional",
    "brokerage": "accounts.brokerage",
    "cash": "accounts.cash",
    "brokerage_basis": "accounts.brokerage_cost_basis",
    # Common - contributions
    "annual_roth": "contributions.annual_roth",
    "annual_traditional": "contributions.annual_traditional",
    "annual_brokerage": "contributions.annual_brokerage",
    "annual_cash": "contributions.annual_cash",
    "contrib_growth_rate": "contributions.annual_contribution_growth_rate",
    # Common - spending and healthcare
    "annual_expenses": "spending.initial_annual_expenses",
    "pre_medicare_premium": "healthcare.pre_medicare_annual_premium",
    "medicare_premium": "healthcare.medicare_annual_premium",
    "medicare_age": "healthcare.medicare_age",
    # Common - social security
    "ss_claim_age": "life_events.ss_claim_age",
    "ss_benefit": "life_events.ss_annual_full_retirement_benefit",
    "ss_years_worked": "life_events.ss_earnings_years_at_current_age",
    "ss_credits": "life_events.ss_credits_at_current_age",
    # Advanced - market
    "stock_return": "market.stock_return",
    "bond_return": "market.bond_return",
    "stock_volatility": "market.stock_volatility",
    "bond_volatility": "market.bond_volatility",
    "inflation": "market.inflation",
    "inflation_vol": "market.inflation_volatility",
    "cash_return": "market.cash_return",
    "stock_dividend_yield": "market.stock_dividend_yield",
    "bond_taxable_yield": "market.bond_taxable_yield",
    "stock_bond_corr": "market.stock_bond_correlation",
    # Advanced - glide path
    "glide_start": "simulation.glide_path_start_stock_pct",
    "glide_end": "simulation.glide_path_end_stock_pct",
    "glide_years": "simulation.glide_path_years",
    "static_stock": "simulation.static_stock_allocation",
    # Advanced - regimes
    "t_df": "simulation.return_distribution_degrees_of_freedom",
    "crisis_prob": "simulation.normal_regime.monthly_crisis_probability",
    "recovery_prob": "simulation.crisis_regime.monthly_recovery_probability",
    "crisis_drag": "simulation.crisis_regime.annual_return_drag",
    "crisis_vol": "simulation.crisis_regime.volatility_multiplier",
    "calm_boost": "simulation.normal_regime.return_boost",
    "calm_vol": "simulation.normal_regime.volatility_multiplier",
    # Advanced - spending adjustments
    "decline_start": "spending.spending_decline_start_age",
    "decline_rate": "spending.annual_spending_decline_rate",
    "decline_end": "spending.spending_decline_end_age",
    "healthcare_rate": "spending.annual_healthcare_increase_rate",
    # Advanced - guardrails
    "guard_cut_thresh": "spending.guardrail_cut_return_threshold",
    "guard_cut_amt": "spending.guardrail_cut_amount",
    "guard_cut_floor": "spending.guardrail_cut_floor",
    "guard_raise_thresh": "spending.guardrail_raise_return_threshold",
    "guard_raise_amt": "spending.guardrail_raise_amount",
    "guard_raise_ceil": "spending.guardrail_raise_ceiling",
    # Advanced - taxes
    "filing_status": "taxes.filing_status",
    "state_tax": "taxes.state_tax_rate",
    "early_penalty": "taxes.early_withdrawal_penalty",
    "penalty_age": "taxes.penalty_free_age",
    "niit_rate": "taxes.niit_rate",
    # Advanced - mortality
    "mortality_model": "life_events.mortality_model",
    "mortality_sex": "life_events.mortality_sex",
    "death_mean": "life_events.death_age_mean",
    "death_std": "life_events.death_age_std",
    "death_min": "life_events.death_age_min",
    "death_max": "life_events.death_age_max",
    # Advanced - spouse
    "spouse_age_offset": "spouse.age_offset",
    "spouse_ss_claim_age": "spouse.ss_claim_age",
    "spouse_ss_benefit": "spouse.ss_annual_full_retirement_benefit",
    "spouse_ss_years_worked": "spouse.ss_earnings_years_at_current_age",
    "spouse_ss_credits": "spouse.ss_credits_at_current_age",
    "survivor_spending": "spouse.survivor_spending_factor",
    "spouse_mortality_model": "spouse.mortality_model",
    "spouse_mortality_sex": "spouse.mortality_sex",
    "spouse_death_mean": "spouse.death_age_mean",
    "spouse_death_std": "spouse.death_age_std",
    "spouse_death_min": "spouse.death_age_min",
    "spouse_death_max": "spouse.death_age_max",
}

CHECKBOX_PATHS = {
    "glide_path": "simulation.glide_path",
    "compensate_drag": "simulation.compensate_crisis_drag",
    "common_random_numbers": "simulation.common_random_numbers",
    "use_72t": "taxes.use_72t_sepp",
    "age55_exception": "taxes.assume_qualified_plan_age55_exception",
    "spouse_enabled": "spouse.enabled",
}


class RetirementApp(ctk.CTk):
    """The application window. self.config holds the last loaded/saved Config;
    widgets are the working copy, combined and validated on Run/Save."""

    def __init__(self):
        """Build the window from the engine's default config and wire up scrolling."""
        super().__init__()
        self.title("Retirement Probability Simulator")
        self.geometry("1180x900")
        self.minsize(950, 720)
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.config = default_config()
        self.entries = {}
        self.checkboxes = {}
        self._scrollable_frames = []
        # Keeps the embedded matplotlib canvases alive; Tk holds only weak refs.
        self._chart_canvases = []
        self._build_ui()
        self._setup_scroll_binding()

    # ------------------------------------------------------------------
    # macOS trackpad scroll fix
    # ------------------------------------------------------------------
    def _setup_scroll_binding(self):
        """Bind scroll events globally and route them to the frame under the cursor.

        CTkScrollableFrame's own _mouse_wheel_all uses -event.delta on macOS, but
        the <TouchpadScroll> event packs X+Y into delta, so it reads garbage values
        and causes bounce-back. Disable it and handle scrolling ourselves.
        """
        for sf in self._scrollable_frames:
            sf._mouse_wheel_all = lambda event: None

        # macOS trackpads on newer Tk builds generate <TouchpadScroll>, not
        # <MouseWheel>; older Tk doesn't know the event name at all, hence the
        # guard. Mice still send <MouseWheel> everywhere.
        try:
            self.bind_all("<TouchpadScroll>", self._on_scroll)
        except Exception:
            pass
        self.bind_all("<MouseWheel>", self._on_scroll)
        if platform.system() != "Darwin":
            self.bind_all("<Button-4>", self._on_scroll)
            self.bind_all("<Button-5>", self._on_scroll)

    def _find_scrollable_parent(self, widget):
        """Walk up the widget tree to the CTkScrollableFrame containing `widget`."""
        w = widget
        while w is not None:
            if isinstance(w, ctk.CTkScrollableFrame):
                return w
            try:
                w = w.master
            except AttributeError:
                break
        return None

    def _on_scroll(self, event):
        """Route scroll events to the scrollable frame under the cursor."""
        try:
            target = event.widget.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return "break"

        if target is None:
            return "break"

        sf = self._find_scrollable_parent(target)
        if sf is None:
            return "break"

        try:
            canvas = sf._parent_canvas
        except AttributeError:
            return "break"

        # Decode the scroll delta
        if str(event.type) == "39":  # TouchpadScroll event type
            # macOS TouchpadScroll packs X and Y into delta:
            # Y in the lower 16 bits (signed), X in the upper 16 bits.
            raw_y = event.delta & 0xFFFF
            if raw_y >= 0x8000:
                raw_y -= 0x10000  # sign-extend 16-bit to Python int
            if raw_y != 0:
                # Clamp for smooth scrolling
                y_delta = max(-3, min(3, raw_y))
                canvas.yview_scroll(-y_delta, "units")
        elif hasattr(event, 'num') and event.num == 4:
            canvas.yview_scroll(-3, "units")
        elif hasattr(event, 'num') and event.num == 5:
            canvas.yview_scroll(3, "units")
        elif event.delta != 0:
            if platform.system() == "Darwin":
                canvas.yview_scroll(-event.delta, "units")
            else:
                canvas.yview_scroll(-event.delta // 120, "units")

        return "break"

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------
    def _add_section(self, parent, row, title):
        """Bold section header spanning the three grid columns."""
        ctk.CTkLabel(parent, text=title,
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=10, pady=(15, 5))

    def _add_entry(self, parent, row, label, key, desc, width=145):
        """One config row: label | text entry (pre-filled from config) | gray help
        text. `key` must exist in FIELD_PATHS; the entry registers in self.entries."""
        ctk.CTkLabel(parent, text=label, anchor="w", width=220).grid(
            row=row, column=0, sticky="w", padx=(10, 5), pady=2)
        entry = ctk.CTkEntry(parent, width=width)
        entry.insert(0, str(get_field(self.config, FIELD_PATHS[key])))
        entry.grid(row=row, column=1, padx=5, pady=2)
        ctk.CTkLabel(parent, text=desc, anchor="w", text_color="gray").grid(
            row=row, column=2, sticky="w", padx=(5, 10), pady=2)
        self.entries[key] = entry

    def _add_checkbox(self, parent, row, label, key, desc):
        """One boolean config row. `key` must exist in CHECKBOX_PATHS; the
        BooleanVar registers in self.checkboxes."""
        var = ctk.BooleanVar(value=bool(get_field(self.config, CHECKBOX_PATHS[key])))
        ctk.CTkCheckBox(parent, text=label, variable=var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=(10, 5), pady=2)
        ctk.CTkLabel(parent, text=desc, anchor="w", text_color="gray").grid(
            row=row, column=2, sticky="w", padx=(5, 10), pady=2)
        self.checkboxes[key] = var

    def _refresh_widgets(self):
        """Push self.config back into every widget (after loading a YAML)."""
        for key, path in FIELD_PATHS.items():
            entry = self.entries[key]
            entry.delete(0, "end")
            entry.insert(0, str(get_field(self.config, path)))
        for key, path in CHECKBOX_PATHS.items():
            self.checkboxes[key].set(bool(get_field(self.config, path)))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        """Top-level layout: load/save bar, settings tabs, run controls, and a
        results area (table + charts) that stays hidden until the first run."""
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(fill="x", padx=10, pady=(10, 0))
        ctk.CTkButton(file_frame, text="Load YAML", width=120,
                      command=self._load_yaml).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(file_frame, text="Save YAML", width=120,
                      command=self._save_yaml).pack(side="left", padx=5, pady=5)
        self.file_label = ctk.CTkLabel(file_frame, text="Using default values", text_color="gray")
        self.file_label.pack(side="left", padx=10, pady=5)

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(5, 0))
        self._build_common(self.tabs.add("Common Settings"))
        self._build_advanced(self.tabs.add("Advanced Settings"))

        ctrl = ctk.CTkFrame(self)
        ctrl.pack(fill="x", padx=10, pady=5)
        self.run_btn = ctk.CTkButton(ctrl, text="Run Simulation", width=170,
                                     font=ctk.CTkFont(size=14, weight="bold"),
                                     command=self._on_run)
        self.run_btn.pack(side="left", padx=10)
        self.status_label = ctk.CTkLabel(ctrl, text="Ready")
        self.status_label.pack(side="left", padx=10)
        self.progress = ctk.CTkProgressBar(ctrl)
        self.progress.pack(side="right", fill="x", expand=True, padx=10)
        self.progress.set(0)

        self.results_frame = ctk.CTkFrame(self)
        self.results_text = ctk.CTkTextbox(self.results_frame, height=230,
                                           font=ctk.CTkFont(family="Courier", size=12))
        self.results_text.pack(fill="x", padx=5, pady=5)
        self.chart_frame = ctk.CTkFrame(self.results_frame)
        self.chart_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_common(self, tab):
        """The everyday inputs: ages, balances, contributions, spending, Social
        Security. Everything a first-time user needs to produce an answer."""
        s = ctk.CTkScrollableFrame(tab)
        s.pack(fill="both", expand=True)
        s.grid_columnconfigure(2, weight=1)
        self._scrollable_frames.append(s)
        r = 0

        self._add_section(s, r, "Your Details"); r += 1
        for args in [
            ("Current Age", "current_age", "Your current age"),
            ("Min Retirement Age", "min_retirement_age", "Earliest retirement age to simulate"),
            ("Max Retirement Age", "max_retirement_age", "Latest retirement age to simulate"),
            ("Target Success Rate", "target_success_probability", "Success threshold (0.95 = 95%)"),
            ("Monte Carlo Runs", "monte_carlo_runs", "More = more accurate but slower (1000-10000)"),
            ("Random Seed", "random_seed", "Fixed seed = reproducible comparisons; -1 = fresh randomness"),
            ("RMD Start Age", "rmd_start_age", "73 if born 1951-1959, 75 if born 1960+ (SECURE 2.0)"),
        ]:
            self._add_entry(s, r, *args); r += 1

        self._add_section(s, r, "Account Balances (today's dollars)"); r += 1
        for args in [
            ("Roth", "roth", "Roth retirement accounts balance (Roth IRA, Roth 401k)"),
            ("Traditional", "traditional", "Traditional retirement accounts balance (IRA, 401k)"),
            ("Brokerage", "brokerage", "Taxable brokerage accounts balance"),
            ("Cash / Savings", "cash", "Cash, savings, money market balance"),
            ("Brokerage Cost Basis", "brokerage_basis",
             "What you PAID for the brokerage holdings; only gains above it are taxed. "
             "The IRS never indexes it, so inflation erodes it into taxable gain"),
        ]:
            self._add_entry(s, r, *args); r += 1

        self._add_section(s, r, "Annual Contributions"); r += 1
        for args in [
            ("Roth", "annual_roth", "Annual Roth contributions (Roth IRA, Roth 401k)"),
            ("Traditional", "annual_traditional", "Annual traditional contributions, including employer match"),
            ("Brokerage", "annual_brokerage", "Annual taxable brokerage contributions"),
            ("Cash", "annual_cash", "Annual cash savings"),
            ("Contribution Growth", "contrib_growth_rate", "REAL annual growth, above inflation (keep near 0)"),
        ]:
            self._add_entry(s, r, *args); r += 1

        self._add_section(s, r, "Spending & Healthcare (premiums ADD to expenses, per person)"); r += 1
        for args in [
            ("Annual Expenses", "annual_expenses", "Annual spending EXCLUDING health premiums"),
            ("Pre-Medicare Premium", "pre_medicare_premium",
             "Annual ACA/marketplace cost before Medicare. The big early-retirement cost"),
            ("Medicare Premium", "medicare_premium", "Annual Part B + D + supplement cost after Medicare age"),
            ("Medicare Age", "medicare_age", "Age Medicare begins and the pre-Medicare premium stops (65)"),
        ]:
            self._add_entry(s, r, *args); r += 1

        self._add_section(s, r, "Social Security"); r += 1
        for args in [
            ("Claim Age", "ss_claim_age", "62-70. Early claims are reduced, late claims credited 8%/yr"),
            ("FRA Benefit (full career)", "ss_benefit",
             "Benefit at full retirement age for a FULL 35-year career (ssa.gov/myaccount)"),
            ("Covered Years So Far", "ss_years_worked",
             "Years of SS-covered earnings you already have (35-year average)"),
            ("Credits So Far", "ss_credits",
             "SSA credits earned (4/year); 40 required for ANY retirement benefit"),
        ]:
            self._add_entry(s, r, *args); r += 1

    def _build_advanced(self, tab):
        """Model internals: market assumptions, allocation, regimes, taxes, the
        spending curve and guardrails, mortality, and the spouse block."""
        s = ctk.CTkScrollableFrame(tab)
        s.pack(fill="both", expand=True)
        s.grid_columnconfigure(2, weight=1)
        self._scrollable_frames.append(s)
        r = 0

        self._add_section(s, r, "Market Assumptions"); r += 1
        for args in [
            ("Stock Return (nominal)", "stock_return", "Expected annual return before inflation (historical ~10%)"),
            ("Bond Return (nominal)", "bond_return", "Expected annual bond return"),
            ("Stock Volatility", "stock_volatility", "Calm-market annual std dev; crises multiply it"),
            ("Bond Volatility", "bond_volatility", "Annual standard deviation of bond returns"),
            ("Inflation", "inflation", "Expected annual inflation (0.03 = 3%)"),
            ("Inflation Volatility", "inflation_vol",
             "Std dev of inflation. Hurts bonds/cash, which pay fixed nominal amounts"),
            ("Cash Return", "cash_return", "Expected nominal return on cash / savings accounts"),
            ("Stock Dividend Yield", "stock_dividend_yield",
             "Taxable-account dividend yield, taxed yearly at LTCG rates (~1.5% for an index fund)"),
            ("Bond Taxable Yield", "bond_taxable_yield",
             "Taxable-account bond interest, taxed yearly at ordinary rates"),
            ("Stock/Bond Correlation", "stock_bond_corr",
             "Range -1 to 1. -0.3 means bonds tend to rise when stocks fall (a hedge)"),
        ]:
            self._add_entry(s, r, *args); r += 1

        self._add_section(s, r, "Asset Allocation (Glide Path)"); r += 1
        self._add_checkbox(s, r, "Use Glide Path", "glide_path",
                           "Shift allocation linearly after retirement (either direction)"); r += 1
        for args in [
            ("Starting Stock %", "glide_start", "Stock allocation at retirement (0.90 = 90% stocks)"),
            ("Ending Stock %", "glide_end", "Final stock allocation (0.50 = 50% stocks)"),
            ("Glide Path Years", "glide_years", "Years to transition from starting to ending allocation"),
            ("Static Stock %", "static_stock", "Fixed stock allocation if glide path is off"),
        ]:
            self._add_entry(s, r, *args); r += 1

        self._add_section(s, r, "Market Crashes (Regime Switching)"); r += 1
        for args in [
            ("T-Distribution DF", "t_df", "Fat-tail severity: lower = more extreme crashes (6 = realistic)"),
            ("Monthly Crisis Probability", "crisis_prob", "Chance of entering a bear market each month (0.015 = 1.5%)"),
            ("Monthly Recovery Probability", "recovery_prob", "Chance of exiting each month (0.055 = ~18-month bears)"),
            ("Crisis Return Drag", "crisis_drag", "Annual return penalty during crisis (-0.12 = -12%)"),
            ("Crisis Volatility Multiplier", "crisis_vol", "How much wilder the market gets in a crisis (2.0 = twice)"),
            ("Calm Return Boost", "calm_boost",
             "Annual boost in calm markets (0.0 = none; drag compensation adjusts it further)"),
            ("Calm Volatility Multiplier", "calm_vol", "Calm-market volatility multiplier (usually 1.0)"),
        ]:
            self._add_entry(s, r, *args); r += 1
        self._add_checkbox(s, r, "Compensate Crisis Drag", "compensate_drag",
                           "Raise calm-market returns so the LONG-RUN average equals your stock return"); r += 1
        self._add_checkbox(s, r, "Common Random Numbers", "common_random_numbers",
                           "Test every retirement age against the SAME lifetimes (leave on for clean comparisons)"); r += 1

        self._add_section(s, r, "Spending Adjustments Over Time"); r += 1
        for args in [
            ("Decline Start Age", "decline_start", "Age when spending naturally starts declining"),
            ("Annual Decline Rate", "decline_rate", "Spending multiplier per year (0.99 = 1% annual decline)"),
            ("Decline End Age", "decline_end", "Age the decline stops and healthcare costs ramp up"),
            ("Healthcare Increase Rate", "healthcare_rate",
             "Annual growth of the health premiums after decline ends (medical inflation above CPI)"),
        ]:
            self._add_entry(s, r, *args); r += 1

        self._add_section(s, r, "Spending Guardrails"); r += 1
        for args in [
            ("Cut Return Threshold", "guard_cut_thresh",
             "Cut spending if the portfolio's return was worse than this in a year (-0.10 = -10%)"),
            ("Cut Amount", "guard_cut_amt", "Multiply spending by this when cutting (0.90 = 10% cut)"),
            ("Cut Floor", "guard_cut_floor", "Never cut spending below this fraction of plan"),
            ("Raise Return Threshold", "guard_raise_thresh",
             "Raise spending if the portfolio's return exceeded this in a year (0.15 = 15%)"),
            ("Raise Amount", "guard_raise_amt", "Multiply spending by this when raising (1.05 = 5% raise)"),
            ("Raise Ceiling", "guard_raise_ceil", "Never raise spending above this fraction of plan"),
        ]:
            self._add_entry(s, r, *args); r += 1

        self._add_section(s, r, "Taxes (brackets and deductions live in the YAML, per filing status)"); r += 1
        for args in [
            ("Filing Status", "filing_status",
             "married_filing_jointly / single / head_of_household. Survivor switches to single"),
            ("State Tax Rate", "state_tax", "Flat approximation. Use 0.0 for TX/FL/WA/NV/TN"),
            ("Early Withdrawal Penalty", "early_penalty",
             "IRS penalty on traditional withdrawals before the age below (0.10 = 10%)"),
            ("Penalty-Free Age", "penalty_age", "Age the early withdrawal penalty stops applying (59.5)"),
            ("NIIT Rate", "niit_rate", "Net Investment Income Tax (0.038 = 3.8% over the AGI threshold)"),
        ]:
            self._add_entry(s, r, *args); r += 1
        self._add_checkbox(s, r, "Assume 72(t) / Roth Ladder", "use_72t",
                           "Waives the 10% penalty. Requires real planning outside this model; off is conservative"); r += 1
        self._add_checkbox(s, r, "Rule-of-55 Exception", "age55_exception",
                           "Penalty-free employer-plan withdrawals from 55. Only for 401k money left in the plan"); r += 1

        self._add_section(s, r, "Life Expectancy"); r += 1
        for args in [
            ("Mortality Model", "mortality_model",
             "ssa_inspired (actuarial, realistic) or normal (plan to a chosen age)"),
            ("Mortality Sex", "mortality_sex", "male or female (sets the actuarial curve)"),
            ("Mean Death Age", "death_mean", "Normal model only: average lifespan for planning"),
            ("Std Deviation", "death_std", "Normal model only: uncertainty in lifespan"),
            ("Minimum Death Age", "death_min", "Normal model only: earliest possible death"),
            ("Maximum Death Age", "death_max", "Hard cap for BOTH models; also sizes the fan chart"),
        ]:
            self._add_entry(s, r, *args); r += 1

        self._add_section(s, r, "Spouse / Partner"); r += 1
        self._add_checkbox(s, r, "Model a Spouse", "spouse_enabled",
                           "Two lifespans, two SS benefits, survivor benefit and single filing on first death"); r += 1
        for args in [
            ("Spouse Age Offset", "spouse_age_offset", "Spouse age minus yours (+2 = two years older)"),
            ("Spouse SS Claim Age", "spouse_ss_claim_age", "Age they start Social Security (62-70)"),
            ("Spouse FRA Benefit", "spouse_ss_benefit",
             "Their full-career FRA benefit; 0 = stay-at-home, spousal benefit applies"),
            ("Spouse Years Worked", "spouse_ss_years_worked", "Their years of SS-covered earnings so far"),
            ("Spouse Credits", "spouse_ss_credits", "Their SSA credits so far (40 needed for an own benefit)"),
            ("Survivor Spending Factor", "survivor_spending",
             "Spending after the first death (0.75 = 75%; housing barely changes)"),
            ("Spouse Mortality Model", "spouse_mortality_model", "ssa_inspired or normal"),
            ("Spouse Mortality Sex", "spouse_mortality_sex", "male or female"),
            ("Spouse Mean Death Age", "spouse_death_mean", "Normal model only"),
            ("Spouse Death Std Dev", "spouse_death_std", "Normal model only"),
            ("Spouse Min Death Age", "spouse_death_min", "Normal model only"),
            ("Spouse Max Death Age", "spouse_death_max", "Hard cap for both models"),
        ]:
            self._add_entry(s, r, *args); r += 1

    # ------------------------------------------------------------------
    # Read values from GUI
    # ------------------------------------------------------------------
    def _build_config(self):
        """Widgets -> a validated Config. Raises with the offending field named."""
        config = copy.deepcopy(self.config)
        for key, path in FIELD_PATHS.items():
            raw = self.entries[key].get().strip()
            try:
                set_field(config, path, raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key}: {raw!r} is invalid ({exc})") from exc
        for key, path in CHECKBOX_PATHS.items():
            set_field(config, path, self.checkboxes[key].get())
        validate_config(config)
        return config

    # ------------------------------------------------------------------
    # Load / save YAML
    # ------------------------------------------------------------------
    def _load_yaml(self):
        """Load a parameter file (validated by the engine) and refresh every widget."""
        path = filedialog.askopenfilename(
            title="Load Simulation Parameters",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.config = load_config(path)
            self._refresh_widgets()
            self.file_label.configure(text=f"Loaded: {os.path.basename(path)}")
        except Exception as exc:
            self.file_label.configure(text=f"Could not load: {exc}")

    def _save_yaml(self):
        """Validate the current widget values and write them to a YAML file.
        Nothing is written if any value is invalid."""
        path = filedialog.asksaveasfilename(
            title="Save Simulation Parameters", defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")])
        if not path:
            return
        try:
            config = self._build_config()
            save_config(config, path)
            self.config = config
            self.file_label.configure(text=f"Saved: {os.path.basename(path)}")
        except Exception as exc:
            self.file_label.configure(text=f"Input error; nothing saved: {exc}")

    # ------------------------------------------------------------------
    # Run the simulation on a worker thread (UI touched only via .after)
    # ------------------------------------------------------------------
    def _on_run(self):
        """Validate inputs, then start the sweep on a daemon thread so the UI
        stays responsive. Input errors are shown in the status line immediately."""
        self.run_btn.configure(state="disabled")
        self.results_text.delete("1.0", "end")
        self.status_label.configure(text="Validating inputs...")
        self.progress.set(0)

        try:
            config = self._build_config()
        except (ValueError, KeyError, ValidationError) as e:
            self.status_label.configure(text=f"Input error: {e}")
            self.run_btn.configure(state="normal")
            return

        thread = threading.Thread(target=self._run_worker, args=(config,), daemon=True)
        thread.start()

    def _run_worker(self, config):
        """The background thread body: sweep every age, then compute the trajectory
        fan for the earliest passing age. All UI updates go through self.after(0,...)
        because Tk widgets may only be touched from the main thread."""
        try:
            sim = RetirementSimulator(config)

            def progress(i, total, age):
                """Engine callback -> status line + progress bar, via the main thread."""
                self.after(0, self._update_status, f"Simulating age {age} ({i + 1}/{total})...")
                self.after(0, self.progress.set, (i + 1) / total)

            sim.compute_probability_curve(progress=progress)

            trajectory_data = None
            result = sim.find_retirement_age()
            if result:
                self.after(0, self._update_status,
                           f"Computing portfolio trajectory for age {result.retirement_age}...")
                ages, percentiles = sim.compute_trajectory_percentiles(
                    result.retirement_age, n_samples=TRAJECTORY_SAMPLES)
                trajectory_data = (ages, percentiles, result.retirement_age)

            self.after(0, self.progress.set, 1.0)
            self.after(0, self._show_results, sim, config, trajectory_data)
        except Exception as exc:
            # Without this, an exception would silently kill the worker thread and
            # leave the Run button disabled forever.
            self.after(0, self._show_error, exc)

    def _update_status(self, text):
        """Set the status line (main-thread only; workers dispatch via .after)."""
        self.status_label.configure(text=text)

    def _show_error(self, exc):
        """Surface a worker-thread exception in the status line and re-arm Run."""
        self.status_label.configure(text=f"Simulation error: {exc}")
        self.run_btn.configure(state="normal")

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    def _show_results(self, sim, config, trajectory_data=None):
        """Render the assumption report + results table and embed the shared
        matplotlib charts. Runs on the main thread (dispatched via .after)."""
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", sim.assumption_report() + "\n\n")
        self.results_text.insert("end", sim.format_results_table())

        # Charts: ONE PER TAB, not stacked in a single figure. Stacking gave each
        # plot half of an already-short area, and at the window's minimum height
        # that left roughly 150px per plot -- not enough for a title and a rotated
        # y-axis label, so they were clipped. A tab gives every plot the full chart
        # area, and adding a third plot later costs one more entry in `charts`.
        for w in self.chart_frame.winfo_children():
            w.destroy()

        charts = [("Success by Age", sim.draw_probability_curve)]
        if trajectory_data:
            traj_ages, pcts, ret_age = trajectory_data
            charts.append(("Portfolio Trajectory",
                           lambda ax: sim.draw_trajectory(ax, traj_ages, pcts,
                                                          ret_age, TRAJECTORY_SAMPLES)))

        tabview = ctk.CTkTabview(self.chart_frame)
        tabview.pack(fill="both", expand=True)
        self._chart_canvases = []
        for name, draw in charts:
            # layout="constrained" re-solves on every draw, so labels stay inside
            # the axes as Tk stretches the canvas; tight_layout is computed once at
            # the figure's nominal size and does not survive a resize.
            fig = Figure(figsize=(9, 4.5), dpi=100, layout="constrained")
            draw(fig.add_subplot(1, 1, 1))
            canvas = FigureCanvasTkAgg(fig, master=tabview.add(name))
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._chart_canvases.append(canvas)

        # Show results section (shrink settings to make room)
        if not self.results_frame.winfo_ismapped():
            self.tabs.pack_configure(expand=False)
            self.results_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.status_label.configure(text="Done!")
        self.run_btn.configure(state="normal")


if __name__ == "__main__":
    freeze_support()
    app = RetirementApp()
    app.mainloop()
