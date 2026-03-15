import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import yaml
import os
import platform
from tkinter import filedialog
from multiprocessing import freeze_support
from retirement_age_calculator import (
    Config, Accounts, Contributions, LifeEvents, Market,
    Simulation, NormalRegime, CrisisRegime, Spending,
    RetirementSimulator
)

# Map from GUI entry keys to YAML paths (section, key)
YAML_MAP_ENTRIES = {
    # Common - accounts
    "roth": ("accounts", "roth"),
    "traditional": ("accounts", "traditional"),
    "brokerage": ("accounts", "brokerage"),
    "cash": ("accounts", "cash"),
    # Common - contributions
    "annual_roth": ("contributions", "annual_roth"),
    "annual_traditional": ("contributions", "annual_traditional"),
    "annual_brokerage": ("contributions", "annual_brokerage"),
    "annual_cash": ("contributions", "annual_cash"),
    "contrib_growth_rate": ("contributions", "annual_contribution_growth_rate"),
    # Common - spending
    "annual_expenses": ("spending", "initial_annual_expenses"),
    # Common - social security
    "ss_claim_age": ("life_events", "ss_claim_age"),
    "ss_benefit": ("life_events", "ss_annual_full_retirement_benefit"),
    # Common - simulation
    "current_age": ("simulation", "current_age"),
    "min_retirement_age": ("simulation", "min_retirement_age"),
    "max_retirement_age": ("simulation", "max_retirement_age"),
    "target_success_probability": ("simulation", "target_success_probability"),
    "monte_carlo_runs": ("simulation", "monte_carlo_runs"),
    # Advanced - market
    "stock_return": ("market", "stock_return"),
    "bond_return": ("market", "bond_return"),
    "stock_volatility": ("market", "stock_volatility"),
    "bond_volatility": ("market", "bond_volatility"),
    "inflation": ("market", "inflation"),
    "cash_return": ("market", "cash_return"),
    "tax_rate": ("market", "tax_rate"),
    "cg_tax_rate": ("market", "capital_gains_tax_rate"),
    "stock_bond_corr": ("market", "stock_bond_correlation"),
    # Advanced - glide path
    "glide_start": ("simulation", "glide_path_start_stock_pct"),
    "glide_end": ("simulation", "glide_path_end_stock_pct"),
    "glide_years": ("simulation", "glide_path_years"),
    "static_stock": ("simulation", "static_stock_allocation"),
    # Advanced - regimes
    "t_df": ("simulation", "return_distribution_degrees_of_freedom"),
    "crisis_prob": ("simulation.normal_regime", "monthly_crisis_probability"),
    "recovery_prob": ("simulation.crisis_regime", "monthly_recovery_probability"),
    "crisis_drag": ("simulation.crisis_regime", "annual_return_drag"),
    "crisis_vol": ("simulation.crisis_regime", "volatility_multiplier"),
    # Advanced - spending adjustments
    "decline_start": ("spending", "spending_decline_start_age"),
    "decline_rate": ("spending", "annual_spending_decline_rate"),
    "decline_end": ("spending", "spending_decline_end_age"),
    "healthcare_rate": ("spending", "annual_healthcare_increase_rate"),
    # Advanced - guardrails
    "guard_cut_thresh": ("spending", "guardrail_cut_threshold"),
    "guard_cut_amt": ("spending", "guardrail_cut_amount"),
    "guard_cut_floor": ("spending", "guardrail_cut_floor"),
    "guard_raise_thresh": ("spending", "guardrail_raise_threshold"),
    "guard_raise_amt": ("spending", "guardrail_raise_amount"),
    "guard_raise_ceil": ("spending", "guardrail_raise_ceiling"),
    # Advanced - life expectancy
    "death_mean": ("life_events", "death_age_mean"),
    "death_std": ("life_events", "death_age_std"),
    "death_min": ("life_events", "death_age_min"),
    "death_max": ("life_events", "death_age_max"),
}

YAML_MAP_CHECKBOXES = {
    "glide_path": ("simulation", "glide_path"),
}


class RetirementApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Retirement Probability Simulator")
        self.geometry("1100x850")
        self.minsize(900, 700)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.entries = {}
        self.checkboxes = {}
        self._canvas_widget = None
        self._scrollable_frames = []

        self._build_ui()
        self._setup_scroll_binding()

    # ------------------------------------------------------------------
    # macOS trackpad scroll fix
    # ------------------------------------------------------------------
    def _setup_scroll_binding(self):
        """Bind mousewheel events globally and route to the correct scrollable frame."""
        # CTkScrollableFrame's _mouse_wheel_all uses -event.delta on macOS,
        # but <TouchpadScroll> packs X+Y into delta, so it gets garbage values
        # and causes bounce-back.  Disable it and handle scrolling ourselves.
        for sf in self._scrollable_frames:
            sf._mouse_wheel_all = lambda event: None

        # macOS trackpad generates <TouchpadScroll>, not <MouseWheel>
        self.bind_all("<TouchpadScroll>", self._on_scroll)
        self.bind_all("<MouseWheel>", self._on_scroll)
        if platform.system() != "Darwin":
            self.bind_all("<Button-4>", self._on_scroll)
            self.bind_all("<Button-5>", self._on_scroll)

    def _find_scrollable_parent(self, widget):
        """Walk up the widget tree to find a CTkScrollableFrame's inner canvas."""
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
        except (AttributeError, Exception):
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
            # macOS TouchpadScroll packs X and Y into delta.
            # Y is in the lower 16 bits (signed), X in upper 16 bits.
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
    # UI helpers
    # ------------------------------------------------------------------
    def _add_section(self, parent, row, title):
        ctk.CTkLabel(
            parent, text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=10, pady=(15, 5))

    def _add_entry(self, parent, row, label, key, default, desc, width=130):
        ctk.CTkLabel(parent, text=label, anchor="w", width=200).grid(
            row=row, column=0, sticky="w", padx=(10, 5), pady=2)
        entry = ctk.CTkEntry(parent, width=width)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, padx=5, pady=2)
        ctk.CTkLabel(parent, text=desc, anchor="w", text_color="gray").grid(
            row=row, column=2, sticky="w", padx=(5, 10), pady=2)
        self.entries[key] = entry

    def _add_checkbox(self, parent, row, label, key, default, desc):
        var = ctk.BooleanVar(value=default)
        ctk.CTkCheckBox(parent, text=label, variable=var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=(10, 5), pady=2)
        ctk.CTkLabel(parent, text=desc, anchor="w", text_color="gray").grid(
            row=row, column=2, sticky="w", padx=(5, 10), pady=2)
        self.checkboxes[key] = var

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Simple pack layout: settings fill the window, results appear below when ready
        # File controls
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(fill="x", padx=10, pady=(10, 0))

        ctk.CTkButton(
            file_frame, text="Load YAML", width=120, command=self._load_yaml,
        ).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(
            file_frame, text="Save YAML", width=120, command=self._save_yaml,
        ).pack(side="left", padx=5, pady=5)
        self.file_label = ctk.CTkLabel(file_frame, text="Using default values", text_color="gray")
        self.file_label.pack(side="left", padx=10, pady=5)

        # Parameter tabs — takes all available space
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(5, 0))
        self._build_common(self.tabs.add("Common Settings"))
        self._build_advanced(self.tabs.add("Advanced Settings"))

        # Controls bar
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(fill="x", padx=10, pady=5)

        self.run_btn = ctk.CTkButton(
            ctrl, text="Run Simulation", width=160,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_run,
        )
        self.run_btn.pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(ctrl, text="Ready")
        self.status_label.pack(side="left", padx=10)

        self.progress = ctk.CTkProgressBar(ctrl)
        self.progress.pack(side="right", fill="x", expand=True, padx=10)
        self.progress.set(0)

        # Results — hidden until simulation completes
        self.results_frame = ctk.CTkFrame(self)
        # Don't pack yet — will be shown after first simulation

        self.results_text = ctk.CTkTextbox(self.results_frame, height=180, font=ctk.CTkFont(family="Courier", size=12))
        self.results_text.pack(fill="x", padx=5, pady=5)

        self.chart_frame = ctk.CTkFrame(self.results_frame)
        self.chart_frame.pack(fill="both", expand=True, padx=5, pady=5)

    # ------------------------------------------------------------------
    # YAML load / save
    # ------------------------------------------------------------------
    def _load_yaml(self):
        path = filedialog.askopenfilename(
            title="Load Simulation Parameters",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return

        with open(path) as f:
            raw = yaml.safe_load(f)

        for gui_key, (section, yaml_key) in YAML_MAP_ENTRIES.items():
            try:
                if "." in section:
                    parts = section.split(".")
                    value = raw[parts[0]][parts[1]][yaml_key]
                else:
                    value = raw[section][yaml_key]
                entry = self.entries[gui_key]
                entry.delete(0, "end")
                entry.insert(0, str(value))
            except (KeyError, TypeError):
                pass

        for gui_key, (section, yaml_key) in YAML_MAP_CHECKBOXES.items():
            try:
                value = raw[section][yaml_key]
                self.checkboxes[gui_key].set(bool(value))
            except (KeyError, TypeError):
                pass

        self.file_label.configure(text=f"Loaded: {os.path.basename(path)}")

    def _save_yaml(self):
        path = filedialog.asksaveasfilename(
            title="Save Simulation Parameters",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return

        raw = {
            "accounts": {},
            "contributions": {},
            "life_events": {},
            "market": {},
            "simulation": {
                "normal_regime": {},
                "crisis_regime": {},
            },
            "spending": {},
        }

        for gui_key, (section, yaml_key) in YAML_MAP_ENTRIES.items():
            try:
                val_str = self.entries[gui_key].get()
                # Try int first, then float, then string
                try:
                    value = int(val_str)
                except ValueError:
                    try:
                        value = float(val_str)
                    except ValueError:
                        value = val_str

                if "." in section:
                    parts = section.split(".")
                    raw[parts[0]][parts[1]][yaml_key] = value
                else:
                    raw[section][yaml_key] = value
            except (KeyError, TypeError):
                pass

        for gui_key, (section, yaml_key) in YAML_MAP_CHECKBOXES.items():
            try:
                raw[section][yaml_key] = self.checkboxes[gui_key].get()
            except (KeyError, TypeError):
                pass

        with open(path, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

        self.file_label.configure(text=f"Saved: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # Common tab
    # ------------------------------------------------------------------
    def _build_common(self, tab):
        s = ctk.CTkScrollableFrame(tab)
        s.pack(fill="both", expand=True)
        s.grid_columnconfigure(2, weight=1)
        self._scrollable_frames.append(s)
        r = 0

        self._add_section(s, r, "Your Details"); r += 1
        self._add_entry(s, r, "Current Age", "current_age", 35,
                        "Your current age"); r += 1
        self._add_entry(s, r, "Min Retirement Age", "min_retirement_age", 40,
                        "Earliest retirement age to simulate"); r += 1
        self._add_entry(s, r, "Max Retirement Age", "max_retirement_age", 65,
                        "Latest retirement age to simulate"); r += 1
        self._add_entry(s, r, "Target Success Rate", "target_success_probability", 0.95,
                        "Success threshold (0.95 = 95%)"); r += 1

        self._add_section(s, r, "Account Balances (today's dollars)"); r += 1
        self._add_entry(s, r, "Roth", "roth", 100000.00,
                        "Roth retirement accounts balance (Roth IRA, Roth 401k)"); r += 1
        self._add_entry(s, r, "Traditional", "traditional", 200000.00,
                        "Traditional retirement accounts balance (IRA, 401k)"); r += 1
        self._add_entry(s, r, "Brokerage", "brokerage", 10000.00,
                        "Taxable brokerage accounts balance"); r += 1
        self._add_entry(s, r, "Cash / Savings", "cash", 75000.00,
                        "Cash, savings, money market balance"); r += 1

        self._add_section(s, r, "Annual Contributions"); r += 1
        self._add_entry(s, r, "Roth", "annual_roth", 15000,
                        "Annual roth retirement accounts contributions (Roth IRA, Roth 401k)"); r += 1
        self._add_entry(s, r, "Traditional", "annual_traditional", 33750,
                        "Annual traditional retirement accounts contributions (including employer match) (IRA, 401k)"); r += 1
        self._add_entry(s, r, "Brokerage", "annual_brokerage", 10000,
                        "Annual taxable brokerage contributions"); r += 1
        self._add_entry(s, r, "Cash", "annual_cash", 0,
                        "Annual cash savings"); r += 1
        self._add_entry(s, r, "Contribution Growth", "contrib_growth_rate", 0.03,
                        "Annual increase in contributions (0.03 = 3%)"); r += 1

        self._add_section(s, r, "Spending"); r += 1
        self._add_entry(s, r, "Annual Expenses", "annual_expenses", 70000,
                        "Current annual spending in today's dollars"); r += 1

        self._add_section(s, r, "Social Security"); r += 1
        self._add_entry(s, r, "Claim Age", "ss_claim_age", 67,
                        "Age you plan to start Social Security"); r += 1
        self._add_entry(s, r, "Annual Benefit", "ss_benefit", 40000,
                        "SS annual benefit at full retirement age (67)"); r += 1

        self._add_section(s, r, "Simulation Quality"); r += 1
        self._add_entry(s, r, "Monte Carlo Runs", "monte_carlo_runs", 5000,
                        "More = more accurate but slower (1000-10000 recommended)"); r += 1

    # ------------------------------------------------------------------
    # Advanced tab
    # ------------------------------------------------------------------
    def _build_advanced(self, tab):
        s = ctk.CTkScrollableFrame(tab)
        s.pack(fill="both", expand=True)
        s.grid_columnconfigure(2, weight=1)
        self._scrollable_frames.append(s)
        r = 0

        self._add_section(s, r, "Market Assumptions"); r += 1
        self._add_entry(s, r, "Stock Return (nominal)", "stock_return", 0.08,
                        "Expected annual return before inflation (historical ~10%)"); r += 1
        self._add_entry(s, r, "Bond Return (nominal)", "bond_return", 0.04,
                        "Expected annual bond return"); r += 1
        self._add_entry(s, r, "Stock Volatility", "stock_volatility", 0.15,
                        "Annual standard deviation of stock returns"); r += 1
        self._add_entry(s, r, "Bond Volatility", "bond_volatility", 0.05,
                        "Annual standard deviation of bond returns"); r += 1
        self._add_entry(s, r, "Inflation", "inflation", 0.03,
                        "Expected annual inflation (0.03 = 3%)"); r += 1
        self._add_entry(s, r, "Cash Return", "cash_return", 0.04,
                        "Expected return on cash / savings accounts"); r += 1
        self._add_entry(s, r, "Income Tax Rate", "tax_rate", 0.22,
                        "Tax rate on traditional 401k/IRA withdrawals"); r += 1
        self._add_entry(s, r, "Capital Gains Tax", "cg_tax_rate", 0.15,
                        "Long-term capital gains tax on brokerage gains"); r += 1
        self._add_entry(s, r, "Stock/Bond Correlation", "stock_bond_corr", -0.3,
                        "Range -1.0 to 1.0. -0.3 means a 10% stock drop yields ~3% bond gain (a hedge)"); r += 1

        self._add_section(s, r, "Asset Allocation (Glide Path)"); r += 1
        self._add_checkbox(s, r, "Use Glide Path", "glide_path", True,
                           "Gradually shift from stocks to bonds after retirement"); r += 1
        self._add_entry(s, r, "Starting Stock %", "glide_start", 0.90,
                        "Stock allocation at retirement (0.90 = 90% stocks)"); r += 1
        self._add_entry(s, r, "Ending Stock %", "glide_end", 0.50,
                        "Final stock allocation (0.50 = 50% stocks)"); r += 1
        self._add_entry(s, r, "Glide Path Years", "glide_years", 31,
                        "Years to transition from starting to ending allocation"); r += 1
        self._add_entry(s, r, "Static Stock %", "static_stock", 0.75,
                        "Fixed stock allocation if glide path is off"); r += 1

        self._add_section(s, r, "Market Crashes (Regime Switching)"); r += 1
        self._add_entry(s, r, "T-Distribution DF", "t_df", 6,
                        "Fat-tail severity: lower = more extreme crashes (6 = realistic)"); r += 1
        self._add_entry(s, r, "Monthly Crisis Probability", "crisis_prob", 0.015,
                        "Chance of entering a bear market each month (0.015 = 1.5%)"); r += 1
        self._add_entry(s, r, "Monthly Recovery Probability", "recovery_prob", 0.15,
                        "Chance of exiting a bear market each month (0.15 = 15%)"); r += 1
        self._add_entry(s, r, "Crisis Return Drag", "crisis_drag", -0.12,
                        "Annual return penalty during crisis (-0.12 = -12%)"); r += 1
        self._add_entry(s, r, "Crisis Volatility Multiplier", "crisis_vol", 2.0,
                        "How much wilder the market gets in a crisis (2.0 = twice)"); r += 1

        self._add_section(s, r, "Spending Adjustments Over Time"); r += 1
        self._add_entry(s, r, "Decline Start Age", "decline_start", 70,
                        "Age when spending naturally starts declining"); r += 1
        self._add_entry(s, r, "Annual Decline Rate", "decline_rate", 0.99,
                        "Spending multiplier per year (0.99 = 1% annual decline)"); r += 1
        self._add_entry(s, r, "Decline End Age", "decline_end", 80,
                        "Age when decline stops and healthcare costs ramp up"); r += 1
        self._add_entry(s, r, "Healthcare Increase Rate", "healthcare_rate", 0.015,
                        "Annual healthcare cost increase after decline ends (0.015 = 1.5%)"); r += 1

        self._add_section(s, r, "Spending Guardrails"); r += 1
        self._add_entry(s, r, "Cut Return Threshold", "guard_cut_thresh", -0.10,
                        "Cut spending if the market's real return was worse than this in a year (-0.10 = -10%)"); r += 1
        self._add_entry(s, r, "Cut Amount", "guard_cut_amt", 0.90,
                        "Multiply spending by this when cutting (0.90 = 10% cut)"); r += 1
        self._add_entry(s, r, "Cut Floor", "guard_cut_floor", 0.70,
                        "Never cut spending below this fraction of initial"); r += 1
        self._add_entry(s, r, "Raise Return Threshold", "guard_raise_thresh", 0.15,
                        "Raise spending if the market's real return exceeded this in a year (0.15 = 15%)"); r += 1
        self._add_entry(s, r, "Raise Amount", "guard_raise_amt", 1.05,
                        "Multiply spending by this when raising (1.05 = 5% raise)"); r += 1
        self._add_entry(s, r, "Raise Ceiling", "guard_raise_ceil", 1.30,
                        "Never raise spending above this fraction of initial"); r += 1

        self._add_section(s, r, "Life Expectancy"); r += 1
        self._add_entry(s, r, "Mean Death Age", "death_mean", 93,
                        "Average lifespan for planning (higher = more conservative)"); r += 1
        self._add_entry(s, r, "Std Deviation", "death_std", 8,
                        "Uncertainty in lifespan (higher = wider range)"); r += 1
        self._add_entry(s, r, "Minimum Death Age", "death_min", 70,
                        "Earliest possible death in simulation"); r += 1
        self._add_entry(s, r, "Maximum Death Age", "death_max", 105,
                        "Latest possible death in simulation"); r += 1

    # ------------------------------------------------------------------
    # Read values from GUI
    # ------------------------------------------------------------------
    def _get_float(self, key):
        return float(self.entries[key].get())

    def _get_int(self, key):
        return int(float(self.entries[key].get()))

    def _get_bool(self, key):
        return self.checkboxes[key].get()

    def _build_config(self):
        return Config(
            accounts=Accounts(
                roth=self._get_float("roth"),
                traditional=self._get_float("traditional"),
                brokerage=self._get_float("brokerage"),
                cash=self._get_float("cash"),
            ),
            contributions=Contributions(
                annual_roth=self._get_float("annual_roth"),
                annual_traditional=self._get_float("annual_traditional"),
                annual_brokerage=self._get_float("annual_brokerage"),
                annual_cash=self._get_float("annual_cash"),
                annual_contribution_growth_rate=self._get_float("contrib_growth_rate"),
            ),
            life_events=LifeEvents(
                death_age_mean=self._get_int("death_mean"),
                death_age_std=self._get_float("death_std"),
                death_age_min=self._get_int("death_min"),
                death_age_max=self._get_int("death_max"),
                ss_claim_age=self._get_int("ss_claim_age"),
                ss_annual_full_retirement_benefit=self._get_float("ss_benefit"),
            ),
            market=Market(
                stock_return=self._get_float("stock_return"),
                bond_return=self._get_float("bond_return"),
                stock_volatility=self._get_float("stock_volatility"),
                bond_volatility=self._get_float("bond_volatility"),
                inflation=self._get_float("inflation"),
                cash_return=self._get_float("cash_return"),
                tax_rate=self._get_float("tax_rate"),
                capital_gains_tax_rate=self._get_float("cg_tax_rate"),
                stock_bond_correlation=self._get_float("stock_bond_corr"),
            ),
            simulation=Simulation(
                current_age=self._get_int("current_age"),
                min_retirement_age=self._get_int("min_retirement_age"),
                max_retirement_age=self._get_int("max_retirement_age"),
                target_success_probability=self._get_float("target_success_probability"),
                monte_carlo_runs=self._get_int("monte_carlo_runs"),
                glide_path=self._get_bool("glide_path"),
                glide_path_start_stock_pct=self._get_float("glide_start"),
                glide_path_end_stock_pct=self._get_float("glide_end"),
                glide_path_years=self._get_int("glide_years"),
                return_distribution_degrees_of_freedom=self._get_int("t_df"),
                static_stock_allocation=self._get_float("static_stock"),
                normal_regime=NormalRegime(
                    return_boost=0.0,
                    volatility_multiplier=1.0,
                    monthly_crisis_probability=self._get_float("crisis_prob"),
                ),
                crisis_regime=CrisisRegime(
                    annual_return_drag=self._get_float("crisis_drag"),
                    volatility_multiplier=self._get_float("crisis_vol"),
                    monthly_recovery_probability=self._get_float("recovery_prob"),
                ),
            ),
            spending=Spending(
                initial_annual_expenses=self._get_float("annual_expenses"),
                spending_decline_start_age=self._get_int("decline_start"),
                annual_spending_decline_rate=self._get_float("decline_rate"),
                spending_decline_end_age=self._get_int("decline_end"),
                annual_healthcare_increase_rate=self._get_float("healthcare_rate"),
                guardrail_cut_return_threshold=self._get_float("guard_cut_thresh"),
                guardrail_cut_amount=self._get_float("guard_cut_amt"),
                guardrail_cut_floor=self._get_float("guard_cut_floor"),
                guardrail_raise_return_threshold=self._get_float("guard_raise_thresh"),
                guardrail_raise_amount=self._get_float("guard_raise_amt"),
                guardrail_raise_ceiling=self._get_float("guard_raise_ceil"),
            ),
        )

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    def _on_run(self):
        self.run_btn.configure(state="disabled")
        self.results_text.delete("1.0", "end")
        self.status_label.configure(text="Validating inputs...")
        self.progress.set(0)

        try:
            config = self._build_config()
        except (ValueError, KeyError) as e:
            self.status_label.configure(text=f"Input error: {e}")
            self.run_btn.configure(state="normal")
            return

        thread = threading.Thread(target=self._run_worker, args=(config,), daemon=True)
        thread.start()

    def _run_worker(self, config):
        sim = RetirementSimulator(config)
        min_age = config.simulation.min_retirement_age
        max_age = config.simulation.max_retirement_age
        total = max_age - min_age + 1

        for i, age in enumerate(range(min_age, max_age + 1)):
            self.after(0, self._update_status, f"Simulating age {age} ({i + 1}/{total})...")
            self.after(0, self.progress.set, i / total)
            sim.probability_results[age] = sim.retirement_probability(age)

        self.after(0, self.progress.set, 1.0)
        self.after(0, self._show_results, sim, config)

    def _update_status(self, text):
        self.status_label.configure(text=text)

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    def _show_results(self, sim, config):
        target = config.simulation.target_success_probability
        result = sim.find_retirement_age()

        self.results_text.delete("1.0", "end")
        if result:
            self.results_text.insert(
                "end",
                f"  Earliest retirement age meeting {target:.0%} target: AGE {result.retirement_age}"
                f"  ({result.success_probability:.1%} success)\n\n",
            )
        else:
            self.results_text.insert(
                "end",
                f"  No retirement age met the {target:.0%} success target in the range tested.\n\n",
            )

        self.results_text.insert("end", "  Age   Success   Avg Minimum     Avg Withdrawal Rate   Avg Final Balance\n")
        self.results_text.insert("end", "  " + "-" * 80 + "\n")
        for age in sorted(sim.probability_results):
            prob, avg_min, avg_wr, avg_final = sim.probability_results[age]
            self.results_text.insert(
                "end",
                f"  {age:>4}   {prob:>6.1%}   ${avg_min:>13,.0f}     {avg_wr:>6.2%}                 ${avg_final:>13,.0f}\n",
            )

        # Chart
        for w in self.chart_frame.winfo_children():
            w.destroy()

        fig = Figure(figsize=(9, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        ages = sorted(sim.probability_results)
        probs = [sim.probability_results[a][0] for a in ages]

        ax.plot(ages, probs, "b-o", markersize=4, linewidth=2)
        ax.axhline(target, color="r", linestyle="--", alpha=0.7,
                    label=f"Target ({target:.0%})")
        if result:
            ax.axvline(result.retirement_age, color="green", linestyle=":",
                       alpha=0.7, label=f"Earliest: age {result.retirement_age}")
        ax.set_xlabel("Retirement Age")
        ax.set_ylabel("Success Probability")
        ax.set_title("Retirement Success Probability by Age")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_widget = canvas

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
