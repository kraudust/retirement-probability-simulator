"""Launcher for the desktop app. Run this, not retirement_gui.py.

    python3 app_main.py

Why this file exists, and why it must stay this small:

The engine runs its Monte Carlo sweep across a pool of worker processes. On macOS
and Windows those workers are created by SPAWN, which means each one starts a
fresh interpreter and RE-IMPORTS the main module to rebuild its state.

If the main module is retirement_gui.py, every worker therefore imports
customtkinter, matplotlib and tkinter -- an entire GUI toolkit -- purely to run
arithmetic it will never draw. With ten workers and one pool per retirement age,
a single sweep paid that cost hundreds of times. Measured in a packaged build:
23.1 seconds per age with the GUI imports, 2.8 seconds without. An 8x penalty,
and it is invisible from the source tree because imports are far cheaper from
disk than from inside an app bundle.

Keeping the heavy imports INSIDE the __main__ guard fixes it. A spawned worker
imports this module under the name "__mp_main__", so the guard is False and the
worker gets an essentially empty module -- it never touches the GUI stack.

The rule to preserve: nothing expensive at module scope in whatever file is the
process entry point. If you add an import here, put it inside the guard.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Give the process real stdio before anything else runs.
#
# A windowed build has no console attached. On Windows, PyInstaller therefore
# leaves sys.stdout, sys.stderr AND sys.stdin as None -- and multiprocessing's
# child-process bootstrap writes to them while starting a worker. The workers
# die during startup, the parent waits forever for results that never arrive,
# and the window goes "Not Responding". That is why Run Simulation hung on
# Windows while macOS was fine: macOS keeps real streams in a windowed app, so
# nothing there ever exercised this path.
#
# This must sit at MODULE scope, not inside the __main__ guard. Spawned workers
# re-import this file as __mp_main__, so the guard is False for them -- but they
# are exactly the processes that need the streams to exist.
# ---------------------------------------------------------------------------
for _stream, _mode in (("stdin", "r"), ("stdout", "w"), ("stderr", "w")):
    if getattr(sys, _stream, None) is None:
        setattr(sys, _stream, open(os.devnull, _mode))

from multiprocessing import freeze_support  # noqa: E402

if __name__ == "__main__":
    # Must come first: it is what lets a spawned worker re-enter this file and
    # take the worker path instead of starting a second application.
    freeze_support()

    from retirement_gui import RetirementApp

    RetirementApp().mainloop()
