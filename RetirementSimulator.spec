# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the desktop app.

Build locally with:   pyinstaller RetirementSimulator.spec
Output:               dist/Retirement Simulator.app   (macOS)
                      dist/RetirementSimulator/       (Windows, a folder to zip)

Three things here are load-bearing:

1. simulation_params.yaml is a DATA FILE. `default_config()` resolves it relative
   to the module's __file__, which inside a bundle is the extracted app directory,
   so the YAML has to be shipped alongside the code or the app cannot start.

2. scipy/pandas/PyQt and the other matplotlib backends are excluded. The app only
   ever draws through TkAgg, and the excludes keep the download to a sane size.

3. This is a ONE-FOLDER build, not one-file. One-file unpacks to a temp directory
   on every launch, which is slow for a bundle this size and interacts badly with
   multiprocessing on Windows -- the engine spawns a worker pool, and each worker
   re-imports the app.
"""

block_cipher = None

a = Analysis(
    ['retirement_gui.py'],
    pathex=[],
    binaries=[],
    # (source, destination-inside-bundle). '.' puts it next to the modules, which
    # is where DEFAULT_CONFIG_PATH looks for it.
    datas=[('simulation_params.yaml', '.')],
    hiddenimports=['customtkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'scipy', 'pandas', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx',
        'matplotlib.backends._backend_gtk', 'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_webagg',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RetirementSimulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # no terminal window behind the GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='RetirementSimulator',
)

# macOS only: wrap the folder into a double-clickable .app. Ignored on Windows.
app = BUNDLE(
    coll,
    name='Retirement Simulator.app',
    icon=None,
    bundle_identifier='com.kraudust.retirementsimulator',
    info_plist={
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',
        'CFBundleShortVersionString': '1.0.0',
    },
)
