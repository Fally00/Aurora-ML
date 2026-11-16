# Build on Windows with: pyinstaller --clean --noconfirm Aurora.spec
hiddenimports = [
    "pynvml",
    "wmi",
    "rich",
    "win32timezone",
    "win32evtlog",
    "win32event",
    "win32evtlogutil",
]

a = Analysis(
    ["aurora_cli.py"],
    pathex=[SPECPATH],
    binaries=[],
    datas=[("Data/watcher_config.json", "Data")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "sklearn"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Aurora",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
