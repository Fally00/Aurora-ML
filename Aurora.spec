# Build on Windows with: pyinstaller --clean --noconfirm Aurora.spec
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("pyqtgraph") + [
    "pynvml",
    "wmi",
    "win32timezone",
    "win32evtlog",
    "win32event",
    "win32evtlogutil",
]

a = Analysis(
    ["Dashboard/tray_app.py"],
    pathex=[SPECPATH],
    binaries=[],
    datas=[("Data/watcher_config.json", "Data")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "sklearn", "streamlit", "plotly"],
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
