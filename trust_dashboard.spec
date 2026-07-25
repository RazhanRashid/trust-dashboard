# -*- mode: python ; coding: utf-8 -*-
# Build with: pyinstaller trust_dashboard.spec
# Must be run on the target OS — PyInstaller does not cross-compile, so a
# Windows .exe has to be built on a Windows machine (or CI runner).
from PyInstaller.utils.hooks import collect_all

datas = [('Physio_analysis/face_landmarker.task', 'Physio_analysis')]
binaries = []
hiddenimports = []

# mediapipe/opensmile/cv2 ship native binaries + data files that PyInstaller's
# static import scan doesn't see; pyqtgraph needs its icon/resource data too.
for pkg in ('mediapipe', 'opensmile', 'pyqtgraph', 'cv2'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TrustDashboard',
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

# onedir (COLLECT), not onefile — onefile re-extracts mediapipe's large native
# libs into a temp dir on every launch, which is slow and trips some AV
# scanners. Ship the TrustDashboard/ folder produced under dist/ instead.
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TrustDashboard',
)
