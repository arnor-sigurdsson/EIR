# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# https://stackoverflow.com/questions/38977929/pyinstaller-creating-exe-runtimeerror-maximum-recursion-depth-exceeded-while-ca
import sys

sys.setrecursionlimit(5000)

options = [("W ignore", None, "OPTION")]

block_cipher = None

datas = []
for to_copy in (
    "tqdm",
    "regex",
    "sacremoses",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "tokenizers",
):
    datas += copy_metadata(to_copy)

datas += collect_data_files("timm", include_py_files=True)

a = Analysis(
    ["../eir/build_module.py"],
    binaries=[],
    datas=datas,
    pathex=[],
    hiddenimports=[
        "sklearn.utils._cython_blas",
        "pkg_resources.py2_warn",
        "sklearn.neighbors._typedefs",
        "sklearn.neighbors._quad_tree",
        "sklearn.utils._weight_vector",
        "sklearn.utils._typedefs",
        "sklearn.neighbors._partition_nodes",
        "sklearn.neighbors",
        "sklearn.tree._utils",
        "sklearn.tree",
        "ray.async_compat",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    options,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="eir",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
