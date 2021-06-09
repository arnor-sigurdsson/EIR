# -*- mode: python ; coding: utf-8 -*-
import os 

# https://stackoverflow.com/questions/38977929/pyinstaller-creating-exe-runtimeerror-maximum-recursion-depth-exceeded-while-ca
import sys
sys.setrecursionlimit(5000)

# work-around for https://github.com/pyinstaller/pyinstaller/issues/4064
# import distutils
# if hasattr(distutils, 'distutils_path') and distutils.distutils_path.endswith('__init__.py'):
#     distutils.distutils_path = os.path.dirname(distutils.distutils_path)

# See: https://stackoverflow.com/questions/57517371/matplotlibdeprecationwarning-with-pyinstaller-exe 
# for mpl DeprecationWarning showing up after compiling

options = [('W ignore', None, 'OPTION')]

block_cipher = None

a = Analysis(['../eir/train.py'],
             binaries=[],
             datas=[],
             pathex=[],
             hiddenimports=[
                 'sklearn.utils._cython_blas',
                 'pkg_resources.py2_warn',
                 'sklearn.neighbors._typedefs',
                 'sklearn.neighbors._quad_tree',
                 'sklearn.utils._weight_vector',
                 'sklearn.neighbors',
                 'sklearn.tree._utils',
                 'sklearn.tree',
                 'ray.async_compat',
             ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          options,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='train',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
