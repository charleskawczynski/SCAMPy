from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import os, platform, string
import sys
# sys.path.append('test')

include_path = [np.get_include()]

def get_all_files_in_path_recursive(root_path):
  f = [[subdir + os.sep + x for x in files] for subdir, dirs, files in os.walk(root_path) if files]
  return [item for sublist in f for item in sublist]

file_exclude = ['setup.py', 'main.py', 'run.py', 'build.py', 'clean.py']

all_files = get_all_files_in_path_recursive("./")
all_files = [x for x in all_files if not x.endswith('.exp')]
all_files = [x for x in all_files if not x.endswith('.obj')]
all_files = [x for x in all_files if not x.endswith('.pyd')]
all_files = [x for x in all_files if not x.endswith('.lib')]
all_files = [x for x in all_files if not x.endswith('.c')]
all_files = [x for x in all_files if not x=='setup']
all_files = [x for x in all_files if not '__pycache__' in x]
all_files = [x for x in all_files if not 'test_cache' in x]
all_files = [x for x in all_files if not any(y in x for y in file_exclude)]
print(all_files)
all_cython = [x for x in all_files if x.endswith('.pxd') or x.endswith('.pyx')]
all_cython = [x.replace(".pyx", "") for x in all_cython]
all_cython = [x.replace(".pxd", "") for x in all_cython]
# all_files = [x.replace('./', "") for x in all_files]
# all_files = [x.replace('/', "") for x in all_files]
# all_files = [x.replace('\\', "") for x in all_files]
# all_files = [x for x in all_files if "Grid" in x]
all_cython = list(set(all_cython))
print(all_cython)

include_path = [np.get_include()]
extensions = []
for f in all_cython:
  f_mod = f
  f_mod = f_mod.replace("./", "")
  f_mod = f_mod.replace("\\", os.sep)
  f_mod = f_mod.replace("/", os.sep)
  f_mod = f_mod.split(os.sep)[-1]
  extensions+=[Extension(f_mod, [f+'.pyx'])]
setup(ext_modules=cythonize(extensions, verbose=1, include_path=include_path))
