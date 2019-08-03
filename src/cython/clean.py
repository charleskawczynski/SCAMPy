import os
import shutil

def get_all_files_in_path_recursive(root_path):
  f = [[subdir + os.sep + x for x in files] for subdir, dirs, files in os.walk(root_path) if files]
  return [item for sublist in f for item in sublist]

root_path = "./"
all_files = get_all_files_in_path_recursive("./")
all_files = [x for x in all_files if not x.endswith('.pyx')]
all_files = [x for x in all_files if not x.endswith('.pxd')]
all_files = [x for x in all_files if not x.endswith('.py')]
all_files = list(set(all_files))

try:
  for f in all_files:
    os.remove(f)
  shutil.rmtree('build')
except:
  pass
