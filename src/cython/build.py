import subprocess
import sys
sys.path.append('test')

pythonVersion = 'python'
subprocess.run(pythonVersion+' setup.py build_ext --inplace', shell=True)

