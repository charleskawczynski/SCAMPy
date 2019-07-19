import subprocess
import sys, os
sys.path.append('test')
pythonVersion = 'python'
subprocess.run(pythonVersion+' clean.py', shell=True)
subprocess.run(pythonVersion+' build.py', shell=True)
subprocess.run(pythonVersion+' test'+os.sep+'test_all.py', shell=True)

