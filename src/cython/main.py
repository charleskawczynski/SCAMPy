import subprocess

pythonVersion = 'python'
subprocess.run(pythonVersion+' setup.py build_ext --inplace', shell=True)

