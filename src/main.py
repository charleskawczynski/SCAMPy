import cProfile
import argparse
import sys
import os
import subprocess
import json

def run(case):
    # Parse information from the command line
    # pythonVersion = 'python'+sys.version[0] # depends on how python is called
    pythonVersion = 'python'
    this_file = os.path.join(os.getcwd(),__file__)
    root_dir = os.path.dirname(os.path.dirname(this_file))+os.sep
    namelist_file = 'generate_namelist.py'
    paramlist_file = 'generate_paramlist.py'
    namelist_file = [root+os.sep+f for root, dirs, files in os.walk(root_dir) for f in files if f==namelist_file][0]
    paramlist_file = [root+os.sep+f for root, dirs, files in os.walk(root_dir) for f in files if f==paramlist_file][0]
    subprocess.run(pythonVersion+' '+namelist_file+' '+case, shell=True)
    subprocess.run(pythonVersion+' '+paramlist_file+' '+case, shell=True)

    file_namelist = open(case+'.in').read()
    namelist = json.loads(file_namelist)
    del file_namelist

    file_paramlist = open('paramlist_'+case+'.in').read()
    paramlist = json.loads(file_paramlist)
    del file_paramlist

    sol = main1d(namelist, paramlist, root_dir)

    return sol

def main(**kwargs):
    case = kwargs['case']
    sol = run(case)

def main1d(namelist, paramlist, root_dir):
    import Simulation1d
    Simulation = Simulation1d.Simulation1d(namelist, paramlist, root_dir)
    Simulation.initialize(namelist)
    sol = Simulation.run()
    print('The simulation has completed.')
    return sol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCAMPy.')
    parser.add_argument('case', type=str, help='SCAMPy case')
    args = parser.parse_args()
    sol = main(**vars(args))
