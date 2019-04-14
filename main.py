import argparse
import sys
import subprocess
import json

def run(case):
    # Parse information from the command line
    pythonVersion = 'python'+sys.version[0]
    subprocess.run('mkdir InputFiles', shell=True)
    subprocess.run('mkdir OutputFiles', shell=True)
    subprocess.run(pythonVersion+' generate_namelist.py '+case, shell=True)
    subprocess.run(pythonVersion+' generate_paramlist.py '+case, shell=True)

    file_namelist = open(case+'.in').read()
    namelist = json.loads(file_namelist)
    del file_namelist

    file_paramlist = open('paramlist_'+case+'.in').read()
    paramlist = json.loads(file_paramlist)
    del file_paramlist

    sol = main1d(namelist, paramlist)

    return sol

def main(**kwargs):
    case = kwargs['case']
    run(case)

def main1d(namelist, paramlist):
    import Simulation1d
    Simulation = Simulation1d.Simulation1d(namelist, paramlist)
    Simulation.initialize(namelist)
    sol = Simulation.run()
    print('The simulation has completed.')
    return sol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCAMPy.')
    parser.add_argument('case', type=str, help='SCAMPy case')
    args = parser.parse_args()
    sol = main(**vars(args))
