import argparse
import json
import pprint
from sys import exit
import uuid
import ast

def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')

    args = parser.parse_args()

    case_name = args.case_name

    if case_name == 'Bomex':
        namelist = Bomex()
    elif case_name == 'life_cycle_Tan2018':
        namelist = life_cycle_Tan2018()
    elif case_name == 'Soares':
        namelist = Soares()
    elif case_name == 'Rico':
        namelist = Rico()
    elif case_name == 'TRMM_LBA':
        namelist = TRMM_LBA()
    elif case_name == 'ARM_SGP':
        namelist = ARM_SGP()
    elif case_name == 'GATE_III':
        namelist = GATE_III()
    elif case_name == 'DYCOMS_RF01':
        namelist = DYCOMS_RF01()
    elif case_name == 'GABLS':
        namelist = GABLS()
    elif case_name == 'SP':
        namelist = SP()
    else:
        print('Not a valid case name')
        exit()

    write_file(namelist)

def Soares():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 150
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 20.0

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 60.0
    namelist['time_stepping']['t_max'] = 8 * 3600.0

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Soares'
    namelist['meta']['casename'] = 'Soares'

    return namelist

def Bomex():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 75
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 100 / 2.5

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 20.0
    namelist['time_stepping']['t_max'] = 21600.0

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Bomex'
    namelist['meta']['casename'] = 'Bomex'

    return namelist

def life_cycle_Tan2018():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 75
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 100 / 2.5

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 30.0
    namelist['time_stepping']['t_max'] = 6*3600.0

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'life_cycle_Tan2018'
    namelist['meta']['casename'] = 'life_cycle_Tan2018'
    return namelist

def Rico():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 100
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 40.0

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 20.0
    namelist['time_stepping']['t_max'] = 86400.0

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Rico'
    namelist['meta']['casename'] = 'Rico'

    return namelist

def TRMM_LBA(): # yair

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 400
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 40

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 30.0
    namelist['time_stepping']['t_max'] = 21590.0

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'inverse_w'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'TRMM_LBA'
    namelist['meta']['casename'] = 'TRMM_LBA'

    return namelist

def ARM_SGP():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 220
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 20

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 14.5

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'ARM_SGP'
    namelist['meta']['casename'] = 'ARM_SGP'

    return namelist

def GATE_III(): # yair
    # adopted from: "Large eddy simulation of Maritime Deep Tropical Convection",
    # By Khairoutdinov et al (2009)  JAMES, vol. 1, article #15
    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 1700
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 10

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 5.0
    namelist['time_stepping']['t_max'] = 3600.0 * 24.0

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'GATE_III'
    namelist['meta']['casename'] = 'GATE_III'

    return namelist

def DYCOMS_RF01():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 120
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 10

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 10.0
    namelist['time_stepping']['t_max'] = 60 * 60 * 4.

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'DYCOMS_RF01'
    namelist['meta']['casename'] = 'DYCOMS_RF01'

    return namelist

def GABLS():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 128
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 3.125

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 0.4
    namelist['time_stepping']['t_max'] = 9 * 3600.0

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1 # Not important in SBL
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2' # Not important in SBL
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = False
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'GABLS'
    namelist['meta']['casename'] = 'GABLS'

    return namelist

# Sullivan Patton not fully implemented - Ignacio
def SP():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 256
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 8

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 5.0
    namelist['time_stepping']['t_max'] = 7200.0

    namelist['turbulence'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = False
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False

    namelist['output'] = {}
    namelist['output']['output_root'] = 'OutputFiles/'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'SP'
    namelist['meta']['casename'] = 'SP'

    return namelist

def write_file(namelist):

    try:
        type(namelist['meta']['simname'])
    except:
        print('Casename not specified in namelist dictionary!')
        print('FatalError')
        exit()

    namelist['meta']['uuid'] = str(uuid.uuid4())

    fh = open(namelist['meta']['simname'] + '.in', 'w')
    #pprint.pprint(namelist)
    json.dump(namelist, fh, sort_keys=True, indent=4)
    fh.close()

    return

if __name__ == '__main__':
    main()
