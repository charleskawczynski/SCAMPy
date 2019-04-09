import argparse
import sys
import numpy as np
import subprocess
import json

# Visual confirmation:

# python
# import test
# import matplotlib.pyplot as plt
# sol = test.main()
# plt.plot(sol.ud_Area, sol.z) # Figure 4a in Tan et al.
# plt.plot(sol.ud_W, sol.z) # Figure 4b in Tan et al.
# plt.plot(sol.gm_QL, sol.z) # Figure 3a in Tan et al.

def main():
    # cases = ('Soares', 'Bomex', 'life_cycle_Tan2018', 'Rico', 'TRMM_LBA', 'ARM_SGP', 'GATE_III', 'DYCOMS_RF01', 'GABLS', 'SP')
    # cases = ('Bomex', 'Soares', )
    cases = ('Bomex', )

    sol_expected = dict()
    for case in cases:
        sol_expected[case] = type('', (), {})()

    vars_to_compare = ('ud_Area', 'ud_W')
    tol = 0.1
    vars_to_compare = ('ud_Area', 'ud_W', 'gm_QL')
    tests = []

    sol_expected['Bomex'].ud_Area = [0.         ,0.        , 0.1       , 0.04317573, 0.03344717, 0.03102328, 0.03073007 ,0.03139641, 0.03264808, 0.03434513, 0.03646311, 0.03905529, 0.04223683 ,0.04616719, 0.05085191, 0.05433439, 0.04761601, 0.03927854, 0.03163376 ,0.0256378 , 0.02120805, 0.01795018, 0.0154987 , 0.01359543, 0.01207731 ,0.01084383, 0.00983022, 0.00899148, 0.00829463, 0.00771524, 0.00723554 ,0.00684286, 0.00652837, 0.00628582, 0.00610984, 0.00599407, 0.00593062 ,0.00591635, 0.0060034 , 0.0067545 , 0.01568283, 0.00064456, 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.        ]
    sol_expected['Bomex'].ud_W    = [0.         ,0.         ,0.41149104 ,0.71927115 ,0.88189502 ,0.9676495, 1.01145916 ,1.02838956 ,1.02586197 ,1.0081834  ,0.9779149  ,0.93633953, 0.88386629 ,0.82119974 ,0.75486688 ,0.72614949 ,0.76016599 ,0.83319505, 0.92282792 ,1.01449717 ,1.09996236 ,1.17503884 ,1.2380248  ,1.28858, 1.32708413 ,1.35422412 ,1.3707057  ,1.3770754  ,1.37364536 ,1.36047714, 1.33738098 ,1.30394495 ,1.25966871 ,1.20430221 ,1.13843172 ,1.06407048, 0.98447922 ,0.9017051  ,0.80486854 ,0.5987542  ,0.02238589 ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.        ]
    sol_expected['Bomex'].gm_QL   = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.53632734e-06, 5.53932467e-06, 6.38458946e-06, 6.62020895e-06, 6.56332807e-06, 6.44154664e-06, 6.29837570e-06, 6.15065432e-06, 5.99818590e-06, 5.83883448e-06, 5.67295106e-06, 5.50285555e-06, 5.33134119e-06, 5.16099299e-06, 4.99440420e-06, 4.83472099e-06, 4.68609491e-06, 4.55386700e-06, 4.44420269e-06, 4.36268452e-06, 4.31221348e-06, 4.29500274e-06, 4.34394083e-06, 4.78160397e-06, 8.89864959e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]

    # sol_expected['Soares'].ud_Area = []
    # sol_expected['Soares'].ud_W    = []
    # sol_expected['Soares'].gm_QL   = []

    for case in cases:
        sol = type('', (), {})()
        # Parse information from the command line
        pythonVersion = 'python'+sys.version[0]
        subprocess.run(pythonVersion+' generate_namelist.py '+case, shell=True)
        subprocess.run(pythonVersion+' generate_paramlist.py '+case, shell=True)

        file_namelist = open(case+'.in').read()
        namelist = json.loads(file_namelist)
        del file_namelist

        file_paramlist = open('paramlist_'+case+'.in').read()
        paramlist = json.loads(file_paramlist)
        del file_paramlist

        sol = main1d(namelist, paramlist)

        var_names = [name for name in dir(sol) if name[:2] != '__' and name[-2:]]
        for vn in vars_to_compare:
            # print('------------------------ '+vn)
            # print(getattr(sol, vn))
            s_expected = getattr(sol_expected[case], vn)[2:-1]
            s_computed = getattr(sol, vn)[2:-1]
            y_amax = np.amax(s_expected)
            err = [abs(x-y)/(y_amax+1) for x,y in zip(s_computed, s_expected)]
            L = all([e < tol for e in err])
            if L:
                tests.append('pass: '+case+', '+vn)
            else:
                tests.append('fail: '+case+', '+vn)
            assert L
    print('Test results:')
    for test in tests:
        print(test)

    return sol

def main1d(namelist, paramlist):
    import Simulation1d
    Simulation = Simulation1d.Simulation1d(namelist, paramlist)
    Simulation.initialize(namelist)
    sol = Simulation.run()
    print('The simulation has completed.')
    return sol

if __name__ == "__main__":
    sol = main()
