import argparse
import sys
import main
import copy
import numpy as np
import subprocess
import json

# Visual confirmation:

# python
# import main
# import matplotlib.pyplot as plt
# sol = main.run('Bomex')
# plt.plot(sol.ud_Area, sol.z) # Figure 4a in Tan et al.
# plt.plot(sol.ud_W, sol.z) # Figure 4b in Tan et al.
# plt.plot(sol.gm_QL, sol.z) # Figure 3a in Tan et al.

def expected_solutions(cases):
    sol_expected = dict()
    for case in cases:
        sol_expected[case] = type('', (), {})()

    sol_expected['Bomex'].ud_Area = np.array([0.         ,0.        , 0.1       , 0.04317573, 0.03344717, 0.03102328, 0.03073007 ,0.03139641, 0.03264808, 0.03434513, 0.03646311, 0.03905529, 0.04223683 ,0.04616719, 0.05085191, 0.05433439, 0.04761601, 0.03927854, 0.03163376 ,0.0256378 , 0.02120805, 0.01795018, 0.0154987 , 0.01359543, 0.01207731 ,0.01084383, 0.00983022, 0.00899148, 0.00829463, 0.00771524, 0.00723554 ,0.00684286, 0.00652837, 0.00628582, 0.00610984, 0.00599407, 0.00593062 ,0.00591635, 0.0060034 , 0.0067545 , 0.01568283, 0.00064456, 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.         ,0.        , 0.        , 0.        , 0.        , 0., 0.        ])
    sol_expected['Bomex'].ud_W    = np.array([0.         ,0.         ,0.41149104 ,0.71927115 ,0.88189502 ,0.9676495, 1.01145916 ,1.02838956 ,1.02586197 ,1.0081834  ,0.9779149  ,0.93633953, 0.88386629 ,0.82119974 ,0.75486688 ,0.72614949 ,0.76016599 ,0.83319505, 0.92282792 ,1.01449717 ,1.09996236 ,1.17503884 ,1.2380248  ,1.28858, 1.32708413 ,1.35422412 ,1.3707057  ,1.3770754  ,1.37364536 ,1.36047714, 1.33738098 ,1.30394495 ,1.25966871 ,1.20430221 ,1.13843172 ,1.06407048, 0.98447922 ,0.9017051  ,0.80486854 ,0.5987542  ,0.02238589 ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.         ,0.         ,0.         ,0.         ,0.         ,0., 0.        ])
    sol_expected['Bomex'].gm_QL   = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.53632734e-06, 5.53932467e-06, 6.38458946e-06, 6.62020895e-06, 6.56332807e-06, 6.44154664e-06, 6.29837570e-06, 6.15065432e-06, 5.99818590e-06, 5.83883448e-06, 5.67295106e-06, 5.50285555e-06, 5.33134119e-06, 5.16099299e-06, 4.99440420e-06, 4.83472099e-06, 4.68609491e-06, 4.55386700e-06, 4.44420269e-06, 4.36268452e-06, 4.31221348e-06, 4.29500274e-06, 4.34394083e-06, 4.78160397e-06, 8.89864959e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])

    sol_expected['Soares'].ud_Area = np.array([0.         ,0.         ,0.1        ,0.04251299 ,0.03277496, 0.02975554, 0.02846231 ,0.02785093 ,0.02763261 ,0.02768593 ,0.02791986, 0.02826043, 0.02866258 ,0.02910877 ,0.02958337 ,0.03006854 ,0.03056124, 0.03107613, 0.03164449 ,0.03230314 ,0.03307768 ,0.03397065 ,0.03495639, 0.03598832, 0.03701224 ,0.03798107 ,0.03887076 ,0.03968393 ,0.04044263, 0.04117368, 0.04189439 ,0.04260757 ,0.04330753 ,0.04399306 ,0.04468032, 0.04540713, 0.04622804 ,0.04720076 ,0.04837014 ,0.04975476 ,0.05134004, 0.05307878, 0.05489866 ,0.05671465 ,0.05844416 ,0.06002179 ,0.06141041, 0.0626057, 0.06363292 ,0.06453711 ,0.06537048 ,0.0661815  ,0.0670094 , 0.06788449, 0.06883257 ,0.06987954 ,0.07105296 ,0.07237927 ,0.07387758, 0.07555273, 0.07739069 ,0.07935814 ,0.08140662 ,0.08347985 ,0.08552239, 0.08748732, 0.08934166 ,0.09106912 ,0.09267039 ,0.09416202 ,0.0955744 , 0.09694952, 0.09833827 ,0.09979725 ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.1        ,0.1        ,0.1       , 0.1, 0.1        ,0.1        ,0.00884189 ,0.         ,0.        , 0., 0.         ,0.         ,0.         ,0.         ,0.        , 0., 0.         ,0.         ,0.         ,0.        ])
    sol_expected['Soares'].ud_W    = np.array([0.         ,0.        , 0.46313483, 0.77179621 ,0.93267333 ,1.02890194, 1.09455795 ,1.14258522, 1.17870998, 1.20586932 ,1.2259284  ,1.24034779, 1.2503688  ,1.25738946, 1.26184132, 1.26442741 ,1.26567543 ,1.26609655, 1.26610132 ,1.26596406, 1.26564896, 1.2650364  ,1.26374162 ,1.26161541, 1.25846623 ,1.25436201, 1.24950008, 1.24420078 ,1.23875303 ,1.23338692, 1.22814916 ,1.2230393 , 1.21798522, 1.21302058 ,1.20823965 ,1.20381718, 1.19991463 ,1.19661786, 1.19387279, 1.19149281 ,1.18918358 ,1.18662269, 1.18351378 ,1.17965016, 1.17493566, 1.16939624 ,1.16316135 ,1.15643502, 1.14945015 ,1.14242619, 1.13553155, 1.12887071 ,1.12248945 ,1.11639867, 1.11059767 ,1.1050887 , 1.09987508, 1.09494804 ,1.09027087 ,1.08577102, 1.08134477 ,1.07687339, 1.07224413, 1.06736858 ,1.06219363 ,1.0567036, 1.05091545 ,1.04487014, 1.03862329, 1.03223714 ,1.02577451 ,1.01929439, 1.01284756 ,1.00646518, 1.00013391, 0.99382746 ,0.9875106  ,0.98114317, 0.97468103 ,0.96807677, 0.96128208, 0.9542561  ,0.94697448 ,0.93942899, 0.93162166 ,0.92356348, 0.91527608, 0.90679124 ,0.89814942 ,0.88939879, 0.88059398 ,0.87179345, 0.86305502, 0.85443016 ,0.84595811 ,0.83766043, 0.82953725 ,0.8215661 , 0.81370402, 0.80589266 ,0.79806584 ,0.79015819, 0.78211362 ,0.77389188, 0.76547233, 0.75685408 ,0.7480527  ,0.73909421, 0.73000762 ,0.72081757, 0.71153848, 0.70217111 ,0.69270175 ,0.68310376, 0.67334047 ,0.66336862, 0.65314146, 0.64261101 ,0.63172964 ,0.62045079, 0.60872958 ,0.59652324, 0.58379145, 0.57049627 ,0.55660114 ,0.54206816, 0.52685317 ,0.51089813, 0.49412056, 0.47639953 ,0.45755617 ,0.43732146, 0.41526762 ,0.39063546, 0.36186395, 0.32640784 ,0.27948314 ,0.21035728, 0.1044034  ,0.00281186, 0.        , 0.         ,0.         ,0., 0.         ,0.        , 0.        , 0.         ,0.         ,0., 0.         ,0.        , 0.        , 0.        ])

    return sol_expected

def test_all_cases():
    all_tests = run_all_cases()
    passed_tests = [results[-1] for test in all_tests for results in test]
    assert all(passed_tests)

def run_all_cases():
    all_cases = ('Soares', 'Bomex', 'life_cycle_Tan2018', 'Rico', 'TRMM_LBA', 'ARM_SGP', 'GATE_III', 'DYCOMS_RF01', 'GABLS', 'SP')
    tol_rel_local = 0.1
    tol_rel = 0.05
    sol_expected = expected_solutions(all_cases)

    cases = ('Soares', 'Bomex')
    # cases = ('Bomex', )

    all_tests = [run_case(case, sol_expected[case], tol_rel_local, tol_rel) for case in cases]

    print('******************************* Summary test results:')
    for tests in all_tests:
        for test in tests:
            print(test)
    return all_tests

def run_case(case, sol_expected, tol_rel_local, tol_rel):
    tests = []
    sol = main.run(case)

    vars_to_compare = [name for name in dir(sol_expected) if name[:2] != '__' and name[-2:]]
    for vn in vars_to_compare:
        print('------------------------ '+vn)
        s_expected = getattr(sol_expected, vn)
        s_computed = getattr(sol, vn)
        print('s_expected = ',s_expected)
        print('s_computed = ',s_computed)
        assert len(s_expected)==len(s_computed)
        y_amax = np.amax(s_expected)
        abs_err = np.array([abs(x-y) for x,y in zip(s_computed, s_expected)])
        rel_err = np.array([x/y_amax for x in abs_err])
        print('y_amax = ', y_amax)
        print('abs_err = ', abs_err)
        print('rel_err = ', rel_err)
        passed_per_element = [e < tol_rel_local for e in rel_err]
        print('passed_per_element = ',passed_per_element)
        n_pass = passed_per_element.count(True)
        n_fail = passed_per_element.count(False)
        print('n_pass = ',n_pass)
        print('n_fail = ',n_fail)
        passed = n_fail/n_pass < tol_rel
        if passed:
            tests.append(('Pass: '+case+', '+vn, rel_err, passed))
        else:
            tests.append(('Fail: '+case+', '+vn, rel_err, passed))

    print('---------- Single test results:')
    for test in tests:
        print(test)
    return tests

if __name__ == "__main__":
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput

    # with PyCallGraph(output=GraphvizOutput()):
    #     sol = test_all_cases()
    sol = test_all_cases()
