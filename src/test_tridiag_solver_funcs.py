
from scipy.sparse import diags
import numpy as np
import random
import funcs_tridiagsolver as TDMA
import TriDiagSolver as TDS

def test_tridiag_solver():
  tol = 0.00000001
  for n in range(3, 10):
    dl = np.random.rand(n-1)
    du = np.random.rand(n-1)
    d = np.random.rand(n);

    A = diags([dl, d, du], [-1,0,1]).toarray()
    b = np.random.rand(len(d))

    x_correct = np.matmul(np.linalg.inv(A), b)

    xtemp = np.zeros(n)
    gamma = np.zeros(n-1)
    beta = np.zeros(n)
    x_TDMA = np.zeros(n)

    TDMA.solve_tridiag(x_TDMA, b, dl, d, du, n, xtemp, gamma, beta)
    err = [abs(x-y) for (x, y) in zip(x_correct, x_TDMA)]
    assert all([x<tol for x in err])

    TDMA.init_β_γ(beta, gamma, dl, d, du, n)
    TDMA.solve_tridiag_stored(x_TDMA, b, dl, beta, gamma, n, xtemp)

    err = [abs(x-y) for (x, y) in zip(x_correct, x_TDMA)]
    assert all([x<tol for x in err])

    dl_mod = np.zeros(n)
    dl_mod[1:] = dl
    du_mod = np.zeros(n)
    du_mod[0:-1] = du
    TDS.tridiag_solve(n, b, dl_mod, d, du_mod)
    x_TDMA = b
    err = [abs(x-y) for (x, y) in zip(x_correct, x_TDMA)]
    assert all([x<tol for x in err])
    print('err = ', err)

test_tridiag_solver()
