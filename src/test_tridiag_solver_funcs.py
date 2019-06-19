
from scipy.sparse import diags
import numpy as np
import random
import TriDiagSolverFuncs as TDMA

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
    print('err = ', err)

test_tridiag_solver()
