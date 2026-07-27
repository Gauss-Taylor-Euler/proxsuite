import sys
import os
from time import time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_BUILD = os.path.join(CUR_DIR, "../../../../../build/bindings/python/proxsuite")
sys.path.insert(0, PYTHON_BUILD)

import numpy as np
from scipy import sparse
import clarabel
import proxsuite_pywrap.proxqp as proxqp
import proxsuite_pywrap.proxgqp as pgqp
import matplotlib.pyplot as plt


def generate_random_qp(n, seed=1):
    np.random.seed(seed)

    n_eq = n // 4
    n_in = n // 4

    M = np.random.randn(n, n)
    P = M @ M.T + 1e-2 * np.eye(n)
    g = np.random.randn(n)

    A_eq = np.random.randn(n_eq, n)
    A_in = np.random.randn(n_in, n)
    A = np.vstack([A_eq, A_in]) 

    v = np.random.randn(n)
    u = A @ v
    b_eq = u[:n_eq] 
    b_in = u[n_eq:] 

    return P, g, A_eq, b_eq, A_in, b_in, n_eq, n_in


n = 1000

H, g, A_eq, b_eq, A_in, b_in, n_eq, n_in = generate_random_qp(n, seed=42)
gqp = pgqp.GQP(n)
gqp.settings.eps_abs = 1e-10
gqp.setObjective(H, g)
orthant = pgqp.PositiveOrthantCone(n_in)

gqp.settings.max_iter =  10
gqp.settings.max_iter_in = 10

gqp.addEqualityConstraint(A_eq, b_eq)

b_in_1d = b_in.reshape(-1)
gqp.addInequalityConstraint(A_in, -b_in_1d, orthant)

result= gqp.solve(debug = True,strategy= pgqp.Strategy.BaseProxqpLike)

#result= gqp.solve(debug = True,strategy= pgqp.Strategy.GMRESSimpleIterativeSolver)


