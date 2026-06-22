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

verbose = False

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


def run_one(n, seed):
    H, g, A_eq, b_eq, A_in, b_in, n_eq, n_in = generate_random_qp(n, seed=seed)

    Psp = sparse.triu(sparse.csc_matrix(H)).tocsc()
    Aeq_sp = sparse.csc_matrix(A_eq) 
    Ain_sp = sparse.csc_matrix(A_in) 

    Aagg = sparse.vstack([Aeq_sp, Ain_sp]).tocsc()

    bagg = np.concatenate([b_eq, b_in]) 

    cones = []
    cones.append(clarabel.ZeroConeT(n_eq))
    cones.append(clarabel.NonnegativeConeT(n_in))

    # Clarabel
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(Psp, g, Aagg, bagg, cones, settings)
    t0 = time()
    res = solver.solve()
    t_clarabel = (time() - t0) * 1e3
    if verbose:
        print(res)

    # ProxQP
    qp = proxqp.dense.QP(n, n_eq, n_in)
    l = np.zeros(b_in.shape) - 1e10
    qp.init(H, g, A_eq, b_eq, A_in, l, b_in)
    t0 = time()
    qp.solve( )
    t_proxqp = (time() - t0) * 1e3

    # ProxGQP
    gqp = pgqp.GQP(n)
    gqp.settings.eps_abs = 1e-10
    gqp.setObjective(H, g)
    orthant = pgqp.PositiveOrthantCone(n_in)

    gqp.addEqualityConstraint(A_eq, b_eq)

    b_in_1d = b_in.reshape(-1)
    gqp.addInequalityConstraint(A_in, -b_in_1d, orthant)

    t0 = time()
    result= gqp.solve(debug = False)
    t_proxgqp = (time() - t0) * 1e3

    if verbose:
        print("x",result.x) 
        print("y",result.y)
        print("z",result.z)
        print("pri_res",result.pri_res)
        print("dual_res", result.dua_res)
        print("outerIters",result.outerIters)
        print("innerIters", result.totalInnerIters)


    return t_clarabel, t_proxqp, t_proxgqp


def main():
    start_n, end_n = 900, 900
    n_trial =  4

    print(f"{'n':>6}  {'clarabel_ms':>12}  {'proxqp_ms':>10}  {'proxgqp_ms':>12}")

    ns_list = []
    clarabel_list = []
    proxqp_list = []
    proxgqp_list = []

    for n in range(start_n, end_n + 1):
        ts = []
        for k in range(n_trial):
            t = run_one(n, seed=12345 + k)
            ts.append(t)

        # average across trials
        ts = np.array(ts)  # shape (seeds, 3)
        t_cl, t_qp, t_gqp = ts.mean(axis=0)

        ns_list.append(n)
        clarabel_list.append(t_cl)
        proxqp_list.append(t_qp)
        proxgqp_list.append(t_gqp)

        print(f"{n:6d}  {t_cl:12.4f}  {t_qp:10.4f}  {t_gqp:12.4f}", flush=True)

    plt.figure(figsize=(10, 6))
    plt.plot(ns_list, clarabel_list, "o-", label="Clarabel", markersize=4)
    plt.plot(ns_list, proxqp_list, "s-", label="ProxQP", markersize=4)
    plt.plot(ns_list, proxgqp_list, "^-", label="ProxGQP", markersize=4)
    plt.xlabel("Problem size n")
    plt.ylabel("Time (ms)")
    plt.title("Solver Timing Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(CUR_DIR, "timing_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
