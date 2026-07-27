import sys
import os
from time import time
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_BUILD = os.path.join(CUR_DIR, "../../../../../build/bindings/python/proxsuite")
sys.path.insert(0, PYTHON_BUILD)
import numpy as np
from scipy import sparse
import clarabel
import proxsuite_pywrap.proxgqp as pgqp

def oneRun(verbose=False):
    #This example is adapted from clarabel python doc : https://clarabel.org/stable/python/getting_started_py/
    H = np.array([[ 3., 1., -1.],
                  [ 1., 4.,  2.],
                  [-1., 2.,  5.]])
    g = np.array([1., 2., -3.])
    #equality constraint
    Aeq = np.array([1., 1., -1.], dtype="float64").reshape((1, 3))
    beq = np.array([1.])
    # equality constraint
    Aineq = np.array([[0., 1., 0.],
                      [0., 0., 1.]])
    bineq = np.array([2., 2.])
    # SOC constraint
    Asoc = -np.identity(3)
    bsoc = np.array([0., 0., 0.])
    #Solving with clarabel
    if verbose:
        print("===Solving with clarabel===")
    HClarabelAdapted =  sparse.triu(sparse.csc_matrix(H)).tocsc()
    AeqClarabelAdapted =  sparse.csc_matrix(Aeq)
    AineqClarabelAdapted = sparse.csc_matrix(Aineq)
    AsocClarabelAdapted = sparse.csc_matrix(Asoc)  
    AAggregatedClarabel = sparse.vstack([AeqClarabelAdapted,AineqClarabelAdapted,AsocClarabelAdapted]).tocsc()
    bAggregatedClarabel = np.concatenate([beq,bineq,bsoc])
    clarabelCones= [clarabel.ZeroConeT(1),
             clarabel.NonnegativeConeT(2),
             clarabel.SecondOrderConeT(3)]
    clarabelSettings = clarabel.DefaultSettings()
    clarabelSettings.verbose =  False
    solver = clarabel.DefaultSolver(HClarabelAdapted,g,AAggregatedClarabel,bAggregatedClarabel,clarabelCones,clarabelSettings)
    tStart = time()
    solution = solver.solve()
    tClarabel = time() - tStart
    if verbose:
        print(solution)
        print(f"Clarabel solve time : {tClarabel*1e3:>12.4f} ms")
        print("===END SOLVING WITH CLARABEL===")
    #Solving with proxgqp
    if verbose:
        print("===Solving with PROXGQP===")
    gqp =  pgqp.GQP(3)
    gqp.settings.eps_abs = 1e-10
    gqp.setObjective(H,g)
    lorentz = pgqp.LorentzCone(3)
    orthant =  pgqp.PositiveOrthantCone(2) 
    #Clarabel use the convent Ax-b\leq_K 0 for a couple (A,b) 
    #Hence the need to change the sign of b below in equality
    gqp.addInequalityConstraint(Aineq,-bineq,orthant)
    #Moreover Clarabel use the convention for the second order cone that (t,x) \in Lorentz \iff t\geq \|x\|_2 i.e place the apex at the start
    #In proxgqp we use the classic (x,t) \in K \iff t\geq \|x\|_2 Lorentz hence we need to change Asoc i.e we place the apex at the end
    perm = [1, 2, 0]
    gqp.addInequalityConstraint(Asoc[perm,:],-bsoc[perm],lorentz)
    #No change needed for equality
    gqp.addEqualityConstraint(Aeq,beq)
    tStart = time()
    result = gqp.solve()
    tSolveGQP = time() - tStart
    if verbose:
        print("x",result.x) 
        print("y",result.y)
        print("z",result.z)
        print("pri_res",result.pri_res)
        print("dual_res", result.dua_res)
        print("outerIters",result.outerIters)
        print("innerIters", result.totalInnerIters)
        print(f"proxgqp solve time  : {tSolveGQP*1e3:>12.4f} ms")
        print("===END SOLVING WITH PROXGQP===")
        # === Summary ===
        print("\n=== Timing Summary One Run===")
        print(f"{'':15s} Solve(ms)")
        print(f"Clarabel {tClarabel*1e3:>12.4f}")
        print(f"proxgqp  {tSolveGQP*1e3:>12.4f}")
    return (tSolveGQP, tClarabel)

oneRun(verbose=True)

# === Averaging over N runs ===
N = 1000
timesGQP = []
timesClarabel = []
for _ in range(N):
    tSolveGQP, tClarabel = oneRun(verbose=False)
    timesGQP.append(tSolveGQP)
    timesClarabel.append(tClarabel)

timesGQP = np.array(timesGQP)
timesClarabel = np.array(timesClarabel)

print(f"\n=== Timing Summary ({N} runs) ===")
print(f"{'':20s} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}")
print(f"{'Clarabel':20s} {timesClarabel.mean()*1e3:>12.4f} {timesClarabel.std()*1e3:>12.4f} {timesClarabel.min()*1e3:>12.4f} {timesClarabel.max()*1e3:>12.4f}")
print(f"{'proxgqp':20s} {timesGQP.mean()*1e3:>12.4f} {timesGQP.std()*1e3:>12.4f} {timesGQP.min()*1e3:>12.4f} {timesGQP.max()*1e3:>12.4f}")
