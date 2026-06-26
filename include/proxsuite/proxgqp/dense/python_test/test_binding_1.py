import sys
import os

from time import time

#So that we can import from the module build
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_BUILD = os.path.join(CUR_DIR, "../../../../../build/bindings/python/proxsuite")
sys.path.insert(0, PYTHON_BUILD)

import numpy as np

import proxsuite_pywrap.proxgqp as pgqp

dim =  2
coneDim =  3

gqp =  pgqp.GQP(dim)

H =  np.zeros((dim,dim))
g =  np.array([-1.0,0.0])
#minimize -x_1
gqp.setObjective(H,g)

C =  np.array([[-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]])
d= np.array([0.0, 0.0, -1.0])

lorentz = pgqp.LorentzCone(coneDim)
#s.t ||(x_1,x_2)||_2 \leq 1
gqp.addInequalityConstraint(C, d, lorentz)

#gqp.settings.max_iter =  4
#gqp.settings.max_iter_in = 4

t0 = time()

result = gqp.solve(debug=True,strategy=pgqp.Strategy.Base)

tf =  time()

print((tf-t0)*1000)

#result = gqp.solve(debug=True,strategy=pgqp.Strategy.BaseWithoutProduct)
print(result.x,result.y,result.z,result.pri_res,result.dua_res)












