import sys
import os

#So that we can import from the module build
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_BUILD = os.path.join(CUR_DIR, "../../../../../build/bindings/python/proxsuite")
sys.path.insert(0, PYTHON_BUILD)

import numpy as np

import proxsuite_pywrap.proxgqp as pgqp


dim =  3

sparseGqp =  pgqp.SparseGQP(dim)

sparseGqp.test()

