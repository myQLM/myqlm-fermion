import unittest
import numpy as np

from qat.plugins import MultipleLaunchesAnalyzer

from qat.core import Observable, Term
from qat.lang.AQASM import Program, RX
from qat.qpus import LinAlg

class TEstMLA(unittest.TestCase):

    def test_keep_min(self):
        prog = Program()
        qbits = prog.qalloc(5)
        for i, qb in enumerate(qbits):
            prog.apply(RX(0.324 * i), qb)
        circ = prog.to_circ()
        
        obs = Observable(5)
        for i in range(5):
            obs.add_term(Term(-0.5, "Z", [i]))
        obs.constant_coeff += 0.5*5
        
        job = circ.to_job("OBS", observable=obs, nbshots=30)
        qpu = LinAlg()
        stack = MultipleLaunchesAnalyzer() | qpu
        res = stack.submit(job)
        
        self.assertEqual(res.value, min(res.meta_data["reached_values"]))
        
if __name__=='__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
