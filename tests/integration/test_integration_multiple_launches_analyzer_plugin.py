import numpy as np

from qat.plugins import MultipleLaunchesAnalyzer

from qat.core import Observable, Term
from qat.lang.AQASM import Program, RX
from qat.qpus import get_default_qpu


def test_keep_min():
    prog = Program()
    qbits = prog.qalloc(5)
    for i, qb in enumerate(qbits):
        prog.apply(RX(0.324 * i), qb)
    circ = prog.to_circ()

    obs = Observable(5)
    for i in range(5):
        obs.add_term(Term(-0.5, "Z", [i]))
    obs.constant_coeff += 0.5 * 5

    job = circ.to_job("OBS", observable=obs, nbshots=30)
    qpu = get_default_qpu()
    stack = MultipleLaunchesAnalyzer() | qpu
    res = stack.submit(job)

    np.testing.assert_equal(res.value, min(res.meta_data["reached_values"]))
