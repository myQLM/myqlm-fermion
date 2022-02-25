#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing transforms
"""
import unittest
from itertools import product
import numpy as np

from qat.core import Term
from qat.dqs.transforms import transform_to_jw_basis, transform_to_parity_basis, transform_to_bk_basis
from qat.dqs.transforms import get_jw_code, get_parity_code, get_bk_code, change_encoding
from qat.dqs.hamiltonians import FermionHamiltonian


class TestBasic(unittest.TestCase):
    pass


def check_basic(trafo, code, hamilt_f):
    def test(self):
        hamilt_s = trafo(hamilt_f)
        A = hamilt_s.get_matrix()
        B = hamilt_f.get_matrix()
        B_recoded = change_encoding(B, code)
        self.assertAlmostEqual(np.linalg.norm(A - B_recoded), 0, delta=1e-10)
    return test


class MakeTestBasic(unittest.TestCase):
    nqbits = 5
    test_list = [("jw", transform_to_jw_basis, get_jw_code(nqbits)),
                 ("parity", transform_to_parity_basis, get_parity_code(nqbits)),
                 ("bk", transform_to_bk_basis, get_bk_code(nqbits))]

    H_f = FermionHamiltonian(nqbits, [Term(1., "C", [0])])
    H_f2 = FermionHamiltonian(nqbits, [Term(1., "C", [1])])
    H_f3 = FermionHamiltonian(nqbits, [Term(1., "CCcc", [0, 1, 1, 0]),
                                       Term(1., "CCcc", [1, 0, 0, 1])])
    H_f4 = FermionHamiltonian(nqbits, [Term(1., "CCcc", [0, 1, 1, 0]),
                                       Term(1., "CCcc", [1, 0, 0, 1]),
                                       Term(0.1, "Cc", [1, 0])])
    h_list = [H_f, H_f2, H_f3, H_f4]

    for (name, trafo, code), (ind, hamilt) in product(test_list, enumerate(h_list)):
        setattr(TestBasic,
                "test_%s_h_%s" % (name, ind),
                check_basic(trafo, code, hamilt))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
