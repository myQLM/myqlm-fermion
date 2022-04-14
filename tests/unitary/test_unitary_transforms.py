#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing transforms
"""
import pytest
import numpy as np

from qat.core import Term
from qat.fermion.transforms import (
    transform_to_jw_basis,
    transform_to_parity_basis,
    transform_to_bk_basis,
)
from qat.fermion.transforms import (
    get_jw_code,
    get_parity_code,
    get_bk_code,
    change_encoding,
)
from qat.fermion.hamiltonians import Hamiltonian


nqbits = 5
test_list = [
    (transform_to_jw_basis, get_jw_code(nqbits)),
    (transform_to_parity_basis, get_parity_code(nqbits)),
    (transform_to_bk_basis, get_bk_code(nqbits)),
]

H_f1 = Hamiltonian(nqbits, [Term(1.0, "C", [0])])
H_f2 = Hamiltonian(nqbits, [Term(1.0, "C", [1])])
H_f3 = Hamiltonian(nqbits, [Term(1.0, "CCcc", [0, 1, 1, 0]), Term(1.0, "CCcc", [1, 0, 0, 1])])
H_f4 = Hamiltonian(
    nqbits,
    [
        Term(1.0, "CCcc", [0, 1, 1, 0]),
        Term(1.0, "CCcc", [1, 0, 0, 1]),
        Term(0.1, "Cc", [1, 0]),
    ],
)
h_list = [H_f1, H_f2, H_f3, H_f4]


@pytest.mark.parametrize("transform_code", test_list)
@pytest.mark.parametrize("fermion_hamiltonian", h_list)
def test_check_basic(transform_code, fermion_hamiltonian):
    transform, code = transform_code
    spin_hamiltonian = transform(fermion_hamiltonian)
    A = spin_hamiltonian.get_matrix()
    B = fermion_hamiltonian.get_matrix()
    B_recoded = change_encoding(B, code)
    np.testing.assert_almost_equal(np.linalg.norm(A - B_recoded), 0, decimal=10)
