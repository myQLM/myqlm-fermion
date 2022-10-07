#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing fermion_algebra.py
"""
import pytest
from qat.core import Term
from qat.fermion.fermion_algebra import normal_order_fermionic_term

def test_normal_ordering():
    term = Term(1, "CcCc", [0, 1, 1, 0])
    new_terms = normal_order_fermionic_term(term)
    assert(len(new_terms)==2)
    for t in new_terms:
        if t.op=="Cc":
            assert(t.qbits == [0, 0])
            assert(t.coeff == 1)
        if t.op=="CCcc":
            assert(t.qbits == [0, 1, 0, 1])
            assert(t.coeff == 1)
