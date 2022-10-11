#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing fermion_algebra.py
"""
import pytest
from qat.fermion.fermion_algebra import FermionicTerm, normal_order_fermionic_term


def test_normal_ordering():
    term = FermionicTerm(1, "CcCc", [0, 1, 1, 0])
    new_terms = normal_order_fermionic_term(term)
    assert len(new_terms) == 2
    for t in new_terms:
        if t.op == "Cc":
            assert t.qbits == [0, 0]
            assert t.coeff == 1
        if t.op == "CCcc":
            assert t.qbits == [0, 1, 0, 1]
            assert t.coeff == 1
            
def test_normal_ordering_one_op():
    term = FermionicTerm(1.0, "c", [0])
    new_term = normal_order_fermionic_term(term)
    assert len(new_term) == 1
    
    new_term = new_term[0]
    assert new_term.op == "c"
    assert new_term.coeff == 1.0
    assert new_term.qbits == [0]
    
    term = FermionicTerm(1.0, "C", [0])
    new_term = normal_order_fermionic_term(term)
    assert len(new_term) == 1
    
    new_term = new_term[0]
    assert new_term.op == "C"
    assert new_term.coeff == 1.0
    assert new_term.qbits == [0]
    
def test_normal_ordering_only_C_operators():
    term = FermionicTerm(2.0, "CC", [1,0])

    new_term = normal_order_fermionic_term(term)
    
    assert len(new_term) == 1
    
    new_term = new_term[0]
    assert new_term.coeff == -2.0
    assert new_term.op == "CC"
    assert new_term.qbits == [0, 1]
    
def test_normal_ordering_only_c_operators():
    term = FermionicTerm(2.0, "cc", [1,0])

    new_term = normal_order_fermionic_term(term)
    
    assert len(new_term) == 1
    
    new_term = new_term[0]
    assert new_term.coeff == -2.0
    assert new_term.op == "cc"
    assert new_term.qbits == [0, 1]
            