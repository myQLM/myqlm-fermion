# -*- coding: utf-8 -*-
"""
Differential expressions
"""

from numbers import Number

from qat.core.variables import evaluate_thrift_expression


def gatedef_to_expr(gatedef):
    """
    Args:
        gatedef ('datamodel.ttypes.GateDefinition') : from a circuit.gateDic
    """
    if len(gatedef.syntax.parameters) == 0:
        return None
    else:
        return evaluate_thrift_expression(gatedef.syntax.parameters[0].string_p)


def detect_linear(expression):
    """
    Returns the derivative coefficient (respectively None) if the expression is a linear (respectively nonlinear) function.
    """

    if isinstance(expression, Number):  # If it's a constant, its linear, with coefficient 0
        return 0.0

    if len(expression.get_variables()) > 1:  # If it has more than one variables returns False
        return None

    diff = expression.differentiate(expression.get_variables()[0])

    if isinstance(diff, Number):
        return diff

    return None
