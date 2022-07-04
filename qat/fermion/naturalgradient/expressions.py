from numbers import Number

from qat.core.variables import evaluate_thrift_expression, Variable


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
    Returns (True, coeff) if the expression is a linear function of a single variable.
    else returns (False, None)

    """
    if isinstance(expression, Number):  # If it's a constant, its linear, with coefficient 0
        return True, 0.0

    if isinstance(expression, Variable):  # If it's a single variable, it's linear with coefficient 1
        return True, 1.0

    if len(expression.get_variables()) > 1:  # If it has more than one variables returns False ### THIS COULD BE IMPROVED!
        return False, None

    if expression.symbol.token in ["+", "-"]:

        # A +/- B is linear iff A and B are both linear
        is_0_lin, coeff_0 = detect_linear(expression.children[0])
        is_1_lin, coeff_1 = detect_linear(expression.children[1])

        if is_0_lin and is_1_lin:
            return True, coeff_0 + coeff_1

        return False, None

    if expression.symbol.token == "*":

        # A * B is linear iff only one is linear and the other is constant
        is_0_lin, coeff_0 = detect_linear(expression.children[0])
        is_1_lin, coeff_1 = detect_linear(expression.children[1])

        if is_0_lin and isinstance(expression.children[1], Number):
            return True, coeff_0 * expression.children[1]

        if is_1_lin and isinstance(expression.children[0], Number):
            return True, coeff_1 * expression.children[0]

        return False, None

    if expression.symbol.token == "**":

        # A ** B is never linear
        return False, None

    if expression.symbol.token == "/":

        # A / B is linear iff A is linear and B is constant (this not true because we could have B = 1/C with C linear)
        is_lin, coeff = detect_linear(expression.children[0])

        if is_lin and isinstance(expression.children[1], Number):
            return True, coeff / expression.children[1]

        return False, None

    if expression.symbol.token == "UMINUS":

        # - A is linear iff A is linear
        is_lin, coeff = detect_linear(expression.children[0])

        if is_lin:
            return True, -coeff

        return False, None

    ## All other symbols are non linear (cos, sin, exp, sqrt, etc)

    return False, None
