# Using code by Simon, from August 10th, 2020

# Handling arithmetic expressions

from qat.lang.AQASM import *
from numbers import Number
from functools import reduce
import sympy as sym

### We can "load" the stringified expressions back into python objects:
from qat.core.variables import evaluate_thrift_expression, ArithExpression, Variable, ALL_SYMBOLS


def gatedef_to_expr(gatedef):
    """
    Args:
        gatedef ('datamodel.ttypes.GateDefinition') : from a circuit.gateDic
    """
    if len(gatedef.syntax.parameters) == 0:
        return None
    else:
        return evaluate_thrift_expression(gatedef.syntax.parameters[0].string_p)


## Expressions are either Variables or ArithExpressions (i.e arithmetic operations with a list of children)
## We can try to write a function that checks that an expression is linear in a single variable


def detect_linear(expression):
    """
    Returns True iff the expression is a linear function of a single variable.
    """
    if isinstance(expression, Number):  # If it's a constant, its linear
        return True
    if isinstance(expression, Variable):  # If it's a single variable, it's linear :)
        return True
    if len(expression.get_variables()) > 1:  # If it has more than one variables returns False ### THIS COULD BE IMPROVED!
        return False
    if expression.symbol.token in ["+", "-"]:
        # A +/- B is linear iff A and B are both linear
        return reduce(lambda a, b: a & b, map(detect_linear, expression.children))
    if expression.symbol.token == "*":
        # A * B is linear iff only one is linear and the other is constant
        return (isinstance(expression.children[0], Number) and detect_linear(expression.children[1])) or (
            isinstance(expression.children[1], Number) and detect_linear(expression.children[0])
        )
    if expression.symbol.token == "**":
        # A ** B is never linear
        return False
    if expression.symbol.token == "/":
        # A / B is linear iff A is linear and B is constant (this not true because we could have B = 1/C with C linear)
        # But i'm too lazy to think this through
        return detect_linear(expression.children[0]) and isinstance(expression.children[1], Number)
    if expression.symbol.token == "UMINUS":
        # - A is linear iff A is linear
        return detect_linear(expression.children[0])
    ## All other symbols are non linear (cos, sin, exp, sqrt, etc)
    return False


## Now we can reuse the same structure to additionaly extract the coefficient:


def improved_detect_linear(expression):
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
        is_0_lin, coeff_0 = improved_detect_linear(expression.children[0])
        is_1_lin, coeff_1 = improved_detect_linear(expression.children[1])
        if is_0_lin and is_1_lin:
            return True, coeff_0 + coeff_1
        return False, None
    if expression.symbol.token == "*":
        # A * B is linear iff only one is linear and the other is constant
        is_0_lin, coeff_0 = improved_detect_linear(expression.children[0])
        is_1_lin, coeff_1 = improved_detect_linear(expression.children[1])
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
        # But i'm too lazy to think this through
        is_lin, coeff = improved_detect_linear(expression.children[0])
        if is_lin and isinstance(expression.children[1], Number):
            return True, coeff / expression.children[1]
        return False, None
    if expression.symbol.token == "UMINUS":
        # - A is linear iff A is linear
        is_lin, coeff = improved_detect_linear(expression.children[0])
        if is_lin:
            return True, -coeff
        return False, None
    ## All other symbols are non linear (cos, sin, exp, sqrt, etc)
    return False, None


def sympy_detect_linear(expression, symvar):
    """
    Returns (True, sympyexpr) if the expression is a linear function of a single variable.
    else returns (False, None)

    """

    if isinstance(expression, Number):  # If it's a constant, its linear, with coefficient 0
        return True, 0 * symvar
    if isinstance(expression, Variable):  # If it's a single variable, it's linear with coefficient 1
        return True, 1 * symvar
    if len(expression.get_variables()) > 1:  # If it has more than one variables returns False ### THIS COULD BE IMPROVED!
        return False, None
    if expression.symbol.token in ["+", "-"]:
        # A +/- B is linear iff A and B are both linear
        is_0_lin, coeff_0 = sympy_detect_linear(expression.children[0], symvar)
        is_1_lin, coeff_1 = sympy_detect_linear(expression.children[1], symvar)
        if is_0_lin and is_1_lin:
            return True, coeff_0 + coeff_1
        return False, None
    if expression.symbol.token == "*":
        # A * B is linear iff only one is linear and the other is constant
        is_0_lin, coeff_0 = sympy_detect_linear(expression.children[0], symvar)
        is_1_lin, coeff_1 = sympy_detect_linear(expression.children[1], symvar)
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
        # But i'm too lazy to think this through
        is_lin, coeff = sympy_detect_linear(expression.children[0], symvar)
        if is_lin and isinstance(expression.children[1], Number):
            return True, coeff / expression.children[1]
        return False, None
    if expression.symbol.token == "UMINUS":
        # - A is linear iff A is linear
        is_lin, coeff = sympy_detect_linear(expression.children[0], symvar)
        if is_lin:
            return True, -coeff
        return False, None
    ## All other symbols are non linear (cos, sin, exp, sqrt, etc)
    return False, None


def test():

    #### A simple parametrized circuit

    prog = Program()
    a = prog.new_var(float, "a")
    b = prog.new_var(float, "b")
    qbits = prog.qalloc(1)
    ## good?
    RZ(2 * a)(qbits)
    RZ(b * 6)(qbits)
    RZ(b / 7 * 6**2 + 123 + 4 * b)(qbits)
    ## not good?
    RZ(1 / b)(qbits)
    RZ(a * b)(qbits)
    ## still good, but tricky
    RZ(1 / (1 / b))(qbits)  # This one because it reduces to "b"
    RZ(a + b)(qbits)  # And this one because it could be split in to RZ(a), RZ(b), both are linear

    circuit = prog.to_circ()

    gate_defs = [circuit.gateDic[op.gate] for op in circuit]

    print("Raw expressions:")
    for g in gate_defs:
        print(g.syntax.parameters[0].string_p)

    ### We can "load" the stringified expressions back into python objects:

    expressions = [gatedef_to_expr(g) for g in gate_defs]
    # expressions = [gd[1][0] for gd in circuit.iterate_simple()] ??? Will this do the same things ?
    print("Nicer expressions:")
    for expr in expressions:
        print(expr)

    ## Expressions are either Variables or ArithExpressions (i.e arithmetic operations with a list of children)
    ## We can try to write a function that checks that an expression is linear in a single variable

    ## ArithExpression are labelled by some arithmetic symbol.
    ## In:
    print("Possible symbols are:")
    for symbol in ALL_SYMBOLS:
        print(symbol)

    print("=" * 20)
    for expression in expressions:
        print(expression, "is ", end="")
        if detect_linear(expression):
            print("linear")
        else:
            print("not linear")

    ## Now we can reuse the same structure to additionaly extract the coefficient:

    print("=" * 20)
    for expression in expressions:
        print(improved_detect_linear(expression))


if __name__ == "__main__":
    test()
