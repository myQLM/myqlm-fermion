import numpy as np
import scipy.sparse as sp
from itertools import product
import qat.linalg
from qat.core.simutil import wavefunction

from qat.lang.AQASM import Program, QRoutine, AbstractGate, X, RY, CNOT
from qat.core import default_gate_set


def fSim_gen(theta, phi): return np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(theta), -1j * np.sin(theta), 0],
        [0, -1j * np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, np.exp(-1j * phi)],
    ],
    dtype="complex",
)


fSim = AbstractGate(
    "fSim",
    [float, float],
    2,
    matrix_generator=lambda theta, phi, mat_gen=fSim_gen: mat_gen(theta, phi),
)

fSim_gate_set = default_gate_set()
fSim_gate_set.add_signature(fSim)


def make_fSim_fan_routine(nbqbits, theta):
    """
    Generates the routine corresponding to the application of a 'fan'
    of fSim gates from the innermost qubits to the ones on the edge of
    the register.

    Args:
        nbqbits (int): (even) number of qubits of the register
        theta (np.array): parameters of the routine (2*(nbqbits - 1) parameters)

    Return:
        :class:`~qat.lang.AQASM.QRoutine': the quantum routine
    """

    qrout = QRoutine()
    ind_theta = 0

    q1 = nbqbits // 2 - 1
    q2 = nbqbits // 2

    qrout.apply(fSim(theta[ind_theta], theta[ind_theta + 1]), q1, q2)
    ind_theta += 2

    for j in range(nbqbits // 2 - 1):
        qrout.apply(
            fSim(theta[ind_theta], theta[ind_theta + 1]), q1 - j - 1, q1 - j)
        ind_theta += 2
        qrout.apply(
            fSim(theta[ind_theta], theta[ind_theta + 1]), q2 + j, q2 + j + 1)
        ind_theta += 2

    return qrout


def make_sugisaki_routine(theta):
    """
    A 4-qubit routine inspired from Sugisaki et al., 10.1021/acscentsci.8b00788 [2019] that acts on

    ...math::
        \ket{0011}

    as:

    .. math::
        \cos(\\theta/2) \ket{0011} + \sin(\\theta/2) \ket{1100}

    Returns:
        QRoutine
    """
    qrout = QRoutine()
    qrout.apply(RY(theta), 0)
    qrout.apply(CNOT, 0, 1)
    qrout.apply(CNOT, 0, 2)
    qrout.apply(CNOT, 1, 3)

    return qrout


def binary_generation(high, low):
    r"""
    This function returns a list of numbers which are the sum of
    all the different composition of 2**high, 2**high-1,...2**low (incl 0)

    Args:
        high (int): maximal power to consider
        low (int): minimal power to consider

    Returns:
        list:

    Example:
        binary_generation(5,5)=[32, 0],
        binary_generation(5,4)=[16, 48, 32, 0],
        binary_generation(5,3)=[8, 24, 16, 40, 56, 48, 32, 0]
    """
    a = np.array(range(high + 1))
    L = []
    X = []
    for k in a[low:]:
        for w in range(len(X)):
            L += [X[w] + 2**k]
        L += [2**k]
        X = L

    L += [0]
    return L


def inv_fractional_binary(binary_list):
    r"""
    This function returns a number given his fractional binary expension

    .. math::
            \sum_{i=0..n-1} b_i / 2^{i+1}

    Args:
        binary_list (list): fractional binary expension of a number between 0 and 1

    Returns:
        float: number associated to the binary_list

    Example:
        inv_fractionnal_binary([1,0, 1])=0.625,
        inv_fractionnal_binary([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])=0.4248046875
    """
    return sum([binary_list[i] / 2 ** (i + 1) for i in range(len(binary_list))])


def fractional_binary(number, precision):
    r"""
    This function returns the fractional binary expension of a number between 0 and 1 ie
    x=0.x_1x_2...x_L with x_1,...,x_L 0 or 1 and L the precision

    Args:
        number (float): number between 0 and 1
        precision (float): accuracy of the fractional binary at 2^{-precision}

    Returns:
        list: fractional binary of the number

    Example:
        fractional_binary(0.625,3)=[1,0,1], fractional_binary(0.425,10)=[0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    """
    if abs(number) > 1:
        raise Exception("Number must be between 0 and 1")

    binary_list = [0 for _ in range(precision)]
    for i in range(1, precision + 1):
        if (number - 1 / 2**i) >= 0:
            binary_list[i - 1] = 1
            number -= 1 / 2**i
    return binary_list


def give_coordinate_list(list1, list2):
    """
    Return a list with elements from list1 with their position in that list if those elements are in list2.
    Args:
        list1 (list): working list
        list2 (list): reference list
    Returns:
        list: elements of list1 and their position in that list if they are in list2

    """
    return [(i, u) for u, i in enumerate(list1) if i in list2]


def bitget(n, pos):
    return (n & (1 << pos)) != 0


def tobin(n, size):
    """returns binary representation of n with size size

    Example
    --------
    tobin(3,4) = '0011'
    """
    nbin = "{0:b}".format(n)
    l = len(nbin)
    for _ in range(size - l):
        nbin = "0" + nbin
    return nbin


def count_bits(n, cutoff, size):
    """takes an integer n and returns the sum of its first 'cutoff' bits

    Example:
        if binary representation is '001110' (size=6 bits),
        if cutoff==3: -> 0+0+1 = 1
    """
    nbin = tobin(n, size)
    rg = range(0, min(cutoff, size))
    s = sum([int(nbin[k]) for k in rg])
    return s


def init_creation_ops(Norb, sparse=False, verbose=False):
    r"""Initialize creation operators for a Fock space with Norb orbitals.

    Args:
        Norb (int): number of spin-orbitals
        sparse (bool, optional): whether to use a sparse representation.
            Defaults to False
        verbose (bool, optional): verbose output. Defaults to False.

    Returns:
        list<np.array/coo_matrix>: list of the matrices of the creation operators
        :math:`\lbrace c^\dagger_i\rbrace_{i=0\dots N_{orb}-1}` in the Fock basis

    Note:
        |10010> is c^dag_0 c^dag_3 |vac>
    """
    c_dagger_dict = {}
    for i in range(Norb):
        row_ind = []
        col_ind = []
        data = []
        if verbose:
            print("----------- creation op of color ", i)
        sign = None
        for j in range(2**Norb):  # for each state j
            if verbose:
                print("  applied to ", tobin(j, Norb))
            if bitget(j, Norb - 1 - i) == 0:  # if pos i in state j is empty
                sign = 1 if count_bits(j, i, Norb) % 2 == 0 else -1
                row_ind.append(j + 2 ** (Norb - i - 1))
                col_ind.append(j)
                data.append(sign)
            if verbose:
                print(
                    ". . . . . . . . ok : nb bits:"
                    + str(count_bits(j, i, Norb))
                    + "----->"
                    + str(sign)
                )

            else:
                if verbose:
                    print(". . . . . . . . ko")

        c_dagger_dict[i] = sp.coo_matrix(
            (data, (row_ind, col_ind)), shape=(2**Norb, 2**Norb)
        )
        if not sparse:
            c_dagger_dict[i] = c_dagger_dict[i].A
    return c_dagger_dict


def dag(A):
    return np.conj(np.transpose(A))


def get_unitary_from_circuit(Qrout, number_qubits):
    r"""
    This function return the matrix of a QRoutine.

    Args:
        Qrout (QRoutine): a quantum routine
        number_qubits (int): the number of qubits

    Returns:
        numpy.ndarray: matrix of the circuit
    """
    unitary_matrix = [0] * 2 ** (number_qubits)
    for i in list(product("IX", repeat=number_qubits)):
        p = Program()
        reg = p.qalloc(number_qubits)
        numero_colonne = 0
        for j in range(len(i)):
            if i[j] == "X":
                numero_colonne += 2 ** (len(i) - j - 1)
                p.apply(X, reg[j])
        p.apply(Qrout, reg[: Qrout.arity])
        circuit = p.to_circ()
        # pylint: disable=E1101
        unitary_matrix[numero_colonne] = list(
            wavefunction(circuit, qat.linalg.LinAlg())
        )
    return np.transpose(unitary_matrix)
