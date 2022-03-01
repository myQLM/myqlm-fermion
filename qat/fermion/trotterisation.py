""" Functions for first-order Trotterization with Jordan-Wigner"""
from math import pi
import numpy as np
from qat.lang.AQASM import QRoutine, PH, CNOT, H, RX, RZ, CustomGate, Z
from .hamiltonians import ElectronicStructureHamiltonian, Hamiltonian


def make_trotterisation_routine(hamiltonian, n_trotter_steps, final_time):
    r"""
    This function first trotterises the evolution operator :math:`e^{-i H t}` of
    a fermionic Hamiltonian or spin Hamiltonian :math:`H` using
    a first order approximation.

    In the fermionic case:

    .. math::
         e^{-i H t} \approx \prod_{k=1}^{n} \left( \prod_{pq} e^{-i \frac{t}{n} h_{pq} c_p^\dagger c_q} \prod_{pqrs} e^{-\frac{i}{2}\frac{t}{n} h_{pqrs} e^{-i c_p^\dagger c_q^\dagger c_r c_s} } \right)

    This operator is then mapped to a product of Pauli operators via a
    Jordan-Wigner transformation and the resulting QRoutine is returned.

    Args:
        hamiltonian (ElectronicStructureHamiltonian): fermionic hamiltonian
        n_trotter_steps (int): number :math:`n` of Trotter steps.
        final_time (float): time :math:`t` in the evolution operator

    Returns:
        QRoutine: gates to apply to perform the time evolution
        of the chemical Hamiltonian with trotterisation

    Notes:
        Here the QRoutine implements a first order trotter approximation,
        higher order approximations are possible.

    """
    if isinstance(hamiltonian, Hamiltonian):
        Qrout = QRoutine()
        for _ in range(n_trotter_steps):
            Qrout.apply(
                make_spin_hamiltonian_trotter_slice(
                    hamiltonian, final_time / n_trotter_steps
                ),
                list(range(hamiltonian.nbqbits)),
            )
        return Qrout

    elif isinstance(hamiltonian, ElectronicStructureHamiltonian):
        Qrout = QRoutine()
        for _ in range(n_trotter_steps):
            Qrout.apply(
                make_trotter_slice_jw(
                    hamiltonian.hpq, hamiltonian.hpqrs, final_time / n_trotter_steps
                ),
                list(range(len(hamiltonian.hpq))),
            )
        return Qrout

    else:
        raise Exception(
            "Hamiltonian must be of type ElectronicStructureHamiltonian or "
            "Hamiltonian, got %s instead" % type(hamiltonian)
        )


def make_spin_hamiltonian_trotter_slice(hamiltonian, coeff=1.0):
    r"""
    Constructs the quantum routine corresponding to the first-order
    trotterization of

    .. math::
            e^{-i * coeff * H}

    where :math:`H` is a spin Hamiltonian.

    Args:
        hamiltonian (Hamiltonian): a spin Hamiltonian

    Returns:
        QRoutine: gates to apply to perform the time evolution

    """

    def _one_operator_circuit(op, qbits):
        r"""Construct cascade of CNOTs corresponds to Pauli string

        Args:
            op (str): string with X,Y,Z,I
            qbits (list<int>): list of bits on which they are applied

        Returns:
            QRoutine, qb: the routine and the index of last qbit
        """
        nqbits = len(qbits)
        _qbits = range(nqbits)
        Qrout = QRoutine()
        for qb, pauli in zip(_qbits, op):
            if pauli == "X":
                Qrout.apply(H, qb)
            if pauli == "Y":
                Qrout.apply(RX(np.pi / 2), qb)

        previous_qb = nqbits - 1
        for qb, pauli in zip(_qbits[::-1][1:], op[::-1][1:]):
            if pauli != "I":
                Qrout.apply(CNOT, previous_qb, qb)
                previous_qb = qb
        return Qrout, qbits[previous_qb]

    Qrout = QRoutine()
    for term in hamiltonian.terms:
        Qrout_one, ref = _one_operator_circuit(term.op, term.qbits)
        if Qrout_one.arity != 0:
            Qrout.apply(Qrout_one, term.qbits)
        Qrout.apply(RZ(2 * coeff * np.real(term.coeff)), ref)
        if Qrout_one.arity != 0:
            Qrout.apply(Qrout_one.dag(), term.qbits)
    return Qrout


def make_trotter_slice_jw(hpq, hpqrs, delta_t):
    r"""
    This function returns the circuit which corresponds to the time evolution
    ( :math:`e^{-it\hat{O}}`) of the chemical Hamiltonian

    .. math::
        H = \sum_{pq} h_{pq}a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s

    in second quantization.

    This uses the Jordan-Wigner transformation as encoding.

    Args:
        hpq (list): 2D list which contains all the hpq terms in the chemical Hamiltonian
        hpqrs (list): 4D list which contains all the hpqrs terms in the chemical Hamiltonian
        t (float): time in the evolution operator

    Returns:
        QRoutine: gates to apply to add the time evolution oh the chemical Hamiltonian

    Warning:
        hpq and hpqrs may be divided by :math:`\hbar `
        If hpq or hpqrs are imaginary it hasn't been tested

    Notes:
        We assume trotterisation because we developp the exponential of H as a product of exponentials.
        We take the convention |0> is empty and |1> is occupied.
        We used a custom gate to make a global phase to have the same expression given by the Jordan Wigner transformation.
    """
    Qrout = QRoutine()
    if len(hpq) != len(hpqrs):
        return "Error hpq and hpqrs must have the same dimension"
    hpqrs = (
        hpqrs / 2
    )  # In order to take in account the 1/2 coefficient in front of the sum

    Qrout.apply(_number_operator_jw(hpq, delta_t), range(len(hpq)))
    Qrout.apply(_excitation_operator_jw(hpq, delta_t), range(len(hpq)))
    Qrout.apply(_coulomb_exchange_operator_jw(hpqrs, delta_t), range(len(hpq)))

    if len(hpqrs) > 2:
        Qrout.apply(_number_excitation_operator_jw(hpqrs, delta_t), range(len(hpq)))
    if len(hpqrs) > 3:
        Qrout.apply(_double_excitation_operator_jw(hpqrs, delta_t), range(len(hpq)))

    return Qrout


def _number_operator_jw(hpq, t):
    r"""
    This function returns the circuit which corresponds to the time evolution
    ( :math:`e^{-it\hat{O}}`) of the number operator
    ( :math:`a_p^\dagger a_p` )  in second quantization after performing
    a Jordan-Wigner transformation.

    Args:
        hpq (list): 2D list which contains all the hpq terms in the chemical Hamiltonian
            :math:`H = \sum_{pq} h_{pq}a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s`
        t (float): time in the evolution operator
    Returns:
        QRoutine: gates to apply to add the time evolution number operator
    Warning:
        hpq may be divided by :math:`\hbar `
    """
    Qrout = QRoutine()
    for i in range(len(hpq)):
        Qrout.apply(PH((-hpq[i][i] * t)), i)
    return Qrout


def _excitation_operator_jw(hpq, t):
    r"""
    This function returns the circuit which corresponds to the time evolution
    ( :math:`e^{-it\hat{O}}`) of the excitation operator
    ( :math:`a_p^\dagger a_q` )  in second quantization after performing
    a Jordan-Wigner transformation.

    Args:
        hpq (list): 2D list which contains all the hpq terms in the chemical Hamiltonian
            :math:`H = \sum_{pq} h_{pq}a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s`
        t (float): time in the evolution operator
    Returns:
        QRoutine: gates to apply to add the time evolution excitation operator
    Warning:
        hpq may be divided by :math:`\hbar `
    """
    Qrout = QRoutine()
    for i in range(len(hpq)):
        # It's equivalent to multiply by identity. It's use to avoid
        # dimensionnal problem when building circuit
        Qrout.apply(PH(0), i)
        for j in range(i):
            if hpq[i][j].real != 0:
                Qrout.apply(H, j)
                Qrout.apply(H, i)
                for k in range(i - j):
                    Qrout.apply(CNOT, [i - k, i - k - 1])
                Qrout.apply(RZ((t * hpq[i][j].real)), j)
                for k in range(i - j):
                    Qrout.apply(CNOT, [j + k + 1, j + k])
                Qrout.apply(H, j)
                Qrout.apply(H, i)
                Qrout.apply(RX(-pi / 2), j)
                Qrout.apply(RX(-pi / 2), i)
                for k in range(i - j):
                    Qrout.apply(CNOT, [i - k, i - k - 1])
                Qrout.apply(RZ((t * hpq[i][j].real)), j)
                for k in range(i - j):
                    Qrout.apply(CNOT, [j + k + 1, j + k])
                Qrout.apply(RX(-pi / 2).dag(), j)
                Qrout.apply(RX(-pi / 2).dag(), i)

    return Qrout


def _coulomb_exchange_operator_jw(hpqrs, t):
    r"""
    This function returns the circuit which corresponds to the time evolution
    ( :math:`e^{-it\hat{O}}`) of the coulomb exchange operator
    ( :math:`a_p^\dagger a_q^\dagger a_q a_p` )  in second quantization after
    performing a Jordan-Wigner transformation.

    Args:
        hpqrs (list): 4D list which contains all the hpqrs terms in the chemical Hamiltonian
            :math:`H = \sum_{pq} h_{pq}a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s`
        t (float): time in the evolution operator
    Returns:
        QRoutine: gates to apply to add the time evolution coulomb exchange operator
    Warning:
        hpqrs may be divided by :math:`\hbar `
    """
    Qrout = QRoutine()
    for p in range(len(hpqrs)):
        # It's equivalent to multiply by identity. It's use to avoid
        # dimensional problem when building circuit
        Qrout.apply(PH(0), p)
        for q in range(p):
            hpqqp = (
                hpqrs[p][q][q][p]
                - hpqrs[q][p][q][p]
                - hpqrs[p][q][p][q]
                + hpqrs[q][p][p][q]
            )
            if hpqqp != 0:
                U = np.array(
                    [[np.exp(-1j * t * hpqqp / 4), 0], [0, np.exp(-1j * t * hpqqp / 4)]]
                )
                # G = CustomGate(U,"global_phase_%1.2f"%w)
                G = CustomGate(U)  # , "global_phase")
                Qrout.apply(G, q)
                Qrout.apply(RZ(-t * hpqqp / 2), q)
                Qrout.apply(RZ(-t * hpqqp / 2), p)
                Qrout.apply(CNOT, [p, q])
                Qrout.apply(RZ(t * hpqqp / 2), q)
                Qrout.apply(CNOT, [p, q])
    return Qrout


def _number_excitation_operator_jw(hpqrs, t):
    r"""
    This function returns the circuit which corresponds to the time evolution
    ( :math:`e^{-it\hat{O}}`) of the number excitation operator
    ( :math:`a_p^\dagger a_q^\dagger a_q a_r` )  in second quantization after
    performing a Jordan-Wigner transformation.

    Args:
        hpqrs (list): 4D list which contains all the hpqrs terms in the chemical Hamiltonian
            :math:`H = \sum_{pq} h_{pq}a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s`
        t (float): time in the evolution operator
    Returns:
        QRoutine: gates to apply to add the time evolution number excitation operator
    Warning:
        hpqrs may be divided by :math:`\hbar `
    """
    Qrout = QRoutine()
    for p in range(len(hpqrs)):
        # It's equivalent to multiply by identity. It's use to avoid
        # dimensionnal problem when building circuit
        Qrout.apply(PH(0), p)
        for q in range(len(hpqrs)):
            if p != q:
                for r in range(p):
                    if r < q < p:
                        hpqqr = (
                            hpqrs[p][q][q][r]
                            - hpqrs[q][p][q][r]
                            - hpqrs[p][q][r][q]
                            + hpqrs[q][p][r][q]
                        )
                        if hpqqr.real != 0:
                            Qrout.apply(H, r)
                            Qrout.apply(H, p)
                            for k in range(p - q - 1):
                                Qrout.apply(CNOT, [p - k, p - k - 1])
                            Qrout.apply(CNOT, [q + 1, q - 1])
                            for k in range(q - r - 1):
                                Qrout.apply(CNOT, [q - k - 1, q - k - 2])

                            Qrout.apply(RZ((-t * hpqqr.real / 2)), r)

                            Qrout.apply(CNOT, [q, r])

                            Qrout.apply(RZ((t * hpqqr.real / 2)), r)

                            Qrout.apply(CNOT, [q, r])

                            for k in range(q - r - 1):
                                Qrout.apply(CNOT, [r + k + 1, r + k])
                            Qrout.apply(CNOT, [q + 1, q - 1])
                            for k in range(p - q - 1):
                                Qrout.apply(CNOT, [q + k + 2, q + k + 1])

                            Qrout.apply(H, r)
                            Qrout.apply(H, p)

                            Qrout.apply(RX(-pi / 2), r)
                            Qrout.apply(RX(-pi / 2), p)
                            for k in range(p - q - 1):
                                Qrout.apply(CNOT, [p - k, p - k - 1])
                            Qrout.apply(CNOT, [q + 1, q - 1])
                            for k in range(q - r - 1):
                                Qrout.apply(CNOT, [q - k - 1, q - k - 2])

                            Qrout.apply(RZ((-t * hpqqr.real / 2)), r)

                            Qrout.apply(CNOT, [q, r])

                            Qrout.apply(RZ((t * hpqqr.real / 2)), r)

                            Qrout.apply(CNOT, [q, r])

                            for k in range(q - r - 1):
                                Qrout.apply(CNOT, [r + k + 1, r + k])
                            Qrout.apply(CNOT, [q + 1, q - 1])
                            for k in range(p - q - 1):
                                Qrout.apply(CNOT, [q + k + 2, q + k + 1])

                            Qrout.apply(RX(-pi / 2).dag(), r)
                            Qrout.apply(RX(-pi / 2).dag(), p)

                    if (q < r) or (q > p):
                        hpqqr = (
                            hpqrs[p][q][q][r]
                            - hpqrs[q][p][q][r]
                            - hpqrs[p][q][r][q]
                            + hpqrs[q][p][r][q]
                        )

                        if hpqqr.real != 0:
                            Qrout.apply(H, r)
                            Qrout.apply(H, p)
                            for k in range(p - r):
                                Qrout.apply(CNOT, [p - k, p - k - 1])

                            Qrout.apply(RZ((t * hpqqr.real / 2)), r)

                            Qrout.apply(CNOT, [q, r])

                            Qrout.apply(RZ((-t * hpqqr.real / 2)), r)

                            Qrout.apply(CNOT, [q, r])

                            for k in range(p - r):
                                Qrout.apply(CNOT, [r + k + 1, r + k])

                            Qrout.apply(H, r)
                            Qrout.apply(H, p)

                            Qrout.apply(RX(-pi / 2), r)
                            Qrout.apply(RX(-pi / 2), p)
                            for k in range(p - r):
                                Qrout.apply(CNOT, [p - k, p - k - 1])

                            Qrout.apply(RZ((t * hpqqr.real / 2)), r)

                            Qrout.apply(CNOT, [q, r])

                            Qrout.apply(RZ((-t * hpqqr.real / 2)), r)

                            Qrout.apply(CNOT, [q, r])

                            for k in range(p - r):
                                Qrout.apply(CNOT, [r + k + 1, r + k])

                            Qrout.apply(RX(-pi / 2).dag(), r)
                            Qrout.apply(RX(-pi / 2).dag(), p)

    return Qrout


def _double_excitation_operator_jw(hpqrs, t):
    r"""
    This function returns the circuit which corresponds to the time evolution
    ( :math:`e^{-it\hat{O}}`) of the double excitation operator
    ( :math:`a_p^\dagger a_q^\dagger a_r a_s` )  in second quantization after
    performing a Jordan-Wigner transformation.

    Args:
        hpqrs (list): 4D list which contains all the hpqrs terms in the chemical Hamiltonian
            :math:`H = \sum_{pq} h_{pq}a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s`
        t (float): time in the evolution operator
    Returns:
        QRoutine: gates to apply to add the time evolution number  double excitation operator
    Warning:
        hpqrs may be divided by :math:`\hbar `
    """
    Qrout = QRoutine()

    Qrout.apply(PH(0), 0)
    Qrout.apply(PH(0), 1)
    Qrout.apply(PH(0), 2)
    for p in range(3, len(hpqrs)):
        # It's equivalent to multiply by identity. It's use to avoid
        # dimensionnal problem when building circuit
        Qrout.apply(PH(0), p)
        for q in range(p):
            for r in range(q):
                for s in range(r):
                    # Unexplain minus sign but it works so ...
                    hpqrs_number = -(
                        hpqrs[p][q][r][s]
                        - hpqrs[q][p][r][s]
                        - hpqrs[p][q][s][r]
                        + hpqrs[q][p][s][r]
                    )
                    if hpqrs_number.real != 0:
                        for k in range(p - q - 2):
                            Qrout.apply(CNOT, [p - k - 1, p - k - 2])
                        if ((q + 1) != p) and ((s + 1) != r):
                            Qrout.apply(CNOT, [q + 1, r - 1])
                        for k in range(r - s - 2):
                            Qrout.apply(CNOT, [r - k - 1, r - k - 2])
                        if (s + 1) == r:
                            if (q + 1) != p:
                                Qrout.apply(Z.ctrl(), q + 1, s)
                        else:
                            Qrout.apply(Z.ctrl(), s + 1, s)

                        Qrout.apply(H, p)
                        Qrout.apply(H, q)
                        Qrout.apply(H, r)
                        Qrout.apply(H, s)
                        Qrout.apply(CNOT, [p, q])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(RZ((t * hpqrs_number.real / 4)), s)
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(H, s)
                        Qrout.apply(H, r)

                        Qrout.apply(RX(-pi / 2), r)
                        Qrout.apply(RX(-pi / 2), s)
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(RZ((-t * hpqrs_number.real / 4)), s)
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [p, q])
                        Qrout.apply(RX(-pi / 2).dag(), s)
                        Qrout.apply(H, q)

                        Qrout.apply(RX(-pi / 2), q)
                        Qrout.apply(H, s)
                        Qrout.apply(CNOT, [p, q])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(RZ((t * hpqrs_number.real / 4)), s)
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(H, s)
                        Qrout.apply(RX(-pi / 2).dag(), r)

                        Qrout.apply(H, r)
                        Qrout.apply(RX(-pi / 2), s)
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(RZ((t * hpqrs_number.real / 4)), s)
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [p, q])
                        Qrout.apply(RX(-pi / 2).dag(), s)
                        Qrout.apply(H, r)
                        Qrout.apply(RX(-pi / 2).dag(), q)
                        Qrout.apply(H, p)

                        Qrout.apply(RX(-pi / 2), p)
                        Qrout.apply(H, q)
                        Qrout.apply(RX(-pi / 2), r)
                        Qrout.apply(H, s)
                        Qrout.apply(CNOT, [p, q])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(RZ((t * hpqrs_number.real / 4)), s)
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(H, s)
                        Qrout.apply(RX(-pi / 2).dag(), r)

                        Qrout.apply(H, r)
                        Qrout.apply(RX(-pi / 2), s)
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(RZ((t * hpqrs_number.real / 4)), s)
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [p, q])
                        Qrout.apply(RX(-pi / 2).dag(), s)

                        Qrout.apply(H, q)
                        Qrout.apply(RX(-pi / 2), q)
                        Qrout.apply(H, s)
                        Qrout.apply(CNOT, [p, q])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(RZ((-t * hpqrs_number.real / 4)), s)
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(H, s)
                        Qrout.apply(H, r)

                        Qrout.apply(RX(-pi / 2), r)
                        Qrout.apply(RX(-pi / 2), s)
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(RZ((t * hpqrs_number.real / 4)), s)
                        Qrout.apply(CNOT, [r, s])
                        Qrout.apply(CNOT, [q, r])
                        Qrout.apply(CNOT, [p, q])
                        Qrout.apply(RX(-pi / 2).dag(), s)
                        Qrout.apply(RX(-pi / 2).dag(), r)
                        Qrout.apply(RX(-pi / 2).dag(), q)
                        Qrout.apply(RX(-pi / 2).dag(), p)

                        if (s + 1) == r:
                            if (q + 1) != p:
                                Qrout.apply(Z.ctrl(), q + 1, s)
                        else:
                            Qrout.apply(Z.ctrl(), s + 1, s)
                        for k in range(r - s - 2):
                            Qrout.apply(CNOT, [s + k + 2, s + k + 1])
                        if ((q + 1) != p) and ((s + 1) != r):
                            Qrout.apply(CNOT, [q + 1, r - 1])
                        for k in range(p - q - 2):
                            Qrout.apply(CNOT, [q + k + 2, q + k + 1])

    return Qrout
