# -*- coding: utf-8 -*-
"""
Autoderivation tools for Natural gradient descent
"""

import copy
from typing import TYPE_CHECKING, Tuple, List
import numpy as np

from qat.core import Observable, Term
from qat.core.variables import Variable, ArithExpression
from qat.core.circuit_builder.matrix_util import get_predef_generator, np_to_circ
from qat.lang.AQASM import X, Y, Z, Program, AbstractGate
from qat.comm.datamodel.ttypes import GateDefinition, Op, GSyntax, Param

from .expressions import gatedef_to_expr, detect_linear

if TYPE_CHECKING:
    from qat.core import Job
    from qat.lang.AQASM.gates import Gate


def _sanity_check_term(pauli_str: str, nqbits: int) -> Tuple[bool, List[str]]:
    """
    Checks if a multiple-gate observable term corresponds to input number of qubits.

    Note:
        There must only be X, Y or Z single qubits gates, with a total number of nqbits.

    Args:
        pauli_str (string) : The name of the multiple qubit Term.
        nqbits (int) : Number of qubits.

    Returns:
        Tuple[bool, List[str]]:
            If valid, the boolean is True and the list contains the one-letter gates

    """

    valid_1qb_gates = ["X", "Y", "Z"]
    is_multiple_term = True
    gate_list = []

    if len(pauli_str) != nqbits:
        is_multiple_term = False

    else:
        for gate in pauli_str:

            if gate not in valid_1qb_gates:
                is_multiple_term = False
                break

            gate_list.append(gate)

    return (is_multiple_term, gate_list)


def _sanity_check_multirotation(pauli_str: str):
    """
    For a multirotation "RXXXY" gives the generator :math:`e^{-i/2.\vartheta XXXX}`.
    """

    model = {"X": X, "Y": Y, "Z": Z}

    is_valid = True
    single_rots = []

    if pauli_str[0] == "R":
        ind_max = len(pauli_str)

        names = iter(pauli_str)
        _ = next(names)
        ind_c = 1
        while is_valid and ind_c < ind_max:
            g_name = next(names)
            ind_c += 1
            if g_name not in model:
                is_valid = False
            else:
                single_rots.append(model[g_name])
    else:
        is_valid = False

    return (is_valid, single_rots)


def partial_derivatives(
    job: "Job",
    parameter_key: str,
    braket_side: str,
    add_ancilla: bool,
    o_gate: "Gate" = None,
    o_qbits: list = None,
    user_custom_gates: dict = None,
):
    r"""
    This function helps compute the real parts of hermitian products Re(<\partial_i \Psi(vartheta)|\partial_j \Psi(vartheta)>)
    between the partial derivatives of a variational ansatz state. Has to be called twice (once with the bra option, once with the
    ket one) to yield the correct circuit.

    Note:
        Parameters are indexed in their order of appearance in the job quantum circuit.

    Warning:
        :attr:`~qat.core.Job.circuit.gateDic` might be modified throughout the process! (mainly: some entries could be added)

    Args:

        job (:class:`~qat.core.Job`) : A QLM job that prepares a variational ansatz state \Psi(vartheta). Parametric gates should be
            set with some program variables.
        braket_side (str) : ("bra"|"ket"|"none") Whether the partial derivative has to be done for the bra or the ket side of the
            product. If set to None, no differentiation is done.
        parameter_key (str) : The key identifier of the parameter with respect to which the bra/ket part has to be derived.
        add_ancilla (bool) : Set to True if an ancilla qubit has to be added and configured. Set to False if one is already present
            (needs to be on the first register qubit)
        o_gate (:class:`~qat.lang.AQASM.gate.Gate`, optional) : Observable gate part of a hamiltonian. Measurement will be different
            if given.
        o_qbits (list, optional) : The list of qubits the o_gate should be applied to.
        user_custom_gates (dic) :  [Default to None] A dictionary where the user can provide entries that help define custom
            constant gates in the circuit (typically gates that do not appear in AQASM defaults).
            Syntax: {gate_name (string) : GateDefinition (GateDefinition)}. NOT SUPPORTING CUSTOM PARAMETERISED GATES YET.
        custom_gates_generators (dic) : [Default to None] A dictionary that provides the generators of the users custom gates. Will
            be used for auto-differentiation. Must be an operation of the following kind: $U = exp(-i/2.\theta.G_U)$. For such an
            operation, associated with a gate named "U", provide for instance {"U" : "G_U"} where you have added the "G_U"-named
            gate to the user_custom_gates dictionary...

    Returns :

        A tuple list with PARAMETRIC job to evaluate Re(<\partial_i \Psi(vartheta)|\partial_j \Psi(vartheta)>) up to some
        coefficients that are given as the first element of the tuple. It has to be post-processed.

    """

    gate_def_to_add = (
        []
    )  # Used to store the custom gate defs that we will have to add to the circuits gateDic after having added a ancilla qubits
    # At least add the three following GateDefinition objects, because they turn up very often:
    gate_def_to_add.append(GateDefinition(name="C-X", arity=2, nbctrls=1, subgate="X"))
    gate_def_to_add.append(GateDefinition(name="C-Y", arity=2, nbctrls=1, subgate="Y"))
    gate_def_to_add.append(GateDefinition(name="C-Z", arity=2, nbctrls=1, subgate="Z"))

    if not (braket_side == "bra" or braket_side == "ket" or braket_side == "none"):
        raise Exception(f"braket_side argument must be 'bra' or 'ket' (or 'none'), got {braket_side}")

    is_og_valid = False  # Default value
    og_term_list = []
    if o_gate is not None:
        if o_qbits is None:
            raise Exception(
                "o_gate was given. The list of qubits the observable o_gate should be applied to must be provided. The first qubit "
                "is 0 since the ancilla must not have been added yet."
            )
        elif not add_ancilla:
            raise Exception(
                "o_gate was given. Handling observable requires to add an ancilla qubits. No ancilla should be on the circuit yet."
            )
        else:
            is_og_valid, og_term_list = _sanity_check_term(o_gate.name, len(o_qbits))

            if not is_og_valid:
                if f"C-{o_gate.name}" not in job.circuit.gateDic:
                    ogname = o_gate.name
                    gate_def_to_add.append(GateDefinition(name=f"C-{ogname}", arity=2, nbctrls=1, subgate=ogname))

    # List to store the differentiable gates generators
    p_list = []

    # A list to store post process coefficients that arise from differentiation
    coeff_list = []

    if braket_side == "none":

        # Adding one dummy Pauli operator
        raise Exception("No differentiation side has been given. Please set it to 'bra' or 'ket'.")

    # braket_side is "bra" or "ket"
    else:  

        # Doing a first circuit review to get the Pauli generator and the number of gates for each parameter
        for (op_op, (ind, (opname, params, qubits))) in zip(job.circuit.ops, enumerate(job.circuit.iterate_simple())):
            op_key = op_op.gate

            if len(params) == 0:
                # This is not a parameterized gate, so we skip it
                continue

            elif len(params) == 1:

                var_test1 = isinstance(params[0], Variable)
                arith_test1 = isinstance(params[0], ArithExpression)

                if not (var_test1 or arith_test1):
                    # Case of a binded gate
                    continue
                param_key = next(iter(params[0].get_variables()))

                if param_key == parameter_key:

                    if opname[0] == "R":

                        coeff = -1j / 2  # Only exp

                        arith_expr = gatedef_to_expr(job.circuit.gateDic[op_key])

                        expr_coeff = detect_linear(arith_expr)

                        if expr_coeff is not None:

                            if np.isreal(expr_coeff):
                                coeff *= np.real(expr_coeff)

                            else:
                                raise Exception("For now, only supporting real coefficient in front of parameters!")

                        else:
                            raise Exception(
                                f"The gate {opname} (key: {op_key}) doesn't contain a valid linear arithmetic expression!"
                            )

                        (is_known_multirotation, pauli_gates_list) = _sanity_check_multirotation(opname)

                        if not is_known_multirotation:

                            if len(opname) > 2:
                                pauli_prod = get_predef_generator()[opname[1]]
                                for pauli in opname[2:]:
                                    pauli_prod = np.kron(pauli_prod, get_predef_generator()[pauli])

                            pauli_gate = AbstractGate(
                                str(opname[1:]), [], len(qubits), matrix_generator=lambda pauli_prod=pauli_prod: pauli_prod
                            )()

                            # Add the control version of the gate
                            gate_def_to_add.append(
                                GateDefinition(name=f"C-{(opname[1:])}", arity=2, nbctrls=1, subgate=(opname[1:]))
                            )
                            pauli_gates_list = [pauli_gate]

                        coeff_list.append(coeff)

                        p_list.append((ind, pauli_gates_list, qubits))

                    else:
                        raise Exception("Not supporting parameterized gates different from rotations yet.")

            else:
                raise Exception(f"Not supporting gates with multiple parameters yet, got {params}")

    jobs_list = {
        ind_job: [0.0, copy.copy(job)] for ind_job in range(len(p_list))
    } # the float is to store the coefficient that comes from differentiation

    # To be able to modify circuits one by one, me must copy them
    for ind_job in jobs_list:
        jobs_list[ind_job][1].observable = copy.copy(jobs_list[ind_job][1].observable)
        jobs_list[ind_job][1].circuit = copy.copy(jobs_list[ind_job][1].circuit)
        jobs_list[ind_job][1].circuit.ops = copy.copy(jobs_list[ind_job][1].circuit.ops)

    ind_job = 0

    for (process_coeff, (indi, pauli_gates_list, p_qbits)) in zip(coeff_list, p_list):

        right_shift = 0

        if add_ancilla:

            prog = Program()
            prog.qalloc(1)
            jobs_list[ind_job][1].circuit = (
                prog.to_circ() * jobs_list[ind_job][1].circuit
            )  # Add an ancilla at the top of the circuit

        # The previous operation will have erased custom gates defs, we add them now from what we have stored
        for gate_def_ in gate_def_to_add:
            jobs_list[ind_job][1].circuit.gateDic[gate_def_.name] = gate_def_

        # Add user custom gates
        if user_custom_gates is not None:
            for gate_c_key in user_custom_gates:
                jobs_list[ind_job][1].circuit.gateDic[gate_c_key] = user_custom_gates[gate_c_key]

        if add_ancilla:

            jobs_list[ind_job][1].circuit.ops.insert(0, Op(type=0, gate="H", qbits=[0]))
            right_shift += 1

        # Going through the original circuit to copy the gates and add generators beforehand if needed
        update_coeff = 1.0 + 0.0 * 1j

        # As we have already copied the circuit, we do not need to iterate the circuit to copy it, but just to insert the generators
        # gates at the right positions
        if len(pauli_gates_list) == 1:

            temp_qbits1 = [0]  # the ancilla index for the control operation
            for qb in p_qbits:

                if add_ancilla:
                    temp_qbits1.append(qb + 1)

                else:
                    temp_qbits1.append(qb)

            temp_qbits = [temp_qbits1]

        else:

            temp_qbits = []
            for qb in p_qbits:

                if add_ancilla:
                    temp_qbits.append([0, qb + 1])

                else:
                    temp_qbits.append([0, qb])

        active_qbits = iter(temp_qbits)
        if braket_side == "bra":

            # The operator is daggered, so we have to perform a X shit on the ancilla qubit
            jobs_list[ind_job][1].circuit.ops.insert(right_shift + indi, Op(type=0, gate="X", qbits=[0]))
            ind_insert = 1

            for pauli_gate in pauli_gates_list:

                jobs_list[ind_job][1].circuit.ops.insert(
                    right_shift + indi + ind_insert, Op(type=0, gate=f"C-{pauli_gate.name}", qbits=next(active_qbits))
                )  # Wether an ancilla has been inserted or not, qubits index have already been shifted.
                ind_insert += 1

            jobs_list[ind_job][1].circuit.ops.insert(right_shift + indi + ind_insert, Op(type=0, gate="X", qbits=[0]))
            update_coeff = np.conj(process_coeff)  # For bra-side, we conjugate the coefficient (typically, -i/2 |-> +i/2)

        else:

            # No X shift needed here
            ind_insert = 0
            for pauli_gate in pauli_gates_list:

                jobs_list[ind_job][1].circuit.ops.insert(
                    right_shift + indi + ind_insert, Op(type=0, gate=f"C-{pauli_gate.name}", qbits=next(active_qbits))
                )  # Wether an ancilla has been inserted or not, qubits index have already been shifted.
                ind_insert += 1

            update_coeff = process_coeff

        # For the hamiltonian gate
        if o_gate is not None:

            # Assume add_ancilla is true
            if is_og_valid:

                # We use the list (ex. ["X", "X", "X"]
                # Assume the XXYYZ (eg) term is for different qubits
                it_qb = iter(o_qbits)
                for g_name in og_term_list:

                    # g_name is expected to be a list of 'X', 'Y' or 'Z'
                    jobs_list[ind_job][1].circuit.ops.append(Op(type=0, gate=f"C-{g_name}", qbits=[0, 1 + next(it_qb)]))

            else:
                temp_qbits = [0]  # the ancilla index for the control operation
                for qb in o_qbits:
                    temp_qbits.append(qb + 1)  # assume add_ancilla is true

                jobs_list[ind_job][1].circuit.ops.append(Op(type=0, gate=f"C-{o_gate.name}", qbits=temp_qbits))

        if add_ancilla:

            jobs_list[ind_job][1].circuit.ops.append(
                Op(type=0, gate="H", qbits=[0])
            )  # Change ancilla bases for measurement along the X axis

            if o_gate is not None:

                # In the case we have added an o_gate
                rx_pi_o2 = 1 / np.sqrt(2) * np.array([[1.0 + 0.0 * 1j, 0.0 - 1j], [0.0 - 1j, 1.0 + 0.0 * 1j]])
                par = Param(type=1, double_p=np.pi / 2)
                jobs_list[ind_job][1].circuit.gateDic["_MeasShift"] = GateDefinition(
                    name="_MeasShift", arity=1, matrix=np_to_circ(rx_pi_o2), syntax=GSyntax(name="RX", parameters=[par])
                )

                jobs_list[ind_job][1].circuit.ops.append(Op(type=0, gate="_MeasShift", qbits=[0]))

        jobs_list[ind_job][1].observable = Observable(jobs_list[ind_job][1].circuit.nbqbits, pauli_terms=[Term(1.0, "Z", [0])])
        jobs_list[ind_job][0] = update_coeff
        ind_job += 1

    return jobs_list.values()


def _product_param_real_part(job, bra_key, ket_key, user_custom_gates=None):
    r"""
    This function will call partial_derivatives in the right fashion to get the parameterized circuits to compute
    :math:`Re<\partial_bra \Psi|\partial_ket \Psi>`.

    Args:

        job (:class:`qat.core.Job`) : Job.
        bra_key (str) : key identifier. Set to "no_differentiation" to do nothing with the bra side.
        ket_key (str) : key identifier. Set to "no_differentiation" to do nothing with the ket side.

    Returns:
        Parameterized Job.

    """

    jobs_list = []

    if bra_key == "no_differentiation" and ket_key == "no_differentiation":
        jobs_list.append(
            partial_derivatives(
                job,
                parameter_key="none",
                braket_side="none",
                add_ancilla=True,
                user_custom_gates=user_custom_gates,
            )
        )

    elif bra_key == "no_differentiation":

        ket_list = partial_derivatives(
            job,
            parameter_key=ket_key,
            braket_side="ket",
            add_ancilla=True,
            user_custom_gates=user_custom_gates,
        )
        for (ket_coeff, new_job) in ket_list:
            jobs_list.append((ket_coeff, new_job))

    elif ket_key == "no_differentiation":

        bra_list = ket_list = partial_derivatives(
            job,
            parameter_key=bra_key,
            braket_side="bra",
            add_ancilla=True,
            user_custom_gates=user_custom_gates,
        )
        for (bra_coeff, new_job) in bra_list:
            jobs_list.append((bra_coeff, new_job))

    else:  # differentiate on both sides

        bra_list = partial_derivatives(
            job,
            parameter_key=bra_key,
            braket_side="bra",
            add_ancilla=True,
            user_custom_gates=user_custom_gates,
        )

        for (bra_coeff, bra_job) in bra_list:

            p_ket_list = []
            p_ket_list = partial_derivatives(
                bra_job,
                parameter_key=ket_key,
                braket_side="ket",
                add_ancilla=False,
                user_custom_gates=user_custom_gates,
            )

            for (ket_coeff, new_job) in p_ket_list:

                jobs_list.append((bra_coeff * ket_coeff, new_job))

    return jobs_list


def hamiltonian_gradient_param_imag_part(job, hamiltonian, partial_key, user_custom_gates=None):
    r"""
    Returns the (coefficients, jobs) list to compute $\partial_{partial_key} <\Psi | H | \Psi> where Psi is the ansatz prepared
    according to the given job.

    Note : For now, this function will only work if the ansatz job parameterized gates are of the shape e^(-i\theta /2) !

    Args :

        job (Job) : The entry job. Its circuit indicates how to prepare parameterized ansatz states.
        hamiltonian (Observable) : The observable corresponds to the hamiltonian corresponding to the energy of interest.
        partial_key (string) : Key identifier with respect to which we want to compute the partial differentiation

    Returns :

        The (coefficients, jobs) list to compute $\partial_{partial_key}. The jobs are PARAMETRIC.

    """

    o_list = []
    h_coeff_list = []

    for _, term in enumerate(hamiltonian.terms):

        # First, construct operator corresponding to O_j
        is_og_valid, _ = _sanity_check_term(term.op, len(term.qbits))

        if is_og_valid:
            op_gate = AbstractGate(term.op, [], len(term.qbits))()  # We will deal with the matrix later in the core function

        else:

            op_gate_matrix = get_predef_generator()[term.op[0]]
            if len(term.op) > 1:

                # Assumes term.qbits is sorted in ascending order
                for op in term.op[1:]:
                    op_gate_matrix = np.kron(op_gate_matrix, get_predef_generator()[op])

            op_gate = AbstractGate(
                f"{(term.op)}", [], len(term.qbits), matrix_generator=lambda op_gate_matrix=op_gate_matrix: op_gate_matrix
            )()

        o_list.append((op_gate, term.qbits))
        h_coeff_list.append(term.coeff)

    # Now construct circuits
    jobs_list = []
    for indj, (o_gate, o_qbits) in enumerate(o_list):

        # Build a circuit with some of the Hamiltonian gates (issues with the xx gate for instance)
        first_jobs_list = partial_derivatives(
            job,
            parameter_key=partial_key,
            braket_side="ket",
            add_ancilla=True,
            o_gate=o_gate,
            o_qbits=o_qbits,
            user_custom_gates=user_custom_gates,
        )

        for (op_coeff, op_job) in first_jobs_list:

            corr_coeff = op_coeff / (-1j / 2)
            if np.isreal(corr_coeff):

                jobs_list.append(
                    (-h_coeff_list[indj] * np.real(corr_coeff), op_job)
                )  # minus sign is because we compute (-1)*Im(<O|U(\theta)^\dagger.O_j.V_i(\theta)|0> with the hadamard test circuit
                # with the "ket" side

            else:
                raise Exception(
                    rf"For now, this function will only work if the ansatz job parameterizedgates are of the shape e^(-i \lambda "
                    r"\\theta \\sigma /2) ! Instead of -i/2*\lambda, got a gen coeff : "
                    f"{np.real(op_coeff)}+{np.imag(op_coeff)} i"
                )

    return jobs_list


def auto_differentiation_gradient_dictionary(job, hamiltonian, parameters_dict, user_custom_gates=None):
    """
    Computes the jobs lists dictionary to evaluate gradient.

    Returns:
        The jobs are PARAMETRIC.
    """

    gradient_jobs_dict = {}
    for var_key in parameters_dict:
        gradient_jobs_dict[var_key] = hamiltonian_gradient_param_imag_part(
            job, hamiltonian, var_key, user_custom_gates=user_custom_gates
        )

    return gradient_jobs_dict


def auto_differentiation_qfim_dictionaries(job, nb_parameters, parameters_index, user_custom_gates=None):
    """
    Computes the three jobs dictionaries used for QFIM computation

    Args :
        job (Job) : The job we aim at differentiating
        nb_parameters (int) : The number of parameters keys
        parameters_index (dict) : A dictionary to link parameters names to their index
        user_custom_gates (dict) :  [Default to None] A dictionary where the user can provide entries that help define custom
            constant gates in the circuit (typically gates that do not appear in AQASM defaults).
            Syntax: {gate_name (string) : GateDefinition (GateDefinition)}. NOT SUPPORTING CUSTOM PARAMETERISED GATES YET.

    Returns :
        Tuple[dict, dict, dict]: Variational jobs.

    """

    qfim_dkdl_jobs_dict = {}
    qfim_dkpsi_jobs_dict = {}
    qfim_psidl_jobs_dict = {}

    for k in range(nb_parameters):

        param_k = parameters_index[k]
        qfim_dkpsi_jobs_dict[param_k] = _product_param_real_part(
            job, bra_key=param_k, ket_key="no_differentiation", user_custom_gates=user_custom_gates
        )
        qfim_psidl_jobs_dict[param_k] = _product_param_real_part(
            job, bra_key="no_differentiation", ket_key=param_k, user_custom_gates=user_custom_gates
        )

        for idx in range(nb_parameters):

            param_l = parameters_index[idx]
            qfim_dkdl_jobs_dict[param_k + ";" + param_l] = _product_param_real_part(
                job, bra_key=param_k, ket_key=param_l, user_custom_gates=user_custom_gates
            )

    return (qfim_dkdl_jobs_dict, qfim_dkpsi_jobs_dict, qfim_psidl_jobs_dict)
