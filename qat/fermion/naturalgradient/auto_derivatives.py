"""
Auto derivatives
"""

import numpy as np
import copy

import logging

from qat.core import Observable, Term, Batch, BatchResult, Result, Circuit, default_gate_set
from qat.core.variables import Variable, ArithExpression
from qat.core.circuit_builder.matrix_util import get_predef_generator
from qat.plugins import AbstractPlugin
from qat.lang.AQASM import H, X, Y, Z, RX, RY, RZ, CNOT, CSIGN, QRoutine, Program, AbstractGate  # Quantum gates we want to use
from qat.comm.datamodel.ttypes import GateDefinition, Op, GSyntax, Param
from qat.core.circuit_builder.matrix_util import np_to_circ

logging.basicConfig(level=logging.WARNING)

# Temporary for XX gate handling
from .custom_gate_set import get_custom_gate_set_1

# For generic arithmetic expressions handling
from .expressions import gatedef_to_expr, improved_detect_linear

##############################################################################################
# Useful function to copy any gate from an existing circuit and avoid Gate Signature issues. #
##############################################################################################

# NOT USED ANYMORE
_CORRESP = {
    "C": "ctrl",
    "D": "dag",
    "T": "transp",
    "CO": "conj",
}


def reconstruct_gate(name, parameters, gate_set):
    split_name = name.split("-")
    true_name = split_name[-1]
    operators = split_name[:-1]
    if parameters is None:
        gate = gate_set[true_name]
    else:
        gate = gate_set[true_name](*parameters)
    for operator in reversed(operators):
        gate = getattr(gate, _CORRESP[operator])()
    return gate


def fix_gate_set(gate_set):
    """
    Fix the gate set by replace the GateSignatures object by proper AbstractGates.
    This has to be done only for the gates in the default gate set.
    """
    for signature in gate_set.gate_signatures.values():
        signature.__class__ = AbstractGate


def split_check_term(term_name, nb_qbits):
    """
    Checks if a multiple-gate observable term is valid:
    XXX
    YYX
    ...
    There must only be X, Y or Z single qubits gates, with a total number of nb_qbits

    Args:
    term_name (string) : the name of the multiple qubit Term
    nb_qbits (int) : the number of qubits it should correspond to

    Returns:
    A tuple (is_multiple_term, gate_list) (boolean, string list). If valid, the boolean is True and the list contains the one-letter gates
    """

    valid_oneqb_gates = ["X", "Y", "Z"]
    is_multiple_term = True
    gate_list = []

    if len(term_name) != nb_qbits:
        is_multiple_term = False
    else:
        for gate in term_name:
            if not (gate in valid_oneqb_gates):
                is_multiple_term = False
                break
            else:
                gate_list.append(gate)

    return (is_multiple_term, gate_list)


def split_multirotation(rot_name):
    """
    For a multirotation "RXXXY" gives the generator
    $e^{-i/2.\vartheta XXXX}

    """
    model = {"X": X, "Y": Y, "Z": Z}

    is_known_multirotation = True
    single_rots = []

    if rot_name[0] == "R":
        ind_max = len(rot_name)

        names = iter(rot_name)
        _ = next(names)
        ind_c = 1
        while is_known_multirotation and ind_c < ind_max:
            g_name = next(names)
            ind_c += 1
            if not (g_name in model):
                is_known_multirotation = False
            else:
                single_rots.append(model[g_name])
    else:
        is_known_multirotation = False

    return (is_known_multirotation, single_rots)


# ***********************************************************************************************************************************************************************#

#################################################
# THE CORE FUNCTION FOR AUTO-DIFFERENTIATION    #
#################################################


def partial_derivatives(
    job,
    parameter_key,
    braket_side,
    add_ancilla,
    O_gate=None,
    O_qbits=None,
    user_custom_gates=None,
    custom_gates_generators=None,
    do_print_status=False,
):
    """
    This function should help compute the real parts of hermitian products Re(<\partial_i \Psi(vartheta)|\partial_j \Psi(vartheta)>)
    between the partial derivatives of a variational ansatz state. Has to be called twice (once with the bra option, once with the ket one) to yield the correct circuit.

                                partial_derivatives(ijob, ket)                 partial_derivatives(pjob, bra)
    Initial ansatz job -----> partially adapted job             ----------->  final job able to compute Re(<..|..>)

    ### Parameters are indexed in their order of appearance in the job quantum circuit. ###
    Using one ancilla qubit.
    ### Beware :  job.circuit.gateDic might be modified throughout the process! (mainly: some entries could be added)

    Args :

        job (qat.core.Job) : A QLM job that prepares a variational ansatz state \Psi(vartheta). Parametric gates sould be set with some program variables.
        braket_side (string) : ("bra"|"ket"|"none") Wether the partial derivative has to be done for the bra or the ket side of the product. No differentiation if set to "none".
        parameter_key (string) : The key identifier of the parameter with respect to which the bra/ket part has to be derived.
        add_ancilla (boolean) : True if an ancilla qubit has to be added and configured, False if one is already present (needs to be on reg[0])

        O_gate (gate) : [Optional] Observable gate part of a hamiltonian. Measurement will be different if given.
        O_qbits (list) : [Optional] The list of qbits the O_gate should be applied to.

        user_custom_gates (dic) :  [Default to None] A dictionary where the user can provide entries that help define custom constant gates in the circuit (typically gates that do not appear in AQASM defaults). Syntax: {gate_name (string) : GateDefinition (GateDefinition)}. NOT SUPPORTING CUSTOM PARAMETERISED GATES YET.
        custom_gates_generators (dic) : [Default to None] A dictionary that provides the generators of the users custom gates. Will be used for autodifferentiation. Must be an operation of the following kind: $U = exp(-i/2.\theta.G_U)$. For such an operation, associated with a gate named "U", provide for instance {"U" : "G_U"} where you have added the "G_U"-named gate to the user_custom_gates dictionary...

    Returns :

        A tuple list with PARAMETRIC job to evaluate Re(<\partial_i \Psi(vartheta)|\partial_j \Psi(vartheta)>) up to some coefficients that are given as the first element of the tuple.
        It has to be post-processed.

    """

    # custom_gates_dict = {"H": H, "X": X, "Y": Y, "Z": Z, "C-X": CNOT, "C-Y": Y.ctrl(), "C-Z": CSIGN, "RX": RX, "RY": RY, "RZ": RZ, "CNOT": CNOT, "CSIGN":CSIGN}
    gate_def_to_add = (
        []
    )  # Will be used to store the custom gate defs that we will have to add to the circuits gateDic after having added a ancilla qubits (this operation resets the custom gates names)...
    # At least add the three following GateDefinition objects, because they turn up very often:
    gate_def_to_add.append(GateDefinition(name="C-X", arity=2, nbctrls=1, subgate="X"))
    gate_def_to_add.append(GateDefinition(name="C-Y", arity=2, nbctrls=1, subgate="Y"))
    gate_def_to_add.append(GateDefinition(name="C-Z", arity=2, nbctrls=1, subgate="Z"))

    if not (braket_side == "bra" or braket_side == "ket" or braket_side == "none"):
        raise Exception("braket_side argument must be 'bra' or 'ket' (or 'none'), got %s" % braket_side)

    is_og_valid = False  # Default value
    og_term_list = []
    if not (O_gate is None):
        if O_qbits is None:
            raise Exception(
                "O_gate was given. The list of qbits the observable O_gate should be applied to must be provided. The first qubit is 0 since the ancilla must not have been added yet."
            )
        elif not (add_ancilla):
            raise Exception(
                "O_gate was given. Handling observable requires to add an ancilla qubits. No ancilla should be on the circuit yet."
            )
        else:
            is_og_valid, og_term_list = split_check_term(O_gate.name, len(O_qbits))

            if not is_og_valid:
                if not ("C-%s" % O_gate.name in job.circuit.gateDic):
                    ogname = O_gate.name
                    # job.circuit.gateDic["C-%s"%ogname] = GateDefinition(name="C-%s"%ogname, arity=2, nbctrls=1, subgate=ogname)
                    gate_def_to_add.append(GateDefinition(name="C-%s" % ogname, arity=2, nbctrls=1, subgate=ogname))

    parameters_dict = job.circuit.var_dic  # The program variables used in the circuit

    P_list = []  # A list to store the differientiable gates generators
    coeff_list = []  # A list to store post process coefficients that arise from differentiation

    logging.info("Initial gates : ")
    if do_print_status:
        print("#####################################################")
        print("Initial gates : ")

    if braket_side == "none":
        # Adding one dummy Pauli operator
        # coeff_list.append(1.+0*1j)
        # P_list.append((-1, None, []))
        raise Exception("No differentiation side has been given. Please set it to 'bra' or 'ket'.")

    else:  # braket_side is "bra" or "ket"
        # Doing a first circuit review to get the Pauli generator and the number of gates for each parameter
        # for ind, (opname, params, qbits) in enumerate(job.circuit.iterate_simple()):
        for (op_op, (ind, (opname, params, qbits))) in zip(job.circuit.ops, enumerate(job.circuit.iterate_simple())):
            op_key = op_op.gate
            if do_print_status:
                print("OPNAME : ", opname)
                print(ind, opname, params, qbits)
            logging.info(str(ind) + " " + str(opname) + " " + str(params) + " " + str(qbits))
            if len(params) == 0:
                continue  # This is not a parameterized gate, so we skip it
            elif len(params) == 1:
                var_test1 = isinstance(params[0], Variable)
                arith_test1 = isinstance(params[0], ArithExpression)
                if not (var_test1 or arith_test1):
                    # Case of an binded gate
                    continue
                param_key = next(iter(params[0].get_variables()))

                if param_key == parameter_key:

                    if opname[0] == "R":

                        # Eventuellement modifier pour prendre les générateurs donnés ?

                        coeff = -1j / 2  # Only exp

                        arith_expr = gatedef_to_expr(job.circuit.gateDic[op_key])

                        (is_expr_valid, expr_coeff) = improved_detect_linear(arith_expr)

                        if is_expr_valid:
                            if np.isreal(expr_coeff):
                                coeff *= np.real(expr_coeff)
                            else:
                                raise Exception("For now, only supporting real coefficient in front of parameters!")
                                # Could be improved in shifting the ancilla qbits to (|0>+e^(i\alpha)|1>)/\sqrt(2)) at the beginning of the Hadamard tests if the operation is e^{-i/2.\lambda.\vartheta.\sigma_gen} where \lambda = |\lamdba|.e^{i\alpha}...
                        else:
                            raise Exception(
                                "The gate %s (key: %s) doesn't contain a valid linear arithmetic expression!" % (opname, op_key)
                            )

                        (is_known_multirotation, pauli_gates_list) = split_multirotation(opname)

                        if not (is_known_multirotation):

                            # For multirotations ???
                            if len(opname) > 2:
                                pauli_prod = get_predef_generator()[opname[1]]
                                for pauli in opname[2:]:
                                    pauli_prod = np.kron(pauli_prod, get_predef_generator()[pauli])

                            pauli_gate = AbstractGate(
                                "%s" % (opname[1:]), [], len(qbits), matrix_generator=lambda pauli_prod=pauli_prod: pauli_prod
                            )()

                            # Add the control version of the gate
                            # print("ADD CTRL GATE C-%s"%(opname[1:]))
                            # job.circuit.gateDic["C-%s"%(opname[1:])] = GateDefinition(name="C-%s"%(opname[1:]), arity=2, nbctrls=1, subgate=(opname[1:]))
                            gate_def_to_add.append(
                                GateDefinition(name="C-%s" % (opname[1:]), arity=2, nbctrls=1, subgate=(opname[1:]))
                            )
                            pauli_gates_list = [pauli_gate]

                        coeff_list.append(coeff)

                        P_list.append((ind, pauli_gates_list, qbits))
                        # P_list.append((ind, pauli_gate, qbits)) #  Keep track of how many and what gates are concerned with the differentiation, and what the generator is

                    else:
                        logging.error("Not supporting parameterised gates different from rotations yet.")
                        raise Exception("Not supporting parameterised gates different from rotations yet.")
                        """
                        if custom_gates_generators is None:
                            logging.error("No custom gates generators have been provided!")
                            raise Exception("No custom gates generators have been provided!")
                        else:
                            if opname in custom_gates_generators:
                                
                                try:
                                    
                                    if custom_gates_generators[opname] in job.circuit.gateDic:
                                        print("There is already a gate named %s in the circuit gateDic. It won't be updated..."%custom_gates_generators[opname])
                                    else:
                                        #job.circuit.gateDic[custom_gates_generators[opname]] = user_custom_gates[custom_gates_generators[opname]]
                                        gate_def_to_add.append(user_custom_gates[custom_gates_generators[opname]])
                                    
                                    
                                    
                                except:
                                    logging.error("No generator gate specs have been provided for operation %s, generator %s in the user_custom_gates dictionary !"%(opname, custom_gates_generators[opname]))
                                    raise Exception("No generator gate specs been provided for operation %s, generator %s in the user_custom_gates dictionary !"%(opname, custom_gates_generators[opname]))
                                
                            else:
                                logging.error("No generator has been provided for operation %s"%opname)
                                raise Exception("No generator has been provided for operation %s"%opname) #For now we only handle Pauli generators, hence rotations
                        """

            else:  # if len(params) > 1:
                logging.error("Not supporting gates with multiple parameters yet, got %s" % params)
                raise Exception("Not supporting gates with multiple parameters yet, got %s" % params)

    if do_print_status and braket_side != "none":
        print("theta dict : ", parameters_dict)
        print("P_list : ", P_list)
        print()
        print(
            "%i gates depend on the chosen %s parameter. %i jobs will eventually be delivered."
            % (len(P_list), parameter_key, len(P_list))
        )
        print()
    logging.info("P_list : " + str(P_list))

    # jobs_list = [] # Its final size will be len(P_list)
    # To speed-up preparation, when only do some shallow copy of the jobs and avoid a full deep-copy
    jobs_list = {
        ind_job: [0.0, copy.copy(job)] for ind_job in range(len(P_list))
    }  # the float is to store the coefficient that comes from differentiation

    # To be able to modify circuits one by one, me must copy them...
    for ind_job in jobs_list:
        jobs_list[ind_job][1].observable = copy.copy(jobs_list[ind_job][1].observable)
        jobs_list[ind_job][1].circuit = copy.copy(jobs_list[ind_job][1].circuit)
        jobs_list[ind_job][1].circuit.ops = copy.copy(jobs_list[ind_job][1].circuit.ops)

    ind_job = 0

    # for (process_coeff, (indi, pauli_gate, P_qbits)) in zip(coeff_list, P_list):
    for (process_coeff, (indi, pauli_gates_list, P_qbits)) in zip(coeff_list, P_list):

        right_shift = 0

        if do_print_status:
            print()
            print("------ New job to be computed ------")

        # New program with new variational parameters
        # prog = Program()

        if add_ancilla:
            # reg = prog.qalloc(job.circuit.nbqbits + 1) # We will need an ancilla qubit
            prog = Program()
            prog.qalloc(1)
            jobs_list[ind_job][1].circuit = (
                prog.to_circ() * jobs_list[ind_job][1].circuit
            )  # Add an ancilla at the top of the circuit

        # The previous operation will have erased custom gates defs, we add them now from what we have stored

        for gate_def_ in gate_def_to_add:
            jobs_list[ind_job][1].circuit.gateDic[gate_def_.name] = gate_def_

        # Add user custom gates
        if not (user_custom_gates is None):
            for gate_c_key in user_custom_gates:
                jobs_list[ind_job][1].circuit.gateDic[gate_c_key] = user_custom_gates[gate_c_key]
        # else:
        # reg = prog.qalloc(job.circuit.nbqbits) # The ancilla qubit has already been added, and is considered treated
        # From now, the ancilla qubit will reg[0]

        # var_dict = {}
        # for var_key in parameters_dict:
        #    var_dict[var_key] = prog.new_var(float, var_key)

        # if do_print_status:
        #    print("New var_dict", var_dict)

        if add_ancilla:
            if do_print_status:
                print("Applying first H gate")
            logging.info("Applying first H gate")
            # prog.apply(H, reg[0]) # Preparing the ancilla in a superposed state
            jobs_list[ind_job][1].circuit.ops.insert(0, Op(type=0, gate="H", qbits=[0]))
            right_shift += 1

        # Going through the original circuit to copy the gates and add generators beforehand if needed

        update_coeff = 1.0 + 0.0 * 1j

        # As we have already copied the circuit, we do not need to iterate the circuit to copy it, but just to insert the generators gates at the right positions

        if do_print_status:
            print("Pauli %s index : " % [pg.name for pg in pauli_gates_list], indi)
        logging.info("Pauli %s index : " % [pg.name for pg in pauli_gates_list] + str(indi))

        if do_print_status:
            print("params : ", params)

        # print("PAULI NAMED ?: ", pauli_gate.name)
        # ctrl_pauli = pauli_gate.ctrl()
        # print("CTRL PAULI NAMED ?: ", ctrl_pauli)

        # ctrl_pauli = custom_gates_dict["C-%s"%pauli_gate.name]
        # if do_print_status:
        # print("NEW CTRL GATE: ", ctrl_pauli.name)

        if len(pauli_gates_list) == 1:
            temp_qbits1 = [0]  # the ancilla index for the control operation
            for qb in P_qbits:
                if add_ancilla:
                    temp_qbits1.append(qb + 1)
                else:
                    temp_qbits1.append(qb)
            temp_qbits = [temp_qbits1]
        else:
            temp_qbits = []
            for qb in P_qbits:
                if add_ancilla:
                    temp_qbits.append([0, qb + 1])
                else:
                    temp_qbits.append([0, qb])

        active_qbits = iter(temp_qbits)
        if braket_side == "bra":
            # The operator is daggered, so we have to perform a X shit on the ancilla qubit
            if do_print_status:
                print("   Applying ctrl generator gate (%s) with an ancilla shift : " % [pg.name for pg in pauli_gates_list])
            logging.info("   Applying ctrl generator gate (%s) with an ancilla shift : " % [pg.name for pg in pauli_gates_list])

            jobs_list[ind_job][1].circuit.ops.insert(right_shift + indi, Op(type=0, gate="X", qbits=[0]))
            ind_insert = 1
            for pauli_gate in pauli_gates_list:

                jobs_list[ind_job][1].circuit.ops.insert(
                    right_shift + indi + ind_insert, Op(type=0, gate="C-%s" % pauli_gate.name, qbits=next(active_qbits))
                )  # Wether an ancilla has been inserted or not, qbits index have already been shifted.
                ind_insert += 1

            jobs_list[ind_job][1].circuit.ops.insert(right_shift + indi + ind_insert, Op(type=0, gate="X", qbits=[0]))
            update_coeff = np.conj(process_coeff)  # For bra-side, we conjugate the coefficient (typically, -i/2 |-> +i/2)

        else:  # braket_side == "ket"
            # No X shift needed here
            ind_insert = 0
            for pauli_gate in pauli_gates_list:
                if do_print_status:
                    print("   Applying ctrl generator gate (%s) with no ancilla shift : " % pauli_gate.name, pauli_gate)
                logging.info("   Applying ctrl generator gate (%s) with no ancilla shift : " % pauli_gate.name + str(pauli_gate))

                jobs_list[ind_job][1].circuit.ops.insert(
                    right_shift + indi + ind_insert, Op(type=0, gate="C-%s" % pauli_gate.name, qbits=next(active_qbits))
                )  # Wether an ancilla has been inserted or not, qbits index have already been shifted.
                ind_insert += 1

            update_coeff = process_coeff

        # For the hamiltonian gate...
        if not (O_gate is None):

            if do_print_status:
                print("Applying the observable gate")

            # assume add_ancilla is true...
            # prog.apply(O_gate.ctrl(), reg[0], [reg[qb+1] for qb in O_qbits])
            # octrl = O_gate.ctrl()
            if is_og_valid:
                # We use the list (ex. ["X", "X", "X"]
                # Assume the XXYYZ (eg) term is for different qbits...
                it_qb = iter(O_qbits)
                for g_name in og_term_list:
                    # g_name is expected to be something like 'X', 'Y' or 'Z'
                    jobs_list[ind_job][1].circuit.ops.append(Op(type=0, gate="C-%s" % g_name, qbits=[0, 1 + next(it_qb)]))
            else:
                temp_qbits = [0]  # the ancilla index for the control operation
                for qb in O_qbits:
                    temp_qbits.append(qb + 1)  # assume add_ancilla is true

                jobs_list[ind_job][1].circuit.ops.append(Op(type=0, gate="C-%s" % O_gate.name, qbits=temp_qbits))

        if add_ancilla:
            if do_print_status:
                print("Applying last H gate to change bases.")

            logging.info("Applying last H gate to change bases.")
            jobs_list[ind_job][1].circuit.ops.append(
                Op(type=0, gate="H", qbits=[0])
            )  # Change ancilla bases for measurement along the X axis
            if not (O_gate is None):
                # In the case we have added an O_gate
                if do_print_status:
                    print("Applying measurement shift for O_gate.")
                RXpiO2 = 1 / np.sqrt(2) * np.array([[1.0 + 0.0 * 1j, 0.0 - 1j], [0.0 - 1j, 1.0 + 0.0 * 1j]])
                par = Param(type=1, double_p=np.pi / 2)
                jobs_list[ind_job][1].circuit.gateDic["_MeasShift"] = GateDefinition(
                    name="_MeasShift", arity=1, matrix=np_to_circ(RXpiO2), syntax=GSyntax(name="RX", parameters=[par])
                )
                jobs_list[ind_job][1].circuit.ops.append(Op(type=0, gate="_MeasShift", qbits=[0]))

        # new_circ.bind_variables(dic_values)

        jobs_list[ind_job][1].observable = Observable(jobs_list[ind_job][1].circuit.nbqbits, pauli_terms=[Term(1.0, "Z", [0])])

        # jobs_list.append((update_coeff, new_job))
        jobs_list[ind_job][0] = update_coeff
        ind_job += 1

    if do_print_status:
        print()
        print("Computation success!")

        print()
        print("******************* The gates that will be delivered (first job only) *******************")
        coe, nouveau_job = jobs_list[0]
        for ind, (opname, params, qbits) in enumerate(nouveau_job.circuit.iterate_simple()):

            if do_print_status:
                print("OPNAME : ", opname)
                print(ind, opname, params, qbits)

        # print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        # for op in jobs_list[0][1].circuit.ops:
        #    print(op.gate, jobs_list[0][1].circuit.gateDic[op.gate])
        # print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    return jobs_list.values()

    # In a plugin, some post_processing would be needed to gather results in the right order


#################################################################################
# Some useful functions to call the core one, with or without paramterised jobs #
#################################################################################


def product_param_real_part(job, bra_key, ket_key, user_custom_gates=None, do_print_status=False):
    """
    This function will call partial_derivatives in the right fashion to get the parameterised circuits to compute Re<\partial_bra \Psi|\partial_ket \Psi>


    Args:

        job (job) :
        bra_key (string) : key identifier. Set to "no_differentiation" to do nothing with the bra side.
        ket_key (string) : key identifier. Set to "no_differentiation" to do nothing with the ket side.

        do_print_status (boolean) : For debug purposes

    Returns:

        A PARAMETRIC job
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
                do_print_status=do_print_status,
            )
        )
        # print("JL", jobs_list)
    elif bra_key == "no_differentiation":

        ket_list = partial_derivatives(
            job,
            parameter_key=ket_key,
            braket_side="ket",
            add_ancilla=True,
            user_custom_gates=user_custom_gates,
            do_print_status=do_print_status,
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
            do_print_status=do_print_status,
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
            do_print_status=do_print_status,
        )

        for (bra_coeff, bra_job) in bra_list:

            p_ket_list = []
            p_ket_list = partial_derivatives(
                bra_job,
                parameter_key=ket_key,
                braket_side="ket",
                add_ancilla=False,
                user_custom_gates=user_custom_gates,
                do_print_status=do_print_status,
            )

            for (ket_coeff, new_job) in p_ket_list:

                jobs_list.append((bra_coeff * ket_coeff, new_job))

    return jobs_list


def compute_product_real_part_from_jobs_list(
    jobs_list, qpu, var_theta_dic, my_gate_set=get_custom_gate_set_1(), do_print_status=False
):
    """
    Args:

        jobs_list ((float, job) tuple list) : A job list like the ones returned by product_param_real_part function

    Returns:
        A complex value
    """
    result = 0.0 + 0.0 * 1j
    for (coeff, my_test_job) in jobs_list:
        # my_test_job.circuit = my_test_job.circuit.bind_variables(var_theta_dic)

        # Temporary for XX gate **************************************************************
        # my_gate_set = get_custom_gate_set_1()
        temp_job = my_test_job(gate_set=my_gate_set, **var_theta_dic)
        # ************************************************************************************

        # temp_job = my_test_job(**var_theta_dic)
        try:
            # val = qpu.submit(my_test_job).value
            val = qpu.submit(temp_job).value

        except:
            for op in my_test_job.circuit.ops:
                print(op.gate, my_test_job.circuit.gateDic[op.gate])
            raise
        result += coeff * val

    return result


def compute_product_real_part(
    qpu, job, bra_key, ket_key, var_theta_dic, user_custom_gates=None, my_gate_set=get_custom_gate_set_1(), do_print_status=False
):
    """
    Will call product_param_real_part, then evaluate at var_theta point and then post process to get the result.

    Args :

        qpu (QPU) : The QPU that should be used
        job (job) :
        bra_key (string) : Set to "no_differentiation" to do nothing with the bra side.
        ket_key (string) : Set to "no_differentiation" to do nothing with the ket side.
        var_theta_dic (dict) : Dictionary {param_key : coordinate}. Indicates where in the parameter space to evaluate ansatz states.

    """

    result = 0.0 + 0.0 * 1j

    jobs_list = product_param_real_part(
        job, bra_key=bra_key, ket_key=ket_key, user_custom_gates=user_custom_gates, do_print_status=do_print_status
    )

    return compute_product_real_part_from_jobs_list(
        jobs_list, qpu, var_theta_dic, my_gate_set=my_gate_set, do_print_status=do_print_status
    )


###########################################################
# For automatic gradient purposes                         #
###########################################################


def hamiltonian_gradient_param_imag_part(job, hamilt, partial_key, user_custom_gates=None, do_print_status=False):
    r"""
    Returns the (coefficients, jobs) list to compute $\partial_{partial_key} <\Psi | H | \Psi> where Psi is the ansatz prepared according to the given job.
    WARNING : For now, this function will only work if the ansatz job paramterised gates are of the shape e^(-i\theta /2) !

    Args :

        job (Job) : The entry job. Its circuit indicates how to prepare parameterised ansatz states.
        hamilt (Observable) : The observable corresponds to the hamiltonian corresponding to the energy of interest.
        partial_key (string) : Key identifier with respect to which we want to compute the partial differentiation

        do_print_status (boolean) : For debug purposes

    Returns :

        The (coefficients, jobs) list to compute $\partial_{partial_key}. The jobs are PARAMETRIC.

    """

    O_list = []
    H_coeff_list = []

    # hamilt = \sum_j \lambda_j O_j

    for ind, term in enumerate(hamilt.terms):
        # first, construct operator corresponding to O_j
        is_og_valid, og_term_list = split_check_term(term.op, len(term.qbits))
        if is_og_valid:
            op_gate = AbstractGate(term.op, [], len(term.qbits))()  # We will deal with the matrix later in the core function
        else:
            op_gate_matrix = get_predef_generator()[term.op[0]]
            if len(term.op) > 1:
                # assumes term.qbits is sorted in ascending order
                for op in term.op[1:]:
                    op_gate_matrix = np.kron(op_gate_matrix, get_predef_generator()[op])
            op_gate = AbstractGate(
                "%s" % (term.op), [], len(term.qbits), matrix_generator=lambda op_gate_matrix=op_gate_matrix: op_gate_matrix
            )()
            # op_gate = AbstractGate("%s_%s"%(term.op, ind), [], len(term.qbits), matrix_generator=lambda op_gate_matrix=op_gate_matrix:op_gate_matrix)()
        O_list.append((op_gate, term.qbits))
        H_coeff_list.append(term.coeff)

    # now construct circuits

    jobs_list = []

    for indj, (O_gate, O_qbits) in enumerate(O_list):

        ### HERE WE WILL BUILD A CIRCUIT WITH SOME OF THE HAMILT GATES (ISSUES with the XX gate for instance)
        first_jobs_list = partial_derivatives(
            job,
            parameter_key=partial_key,
            braket_side="ket",
            add_ancilla=True,
            O_gate=O_gate,
            O_qbits=O_qbits,
            user_custom_gates=user_custom_gates,
            do_print_status=do_print_status,
        )

        # for (_, op_job) in first_jobs_list:
        for (op_coeff, op_job) in first_jobs_list:
            corr_coeff = op_coeff / (-1j / 2)
            if np.isreal(corr_coeff):
                jobs_list.append(
                    (-H_coeff_list[indj] * np.real(corr_coeff), op_job)
                )  # minus sign is because we compute (-1)*Im(<O|U(\theta)^\dagger.O_j.V_i(\theta)|0> with the hadamard test circuit with the "ket" side

            else:
                raise Exception(
                    "For now, this function will only work if the ansatz job paramterised gates are of the shape e^(-i \lambda \\theta \\sigma /2) ! Instead of -i/2*\lambda, got a gen coeff : %f+%f i"
                    % (np.real(op_coeff), np.imag(op_coeff))
                )

    return jobs_list


def auto_differentiation_gradient_dictionary(job, hamilt, parameters_dict, user_custom_gates=None, do_print_status=False):
    """
    Computes the jobs lists dictionary to evaluate gradient.

    Returns:
        The jobs are PARAMETRIC.
    """
    gradient_jobs_dict = {}
    for var_key in parameters_dict:
        gradient_jobs_dict[var_key] = hamiltonian_gradient_param_imag_part(
            job, hamilt, var_key, user_custom_gates=user_custom_gates, do_print_status=do_print_status
        )

    return gradient_jobs_dict


def compute_gradient_term_from_jobs_list(jobs_list, qpu, var_theta_dic, my_gate_set=get_custom_gate_set_1(), do_print_status=False):
    """
    From a jobs_list (provided by hamiltonian_gradient_param_imag_part function) sums up the results and returns the gradient term.
    Used by compute gradient function.

    Returns:
        A real value (float)
    """
    grad_term = 0.0
    for (coeff, my_test_job) in jobs_list:
        # my_test_job.circuit = my_test_job.circuit.bind_variables(var_theta_dic)

        # Temporary for XX gate **************************************************************
        # my_gate_set = get_custom_gate_set_1()
        temp_job = my_test_job(gate_set=my_gate_set, **var_theta_dic)
        # ************************************************************************************

        # temp_job = my_test_job(**var_theta_dic)
        try:
            # val = qpu.submit(my_test_job).value
            val = qpu.submit(temp_job).value
        except:
            for op in my_test_job.circuit.ops:
                print(op.gate, my_test_job.circuit.gateDic[op.gate])
            raise

        grad_term += coeff * val

    return grad_term


def compute_gradient_from_jobs_dict(jobs_dict, qpu, var_theta_dic, my_gate_set=get_custom_gate_set_1(), do_print_status=False):
    """
    From a jobs_dict {param_key: jobs_list}sums up the results and returns the gradient dict.
    Parameters keys from jobs_dict and var_theta_dic must be the same.

    Returns:
        A dictionary {param_key: gradient value (float)}
    """

    q_gradient = {var_key: 0.0 for var_key in var_theta_dic}

    for var_key in jobs_dict:
        jobs_list = jobs_dict[var_key]
        q_gradient[var_key] = compute_gradient_term_from_jobs_list(
            jobs_list, qpu, var_theta_dic, my_gate_set=my_gate_set, do_print_status=False
        )

    return q_gradient


def compute_gradient(
    qpu, job, hamilt, var_theta_dic, user_custom_gates=None, my_gate_set=get_custom_gate_set_1(), do_print_status=False
):
    """
    Calls hamiltonian_gradient_param_imag_part to compute with the given QPU the energy gradient with respect to the parameters of the provided job circuit
    Args :


        var_theta_dic (dict) : Parameters values to bind variables

    Returns :

        A dictionary (parameter_key : partial derivative wrt the parameter)
    """
    parameters_dict = job.circuit.var_dic  # The parameters
    q_gradient = {var_key: 0.0 for var_key in parameters_dict}

    gradient_jobs_dict = auto_differentiation_gradient_dictionary(job, hamilt, parameters_dict, user_custom_gates=user_custom_gates)

    q_gradient = compute_gradient_from_jobs_dict(
        gradient_jobs_dict, qpu, var_theta_dic, my_gate_set=my_gate_set, do_print_status=do_print_status
    )

    return q_gradient


##################################################################################################
# For automatic QFIM (Quantum Fisher Information Matrix) purposes                                #
##################################################################################################


def auto_differentiation_QFIM_dictionaries(job, nb_parameters, parameters_index, user_custom_gates=None):
    """
    Computes the three jobs dictionaries used for QFIM computation

    Args :
        job (Job) : The job we aim at differentiating
        nb_parameters (int) : The number of parameters keys
        parameters_index (dict) : A dictionary to link parameters names to their index
        user_custom_gates (dict) :  [Default to None] A dictionary where the user can provide entries that help define custom constant gates in the circuit (typically gates that do not appear in AQASM defaults). Syntax: {gate_name (string) : GateDefinition (GateDefinition)}. NOT SUPPORTING CUSTOM PARAMETERISED GATES YET.

    Returns :
        A three-dictionary tuple. The jobs inside are PARAMETRIC

    ADAPT TO USE ONLY TWO DICTS (USING HERMITIAN SYMMETRY) ???
    ### TO THINK ABOUT: Only compute superior triangle if all operators are hermitian ? Q: <|>^dagger = <|>^conjugate ?
    ### Always put ones on the diagonal if all operations are unitary ?
    """

    QFIM_dkdl_jobs_dict = {}
    QFIM_dkpsi_jobs_dict = {}
    QFIM_psidl_jobs_dict = {}  # Use np.conj with the result instead ?

    for k in range(nb_parameters):

        param_k = parameters_index[k]
        QFIM_dkpsi_jobs_dict[param_k] = product_param_real_part(
            job, bra_key=param_k, ket_key="no_differentiation", user_custom_gates=user_custom_gates, do_print_status=False
        )
        QFIM_psidl_jobs_dict[param_k] = product_param_real_part(
            job, bra_key="no_differentiation", ket_key=param_k, user_custom_gates=user_custom_gates, do_print_status=False
        )  # Use symmetry instead ?

        for l in range(nb_parameters):

            param_l = parameters_index[l]
            QFIM_dkdl_jobs_dict[param_k + ";" + param_l] = product_param_real_part(
                job, bra_key=param_k, ket_key=param_l, user_custom_gates=user_custom_gates, do_print_status=False
            )

    return (QFIM_dkdl_jobs_dict, QFIM_dkpsi_jobs_dict, QFIM_psidl_jobs_dict)


def compute_pure_QFIM_term_from_jobs_dicts(
    QFIM_dkdl_jobs_dict,
    QFIM_dkpsi_jobs_dict,
    QFIM_psidl_jobs_dict,
    qpu,
    var_theta_dic,
    my_gate_set=get_custom_gate_set_1(),
    do_print_status=False,
):
    """
    Using jobs_dicts
        QFIM_dkdl_jobs_dict = {var_key: [(coeff, job) as a jobs_list] for every var_key (parameters names)}
        QFIM_dkpsi_jobs_dict = {var_key: [(coeff, job) as a jobs_list] for every var_key (parameters names)}
        QFIM_psidl_jobs_dict = {var_keys: [(coeff, job) as a jobs_list] for every var_keys)} ex var_keys=var_key1+";"+"var_key2" (the ';' is important)

    Return:

        The QFIM (Quantum Fisher Information Matrix) for pure states. A float numpy array.
    """

    nb_parameters = len(var_theta_dic)
    parameters_index = {}

    i = 0
    for varkey in var_theta_dic:
        parameters_index[i] = varkey
        i += 1

    pure_QFIM = np.zeros((nb_parameters, nb_parameters), dtype=float)

    for k in range(nb_parameters):

        param_k = parameters_index[k]
        dkpsi = compute_product_real_part_from_jobs_list(
            QFIM_dkpsi_jobs_dict[param_k], qpu, var_theta_dic, my_gate_set=my_gate_set, do_print_status=do_print_status
        )  # should be a pure imaginary number

        for l in range(nb_parameters):

            param_l = parameters_index[l]
            psidl = compute_product_real_part_from_jobs_list(
                QFIM_psidl_jobs_dict[param_l], qpu, var_theta_dic, my_gate_set=my_gate_set, do_print_status=do_print_status
            )  # should be a pure imaginary number

            dkdl = compute_product_real_part_from_jobs_list(
                QFIM_dkdl_jobs_dict[param_k + ";" + param_l],
                qpu,
                var_theta_dic,
                my_gate_set=my_gate_set,
                do_print_status=do_print_status,
            )

            if do_print_status:
                print("<d[%i]|d[%i]> = " % (k, l), dkdl)
                print("<d[%i]|Psi> = " % k, dkpsi)
                print("<Psi|d[%i]> = " % l, psidl)

            pure_QFIM[k][l] = 4 * (dkdl - dkpsi * psidl)  # OK according to Re parts since the <dkPsi|Psi> is in iR

    return pure_QFIM


def compute_QFIM(qpu, job, var_theta_dic, user_custom_gates=None, my_gate_set=get_custom_gate_set_1(), do_print_status=False):
    """
    Compute Quantum Fisher Information Matrix associated to the ansatz job and the parameters values.

    Return:
        A tuple (parameters_index, QFIM). The index gives the order wrt which differentiations were performed.
    """

    nb_parameters = len(var_theta_dic)
    parameters_index = {}

    i = 0
    for varkey in var_theta_dic:
        parameters_index[i] = varkey
        i += 1

    (QFIM_dkdl_jobs_dict, QFIM_dkpsi_jobs_dict, QFIM_psidl_jobs_dict) = auto_differentiation_QFIM_dictionaries(
        job, nb_parameters, parameters_index, user_custom_gates=user_custom_gates
    )  # The three QFIM dictionaries.

    QFIM = compute_pure_QFIM_term_from_jobs_dicts(
        QFIM_dkdl_jobs_dict,
        QFIM_dkpsi_jobs_dict,
        QFIM_psidl_jobs_dict,
        qpu,
        var_theta_dic,
        my_gate_set=my_gate_set,
        do_print_status=do_print_status,
    )

    return (parameters_index, QFIM)
