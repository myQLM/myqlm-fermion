"""
Useful functions to implement fermionic Bogolyubov transformations (= rotations of fermionic modes) that connect gaussian states.
"""

import itertools

import numpy as np

from scipy.linalg import expm, logm
from qat.fermion.hamiltonians import ElectronicStructureHamiltonian
from qat.fermion.util import init_creation_ops, dag


def herm_conj(mat):

    return np.transpose(np.conj(mat))

def check_unitarity(mat):
    
    size = np.shape(mat)[0]
    
    diff = np.linalg.norm(np.eye(size) - mat.dot(herm_conj(mat)))
    
    try:
        assert diff<1e-12
    except:
        raise
        
def generate_unitary_matrix(size, seed=None):
    """
    Create a random unitary matrix by exponentiating a symmetric matrix:
    :math:`U = \exp(iS)` where :math:`S = M + M^{\dagger}`, M being drawn at random.

    Args:
        size (int): size of the matrix to generate
        seed (int, optional): seed for random draw

    Output:
        U (numpy array): a unitary matrix 
    """ 
    np.random.seed(seed=seed)
    
    mat = np.random.random((size, size))
    sym_mat = mat + herm_conj(mat)
    
    U = expm(1j*sym_mat)
    
    check_unitarity(U)
        
    return U 


def make_quadratic_hamiltonian(hpq):
    """
    Construct the quadratic ElectronicStructureHamiltonian
    :math:`H = \sum \limits_{pq} h_{pq} a^{\dagger}_p a_q`
    
    Args:
        hpq (numpy array): the matrix defining the Hamiltonian
    
    Return:
        H (ElectronicStructureHamiltonian)
    """

    M = np.shape(hpq)[0]

    try:
        assert np.linalg.norm(hpq - herm_conj(hpq))<1e-12
    except:
        raise

    H = ElectronicStructureHamiltonian(hpq=hpq, hpqrs=np.zeros((M, M, M, M)), constant_coeff=0)

    return H

def make_rotation_thouless(r, antisymmetrize=False):
    """
    Create the Bogolyubov transformation operator associated to r

    Args:
        r (numpy array): a unitary matrix

    """

    M = r.shape[0]
    a_dag = init_creation_ops(M)
    
    mat_to_exponentiate = np.zeros((2**M, 2**M), dtype='complex128')
    log_r = logm(r)
    
    for i, j in itertools.product(range(M), repeat=2):
        ops = a_dag[i].dot(dag(a_dag[j]))
        if antisymmetrize:
            mat_to_exponentiate += log_r[i, j]*(ops - dag(ops))
        else:
            mat_to_exponentiate += log_r[i, j]*ops
         
        
    U_cal = expm(mat_to_exponentiate)

    check_unitarity(U_cal)
    
    return U_cal

def compute_diff_to_thouless(r, antisymmetrize=False):
    """"
    Check that the unitary transformation r (rotation if real-valued) of the fermionic modes
    a_dag, namely b^{\dagger}_k = \sum_l r_{kl} a^{\dagger}_l
    translates into Thouless' theorem's prescription in the Fock space, namely:
    U_cal = \exp(\sum_{pq} (\log r)_{pq} (a^{\dagger}_p a_q - h.c.))).
    
    Specifically, this is done checking that
    
    H = U_cal^{\dagger} H_tilde U_cal
    
    with H = \sum_{pq} h_{pq} a^{\dagger}_p a_q some quadratic Hamiltonian (h hermitian)
    and H_tilde = \sum_{pq} h_{pq} b^{\dagger}_p b_q = \sum_{pq} h_tilde_{pq} a^{\dagger}_p a_q.
    
    Args:
        r (numpy array): a unitary matrix
        antisymmetrize (bool, optional): whether to use an antisymmetrical version of the operator products
                                         appearing in the definition of the rotation in the Fock space

    Returns:
        diff (float): the difference in terms of conjugated action between Thouless' prescription 
                      and the operator, as computed on a random Hamiltonian instance
        
    """
    
    M = np.shape(r)[0] # number of fermionic modes
    
    hpq = np.random.random((M, M))
    hpq = hpq + np.transpose(hpq)

    hpq_tilde = np.dot(r, np.dot(hpq, herm_conj(r)))

    H = make_quadratic_hamiltonian(hpq)
    H_tilde = make_quadratic_hamiltonian(hpq_tilde)
    
    # compute the unitary implementing the rotation of the fermionic modes
    # in the Fock space
    U_cal = make_rotation_thouless(r, antisymmetrize=antisymmetrize)
    
    H_mat = H.get_matrix()
    H_tilde_mat = H_tilde.get_matrix()
    
    product = np.dot(U_cal, np.dot(H_mat, herm_conj(U_cal)))
    diff = np.linalg.norm(H_tilde_mat - product)
    
    return diff

