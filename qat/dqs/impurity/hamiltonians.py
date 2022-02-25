# -*- coding : utf-8 -*-
"""
Container for impurity Hamiltonians.
"""

import numpy as np

from qat.dqs.hamiltonians import ElectronicStructureHamiltonian
from qat.dqs.qchem.ucc import transform_integrals_to_new_basis

from qat.lang.AQASM.math_util import reverse_int

def make_anderson_model(U, mu, V, epsilon):
    r"""
    Returns the canonical second quantized form 
    
    .. math::
        H_{\mathrm{CSQ}} = \sum_{p,q} h_{pq} f_p^\dagger f_q + \frac{1}{2}\sum_{p,q,r,s} h_{pqrs} f_p^\dagger f_q^\dagger f_r f_s 
        
    of a single impurity coupled with :math:`n_b` bath modes Anderson model Hamiltonian 
    
    .. math::
        H_{\mathrm{SIAM}} = U c_{\uparrow}^\dagger c_{\uparrow} c_{\downarrow}^\dagger c_{\downarrow} - \mu(c_{\uparrow}^\dagger c_{\uparrow}+c_{\downarrow}^\dagger c_{\downarrow}) 
        + \sum_{i=1..n_b} \sum_{\sigma=\uparrow,\downarrow} V_i (c_{\sigma}^\dagger a_{i,\sigma} + \mathrm{h.c.}) \\
        + \sum_{i=1..n_b} \sum_{\sigma=\uparrow,\downarrow} \epsilon_i a_{i,\sigma}^\dagger a_{i,\sigma}.

    Args:
        U (float): Coulomb repulsion intensity
        mu (float): chemical potential 
        V (vector of size the number of bath modes): tunneling energies 
        epsilon (vector of size the number of bath modes): bath modes energies 

    Returns:
        :class:`~qat.dqs.hamiltonians.ElectronicStructureHamiltonian` object constructed from :math:`h_{pq}` (matrix of size :math:`(2n_b+2) \times (2n_b+2)`) and :math:`h_{pqrs}` (4D tensor with size :math:`2n_b+2` in each dimension)

    .. note::
        Convention:
        :math:`f_0` corresponds to :math:`c_{\uparrow}` (annihilation in the 'up' mode of the impurity),
        :math:`f_1` corresponds to :math:`c_{\downarrow}` (annihilation in the 'down' mode of the impurity),
        :math:`f_2` corresponds to :math:`a_{1,\uparrow}` (annihilation in the 'up' mode of the 1st bath mode),
        :math:`f_3` corresponds to :math:`a_{1,\downarrow}` (annihilation in the 'down' mode of the 1st bath mode),
        and so on.    

    """
    #number of bath modes
    n_b = len(V)
    if len(epsilon) != n_b:
        raise Exception('Error : The bath modes energies vector must be the same size as the tunneling energies vector.')

    #number of fermionic (annihilation) operators f
    fermop_number = 2*n_b+2 

    h_pq = np.zeros((fermop_number,fermop_number))
    h_pqrs = np.zeros((fermop_number,fermop_number,fermop_number,fermop_number))

    #single spin localized on the impurity
    h_pq[0,0] = -mu
    h_pq[1,1] = -mu


    #bath modes terms
    for i in range(0,n_b):
        h_pq[2*(i+1),2*(i+1)] += epsilon[i]
        h_pq[2*(i+1)+1,2*(i+1)+1] += epsilon[i]


    #hopping terms
    for i in range(0,n_b):
        h_pq[0,2*(i+1)] += V[i]
        h_pq[2*(i+1),0] += V[i]
        h_pq[1,2*(i+1)+1] += V[i]
        h_pq[2*(i+1)+1,1] += V[i]


    #Coulomb repulsion when the impurity is occupied by two spins. The minus sign comes from the commutation we need to do in the U-term to get the operators in the right order.
    h_pqrs[0,1,0,1] = -U
    h_pqrs[1,0,1,0] = -U


    return ElectronicStructureHamiltonian(h_pq, h_pqrs)

def make_embedded_model(U, mu, D, lambda_c, t_loc=None, int_kernel=None, grouping='spins'):
    r"""
    Returns the canonical second quantized form 
    
    .. math::
        H_{\mathrm{CSQ}} = \sum_{p,q} h_{pq} f_p^\dagger f_q + \frac{1}{2}\sum_{p,q,r,s} h_{pqrs} f_p^\dagger f_q^\dagger f_r f_s + c\mathbb{I}
    
    of an embedded hamiltonian
    
    .. math::
        H_{\mathrm{emb}} = U \sum \limits_{i,j,k,l=1}^{2M} I_{ijkl} f^{\dagger}_i f_j f^{\dagger}_k f_l
                       - \mu \sum \limits_{i=1}^{M} f^{\dagger}_{i} f_{j} 
                       + \sum \limits_{i, j=1}^{M} t^{\mathrm{loc}}_{ij} f^{\dagger}_i f_j \\
                       + \sum \limits_{i,j=1}^{M} (D_{ij} f^{\dagger}_{i} f_{M+j} + \mathrm{h.c.}) \\
                       + \sum \limits_{i,j=1}^{M} \lambda^c_{ij} f_{M+i} f^{\dagger}_{M+j}
    
    where :math:`M` is the number of orbitals (imp+bath). Indices here correspond to the spin-orbitals ordering referred to as 'cluster' (see below).
    
    Args:
        U (float): onsite repulsion on impurity sites 
        mu (float): chemical potential 
        D (2D array): hopping matrix (aka hybridization) between the correlated orbitals and the uncorrelated bath
        lambda_c (2D array): hopping matrix of the uncorrelated sites 
        t_loc (2D array, optional): hopping matrix of the correlated sites 
        int_kernel (4D array, optional): array :math:`I` with 1 at position :math:`i, j, k, l` where :math:`U` must be put 
                                        (conv. for associated term: :math:`c^{\dagger}c^{\dagger}cc`). Defaults to None,
                                        in which case :math:`U` is put before terms :math:`c^{\dagger}_{2i}c^{\dagger}_{2i+1}c_{2i}c_{2i+1}, i=1..M/2` if grouping is 'clusters', :math:`c^{\dagger}_{i}c^{\dagger}_{i+M}c_{i}c_{i+M}, i=1..M/2` if grouping is 'spins'.  
        grouping (str, optional): defines how spin-orbitals indices are ordered (see below), defaults to 'spins'. 
                       
    Returns:
        :class:`~qat.dqs.hamiltonians.ElectronicStructureHamiltonian`
        
    The two grouping strategies are the following:
    
    - **"clusters"**: the first :math:`M` orbitals SO are :math:`(\uparrow, \mathrm{imp}_0), (\downarrow, \mathrm{imp}_0),..., (\uparrow, \mathrm{imp}_{M-1}), (\downarrow, \mathrm{imp}_{M-1})` and the last :math:`M` orbitals are bath orbitals with similar ordering.
    - **"spins"**: the first :math:`M` orbitals are :math:`(\uparrow, \mathrm{imp}_0), (\uparrow, \mathrm{imp}_1), ..., (\uparrow, \mathrm{bath}_{M-2}), (\uparrow, \mathrm{bath}_{M-1})` and the last :math:`M` orbitals are down orbitals with similar ordering.
    """
    
    M = np.shape(lambda_c)[0] #number of SO in each cluster (imp and bath) = 2*cluster size
    
    h_pq = np.zeros((2*M, 2*M), dtype=np.complex_)
    
    if int_kernel is None:
        h_pqrs = np.zeros((2*M, 2*M, 2*M, 2*M))
    else:
        h_pqrs = -U*int_kernel
    
    const_coeff = 0
    
    for i in range(M):
        
        h_pq[i, i] += -mu #energy of impurity levels
        
        for j in range(M):
            h_pq[i + M, j + M] += -lambda_c[i, j] #energy of uncorrelated levels
            h_pq[i, j + M] += D[i, j] #hopping between the two clusters
            h_pq[j + M, i] += np.conj(D[i, j]) #hopping 
            if t_loc is not None:
                h_pq[i, j] += t_loc[i, j]
                
        const_coeff += lambda_c[i, i]
    
    if grouping=='spins':
        perm_mat = np.zeros((2*M, 2*M)) #permutation matrix, beware: goes from spin ord to cluster ord!

        for i in range(M):
            perm_mat[2*i, i] = 1
            perm_mat[2*i + 1, i + M] = 1
        
        if int_kernel is None and U!=0:
            for i in range(M//2):
                a = ind_clusters_ord(2*i, M)
                b = ind_clusters_ord(2*i + 1, M)
                h_pqrs[a, b, a, b] = -U 
                h_pqrs[b, a, b, a] = -U
        
        h_pq = np.einsum("ap, pq, bq", perm_mat, h_pq, perm_mat)
        
        
    elif grouping=='clusters':
        if int_kernel is None and U!=0:
            for i in range(M//2):
                h_pqrs[2*i, 2*i + 1, 2*i, 2*i + 1] = -U #minus sign comes from the def. of hpqrs: term c_dag c_dag c c
                h_pqrs[2*i + 1, 2*i, 2*i + 1, 2*i] = -U
    else:
        print('Grouping must be either ''clusters'' or ''spins''.')
       
    
    return ElectronicStructureHamiltonian(h_pq, h_pqrs, const_coeff)

def ind_clusters_ord(ind_spins_ord, M):
    """
    Computes the indice with cluster-ordering (up, dn, ..., up, dn)_imp(up, dn, ..., up, dn)_bath 
    of spin-orbital of index ind_clusters_ord in spin-ordering  (up_imp1, up_imp2, ..., up_bath1, ..., up_bathM)(dn_imp1, dn_imp2, ..., dn_bath1, ..., dn_bathM)
    
    Args:
        ind_clusters_ord (int): indice (with spin-ordering) of the spin-orbital we want to compute the indice in cluster-ordering of
        M (int): number of orbitals (imp+bath)
    """
    
    i = ind_spins_ord
    
    if i < M:
        return 2*i
    else:
        try:
            assert(i < 2*M)
        except AssertionError:
            print('index must be lesser than 2*M')
        else:
            return 2*i - (2*M-1)
        
def ind_spins_ord(ind_clusters_ord, M):
    """
    Computes the indice with spin-ordering (up_imp1, up_imp2, ..., up_bath1, ..., up_bathM)(dn_imp1, dn_imp2, ..., dn_bath1, ..., dn_bathM)
    of spin-orbital of index ind_clusters_ord in cluster-ordering (up, dn, ..., up, dn)_imp(up, dn, ..., up, dn)_bath
    
    Args:
        ind_clusters_ord (int): indice (with cluster-ordering) of the spin-orbital we want to compute the indice in spin-ordering of
        M (int): number of orbitals (imp+bath)
    """
    
    i = ind_clusters_ord
    ind_spins_ord = (i%2-1) * (-i//2) + (i%2) * ((i-1)//2 + M)
    return ind_spins_ord
