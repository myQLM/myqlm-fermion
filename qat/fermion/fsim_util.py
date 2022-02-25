import numpy as np
from qat.lang.AQASM import Program, QRoutine, AbstractGate, X, RY, CNOT
from qat.core import default_gate_set

fSim_gen = lambda theta, phi : np.array([[1,0,0,0],
                                  [0,np.cos(theta),-1j*np.sin(theta),0],
                                  [0,-1j*np.sin(theta),np.cos(theta),0],
                                  [0,0,0,np.exp(-1j*phi)]
                                  ],dtype=np.complex_)


fSim = AbstractGate("fSim", [float, float], 2,
                   matrix_generator=lambda theta, phi, mat_gen=fSim_gen: mat_gen(theta, phi))

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

    q1 = nbqbits//2 - 1
    q2 = nbqbits//2
            
    qrout.apply(fSim(theta[ind_theta], theta[ind_theta + 1]), q1, q2)
    ind_theta += 2

    for j in range(nbqbits//2 - 1):
        qrout.apply(fSim(theta[ind_theta], theta[ind_theta + 1]), q1-j-1, q1-j)
        ind_theta += 2
        qrout.apply(fSim(theta[ind_theta], theta[ind_theta + 1]), q2+j, q2+j+1)
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