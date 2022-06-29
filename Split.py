import numpy as np
import matplotlib.pyplot as plt
from tvsclib.stage import Stage
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem
import tvsclib.utils as utils
import tvsclib.math as math


from tvsclib.canonical_form import CanonicalForm


def split_stage(stage,i_in,i_out):
    A=stage.A_matrix
    B=stage.B_matrix
    C=stage.C_matrix
    D=stage.D_matrix
    (d_stateo,d_statei)=A.shape
    U,s,Vt = np.linalg.svd(np.block([[C[i_out:,:],D[i_out:,:i_in]],[A,B[:,:i_in]]]),full_matrices=False)

    s_sqrt = np.sqrt(s)
    stage_alpha=Stage(s_sqrt.reshape(-1,1)*Vt[:,:d_statei],                 s_sqrt.reshape(-1,1)*Vt[:,d_statei:],                 C[:i_out,:],                 D[:i_out,:i_in])
    stage_beta=Stage(U[U.shape[0]-d_stateo:,:]*s_sqrt.reshape(1,-1),               B[:,i_in:],               U[:U.shape[0]-d_stateo,:]*s_sqrt.reshape(1,-1),               D[i_out:,i_in:])

    return stage_alpha,stage_beta

def split_stage_anti(stage,i_in,i_out,D):
    A=stage.A_matrix
    B=stage.B_matrix
    C=stage.C_matrix
    #D=stage.D_matrix
    (d_stateo,d_statei)=A.shape
    U,s,Vt = np.linalg.svd(np.block([[C[:i_out,:],D[:i_out,i_in:]],[A,B[:,i_in:]]]),full_matrices=False)

    s_sqrt = np.sqrt(s)
    stage_alpha=Stage(s_sqrt.reshape(-1,1)*Vt[:,:d_statei],                 s_sqrt.reshape(-1,1)*Vt[:,d_statei:],                 C[i_out:,:],                 np.zeros_like(D[:i_out,:i_in]))
    stage_beta=Stage(U[U.shape[0]-d_stateo:,:]*s_sqrt.reshape(1,-1),               B[:,:i_in],               U[:U.shape[0]-d_stateo,:]*s_sqrt.reshape(1,-1),               np.zeros_like(D[i_out:,i_in:]))

    return stage_alpha,stage_beta

def split(system,k,i_in,i_out):
    stage_alpha,stage_beta = split_stage(system.stages[k],i_in,i_out)
    system.stages[k]=stage_alpha
    system.stages.insert(k+1,stage_beta)

def split_anti(system,k,i_in,i_out,D):
    stage_alpha,stage_beta = split_stage_anti(system.stages[k],i_in,i_out,D)
    system.stages[k]=stage_beta
    system.stages.insert(k+1,stage_alpha)

def split_mixed(system,k,i_in,i_out):
    split_anti(system.anticausal_system,k,i_in,i_out,system.causal_system.stages[k].D_matrix)
    split(system.causal_system,k,i_in,i_out)





def initial(T):
    """
    returns a tupel with the initial lists
    for As,Bs,Cs,Ds

    """
    return StrictSystem(stages=[Stage(np.zeros((0,0)),np.zeros((0,T.shape[1])),np.zeros((T.shape[0],0)),T)],causal=True)

def initial_mixed(T):
    """
    returns a tupel with the initial lists
    for As,Bs,Cs,Ds
    """
    return MixedSystem(\
            causal_system=StrictSystem(stages=[Stage(np.zeros((0,0)),np.zeros((0,T.shape[1])),np.zeros((T.shape[0],0)),T)],causal=True),\
        anticausal_system=StrictSystem(stages=[Stage(np.zeros((0,0)),np.zeros((0,T.shape[1])),np.zeros((T.shape[0],0)),np.zeros_like(T))],causal=False)\
                      )


class Stage_sigmas:
    """Class to make it possible to store the singular values alongside.

    This makes it possible to trivially compute input normal, output normal and balanced realizations
    """
    def __init__(self, A_tilde:np.ndarray, B_tilde:np.ndarray, C_tilde:np.ndarray, D_matrix:np.ndarray,\
    s_out:np.ndarray,s_in:np.ndarray,canonical_form=CanonicalForm.OUTPUT,copy = True):
        """__init__ Constructor

        Args:
            A_tilde  (np.ndarray): System matrix
            B_tilde  (np.ndarray): Input matrix
            C_tilde  (np.ndarray): Output matrix
            D_matrix (np.ndarray): Pass through matrix
            s_out    (np.ndarray): singular values of H_{k+1}
            s_in     (np.ndarray): singular values of H_k
            canonical_form (tvsclib.CanonicalForm,optional):  default canonical_form

            copy: if True, the matricies are copied. Default is True
        """
        if copy:
            self.A_tilde = A_tilde.copy()
            self.B_tilde = B_tilde.copy()
            self.C_tilde = C_tilde.copy()
            self.D_matrix = D_matrix.copy()
            self.s_out = s_out.copy()
            self.s_in = s_in.copy()
        else:
            self.A_tilde = A_tilde
            self.B_tilde = B_tilde
            self.C_tilde = C_tilde
            self.D_matrix = D_matrix
            self.s_out = s_out
            self.s_in = s_in
        self.canonical_form=canonical_form

    @property
    def A_matrix(self):
        if self.canonical_form==CanonicalForm.OUTPUT:
            return self.s_out.reshape(-1,1)*self.A_tilde
        elif self.canonical_form==CanonicalForm.INPUT:
            return self.A_tilde*self.s_in.reshape(1,-1)
        else:
            return np.sqrt(self.s_out.reshape(-1,1))*self.A_tilde*np.sqrt(self.s_in.reshape(1,-1))

    @property
    def B_matrix(self):
        if self.canonical_form==CanonicalForm.OUTPUT:
            return self.s_out.reshape(-1,1)*self.B_tilde
        elif self.canonical_form==CanonicalForm.INPUT:
            return self.B_tilde
        else:
            return np.sqrt(self.s_out.reshape(-1,1))*self.B_tilde

    @property
    def C_matrix(self):
        if self.canonical_form==CanonicalForm.OUTPUT:
            return self.C_tilde
        elif self.canonical_form==CanonicalForm.INPUT:
            return self.C_tilde*self.s_in.reshape(1,-1)
        else:
            return self.C_tilde*np.sqrt(self.s_in.reshape(1,-1))


    def input_normal(self):
        return Stage(self.A_tilde*self.s_in.reshape(1,-1),\
                     self.B_tilde,\
                     self.C_tilde*self.s_in.reshape(1,-1),\
                     self.D_matrix)

    def output_normal(self):
        return Stage(self.s_out.reshape(-1,1)*self.A_tilde,\
                     self.s_out.reshape(-1,1)*self.B_tilde,\
                     self.C_tilde,\
                     self.D_matrix)

    def balanced(self):
        sqrts_in = np.sqrt(self.s_in.reshape(1,-1))
        sqrts_out = np.sqrt(self.s_out.reshape(-1,1))
        return Stage(sqrts_out*self.A_tilde*sqrts_in,\
                     sqrts_out*self.B_tilde,\
                     self.C_tilde*sqrts_in,\
                     self.D_matrix)

    def output_input_normal(self):
        """
        This can be used to get a stage where the previous R_k is input normal
        and the next O_{k+1} is output normal
        """
        return Stage(self.s_out.reshape(-1,1)*self.A_tilde*self.s_in.reshape(1,-1),\
                     self.s_out.reshape(-1,1)*self.B_tilde,\
                     self.C_tilde*self.s_in.reshape(1,-1),
                     self.D_matrix)


    @property
    def dim_in(self) -> int:
        """dim_in Input size

        Returns:
            int: Input size
        """
        return self.B_tilde.shape[1]

    @property
    def dim_out(self) -> int:
        """dim_out Output suze

        Returns:
            int: Output suze
        """
        return self.C_tilde.shape[0]

    @property
    def dim_state(self) -> int:
        """dim_state Size of the state space

        Note here that the state dim for a stage is the input state dims.
        This makes the indecing consistent with the formulas
        Returns:
            int: Size of the state space
        """
        return self.A_tilde.shape[1]


def split_stage_sigmas(stage,i_in,i_out,epsilon=1e-12):
    #we need this stage such that previous is input normal and later is output normal
    s_in = stage.s_in
    s_out= stage.s_out
    stage_tr = stage.output_input_normal()
    A=stage_tr.A_matrix
    B=stage_tr.B_matrix
    C=stage_tr.C_matrix
    D=stage.D_matrix
    #we sometimes use B_tilde and C_tilde to avoid dividing it by sigma again

    (d_stateo,d_statei)=A.shape
    U,s,Vt = np.linalg.svd(np.block([[C[i_out:,:],D[i_out:,:i_in]],[A,B[:,:i_in]]]),full_matrices=False)
    n = np.count_nonzero(s>epsilon)
    U = U[:,:n]
    s = s[:n]
    Vt = Vt[:n]

    stage_alpha=Stage_sigmas(
                 Vt[:,:d_statei]/s_in.reshape(1,-1),\
                 Vt[:,d_statei:],\
                 stage.C_tilde[:i_out,:],\
                 D[:i_out,:i_in],\
                 s,s_in)
    stage_beta=Stage_sigmas(
               U[U.shape[0]-d_stateo:,:]/s_out.reshape(-1,1),\
               stage.B_tilde[:,i_in:],\
               U[:U.shape[0]-d_stateo,:],\
               D[i_out:,i_in:],\
               s_out,s)

    return stage_alpha,stage_beta

def split_stage_sigmas_anti(stage,i_in,i_out,D,epsilon=1e-12):
    #we need this stage such that previous is input normal and later is output normal
    s_in = stage.s_in
    s_out= stage.s_out
    stage_tr = stage.output_input_normal()
    A=stage_tr.A_matrix
    B=stage_tr.B_matrix
    C=stage_tr.C_matrix
    #D=stage.D_matrix
    #we sometimes use B_tilde and C_tilde to avoid dividing it by sigma again

    (d_stateo,d_statei)=A.shape
    U,s,Vt = np.linalg.svd(np.block([[C[:i_out,:],D[:i_out,i_in:]],[A,B[:,i_in:]]]),full_matrices=False)
    n = np.count_nonzero(s>epsilon)
    U = U[:,:n]
    s = s[:n]
    Vt = Vt[:n]

    stage_alpha=Stage_sigmas(
                 Vt[:,:d_statei]/s_in.reshape(1,-1),\
                 Vt[:,d_statei:],\
                 stage.C_tilde[i_out:,:],\
                 np.zeros_like(D[i_out:,i_in:]),\
                 s,s_in)
    stage_beta=Stage_sigmas(
               U[U.shape[0]-d_stateo:,:]/s_out.reshape(-1,1),\
               stage.B_tilde[:,:i_in],\
               U[:U.shape[0]-d_stateo,:],\
               np.zeros_like(D[:i_out,:i_in]),
               s_out,s)
    return stage_alpha,stage_beta

def split_sigmas(system,k,i_in,i_out,epsilon=1e-12):
    stage_alpha,stage_beta = split_stage_sigmas(system.stages[k],i_in,i_out,epsilon=epsilon)
    system.stages[k]=stage_alpha
    system.stages.insert(k+1,stage_beta)

def split_sigmas_anti(system,k,i_in,i_out,D,epsilon=1e-12):
    stage_alpha,stage_beta = split_stage_sigmas_anti(system.stages[k],i_in,i_out,D,epsilon=epsilon)
    system.stages[k]=stage_beta
    system.stages.insert(k+1,stage_alpha)

def split_sigmas_mixed(system,k,i_in,i_out,epsilon=1e-12):
    split_sigmas_anti(system.anticausal_system,k,i_in,i_out,system.causal_system.stages[k].D_matrix,epsilon=epsilon)
    split_sigmas(system.causal_system,k,i_in,i_out,epsilon=epsilon)




def initial_sigmas(T):
    """

    returns a tupel with the initial lists
    for As,Bs,Cs,Ds

    """

    return StrictSystem(stages=[Stage_sigmas(np.zeros((0,0)),
                                             np.zeros((0,T.shape[1])),
                                             np.zeros((T.shape[0],0)),
                                             T,np.zeros(0),np.zeros(0))],causal=True)

def initial_sigmas_mixed(T):
    """

    returns a tupel with the initial lists
    for As,Bs,Cs,Ds

    """
    return MixedSystem(causal_system=StrictSystem(stages=[Stage_sigmas(np.zeros((0,0)),
                                             np.zeros((0,T.shape[1])),
                                             np.zeros((T.shape[0],0)),
                                             T,np.zeros(0),np.zeros(0))],causal=True),
                   anticausal_system=StrictSystem(stages=[Stage_sigmas(np.zeros((0,0)),\
                                            np.zeros((0,T.shape[1])),\
                                            np.zeros((T.shape[0],0)),\
                                            np.zeros_like(T),np.zeros(0),np.zeros(0))],causal=False))

def get_system(system,canonical_form=CanonicalForm.OUTPUT):
    if canonical_form==CanonicalForm.INPUT:
        stages = [stage.input_normal() for stage in system.stages]
    elif canonical_form==CanonicalForm.OUTPUT:
        stages = [stage.output_normal() for stage in system.stages]
    elif canonical_form==CanonicalForm.BALANCED:
        stages = [stage.balanced() for stage in system.stages]
    return StrictSystem(stages=stages,causal=system.causal)

def get_system_mixed(system,canonical_form=CanonicalForm.OUTPUT):
    return MixedSystem(causal_system=get_system(system.causal_system,canonical_form=canonical_form),\
            anticausal_system=get_system(system.anticausal_system,canonical_form=canonical_form))


def identification_split_system(T,N,epsilon=1e-12,canonical_form=None,compute_sigmas=False):
    """identify creates a mixed system to represent the matrix T using splitting

    Args:
        T (np.ndarray):                     Transfer Operator
        N (int):                            number of splittings. Final system will have 2^N stages
        epsilon (float,optional):           epsilon for balanceed truncation, Default is 1e-15
        canonical_form (tvsclib.CanonicalForm,optional):  Cannonical form of system, Default is None
        compute_sigmas (bool,optional):     If True, the function returns the Hankel singular values, default is False
    Returns:
        system (StrictSystem):              Time varying system
        sigmas ():                          Tupels with sigmas for causal and anticasual Hankel matrices
    """
    sys = initial_sigmas_mixed(T)

    P_col = np.arange(np.sum(sys.dims_in) ,dtype=int)
    P_row = np.arange(np.sum(sys.dims_out),dtype=int)

    Ps_col =np.zeros((N,P_col.size),dtype=int)
    Ps_row =np.zeros((N,P_row.size),dtype=int)
    for n in range(N):
        for k in range(len(sys.causal_system.stages)-1,-1,-1): #reverse ordering makes indexing easier
            i_in =sys.causal_system.stages[k].dim_in//2
            i_out=sys.causal_system.stages[k].dim_out//2
            split_sigmas_mixed(sys,k,i_in,i_out,epsilon=epsilon)

    if compute_sigmas:
        sigmas_causal =[stage.s_in for stage in sys.causal_system.stages][1:]
        sigmas_anticausal =[stage.s_in for stage in sys.anticausal_system.stages][:-1]
    if not canonical_form is None:
        raise NotImplementedError("Not implmented yet use convert functions form file")

    if compute_sigmas:
        return sys,(sigmas_causal,sigmas_anticausal)
    else:
        return sys
