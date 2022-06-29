from tvsclib.strict_system import StrictSystem
from tvsclib.stage import Stage
from tvsclib.mixed_system import MixedSystem
from tvsclib.approximation import Approximation

import numpy as np
import scipy.linalg as linalg

from tvsclib.transformations.output_normal import OutputNormal
from tvsclib.transformations.input_normal import InputNormal
from tvsclib.transformations.reduction import Reduction
import tvsclib.utils as utils
import tvsclib.math as math
import traceback


def move_left_causal(stages,m_l,epsilon = 1e-16):
    """
        Move left:
        function that moves the boundary to the left by m steps and transform it to input normal

        Args:
            stages (List[Stage]): List with two stages, boundary inbetween will be moved
            m_l (int):            Distance to move
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two stages with moved boundaries,
            s_l (List[float]):    Singular values
    """
    b = stages[0].B_matrix[:,-m_l:]
    U,s_l,Vt= np.linalg.svd(np.hstack([stages[0].A_matrix,stages[0].B_matrix[:,:-m_l]]),full_matrices=False)
    n = np.count_nonzero(s_l>epsilon)
    Us=U[:,:n]*s_l[:n]
    s_l = s_l[:n]

    stages_l = [
        Stage(Vt[:n,:stages[0].A_matrix.shape[1]],Vt[:n,stages[0].A_matrix.shape[1]:],\
            stages[0].C_matrix,stages[0].D_matrix[:,:-m_l],copy=False),
        Stage(stages[1].A_matrix@Us,np.hstack((stages[1].A_matrix@b,stages[1].B_matrix)),\
            stages[1].C_matrix@Us,np.hstack((stages[1].C_matrix@b,stages[1].D_matrix)),copy=False)
            ]
    return stages_l,s_l

def move_left_anticausal(stages_anti,m_l,d,epsilon = 1e-16):
    """
        Move left:
        function that moves the boundary to the left by m steps and transform it to output normal

        Args:
            stages_anti (List[Stage]): List with two stages, boundary inbetween will be moved
            m_l (int):            Distance to move
            d (matrix):           matrix to attach
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two stages with moved boundaries,
            s_l (List[float]):    Singular values
    """

    b = stages_anti[0].B_matrix[:,-m_l:]
    #d = stages_causal[i-1].D_matrix[:,-m_l:]
    d_add = np.zeros((stages_anti[1].D_matrix.shape[0],m_l))
    #d_add = stages_causal[i].C_matrix@stages_causal[i-1].B_matrix[:,-1:]

    U,s_al,Vt= np.linalg.svd(np.block([[b,stages_anti[0].A_matrix],
                                   [d,stages_anti[0].C_matrix]]),full_matrices=False)
    n = np.count_nonzero(s_al>epsilon)
    sVt=s_al[:n].reshape(-1,1)*Vt[:n,:]
    s_al = s_al[:n]
    stages_anti_l=[
        Stage(U[:stages_anti[0].A_matrix.shape[0],:n],stages_anti[0].B_matrix[:,:-m_l],\
          U[stages_anti[0].A_matrix.shape[0]:,:n],stages_anti[0].D_matrix[:,:-m_l],copy=False),
     #Here the A and B are more complicated as we have to stack them
        Stage(sVt@(np.vstack([np.zeros((m_l,stages_anti[1].A_matrix.shape[1])),stages_anti[1].A_matrix])),
          sVt@(np.block([[np.eye(m_l),np.zeros((m_l,stages_anti[1].B_matrix.shape[1]))],
                         [np.zeros((stages_anti[1].B_matrix.shape[0],m_l)),stages_anti[1].B_matrix]])),
          stages_anti[1].C_matrix,np.hstack([d_add,stages_anti[1].D_matrix]),copy=False)
          ]
    return stages_anti_l,s_al

def transform_input_normal_causal(stages,epsilon = 1e-16):
    """
        Transforms input normal:

        Args:
            stages (List[Stage]): List with two stages to transform
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two transformed stages
            s (List[float]):    Singular values
    """
    U,s,Vt= np.linalg.svd(np.hstack([stages[0].A_matrix,stages[0].B_matrix]),full_matrices=False)
    n = np.count_nonzero(s>epsilon)
    Us=U[:,:n]*s[:n]
    s = s[:n]

    stages_n=[
        Stage(Vt[:n,:stages[0].A_matrix.shape[1]],Vt[:n,stages[0].A_matrix.shape[1]:],\
                stages[0].C_matrix,stages[0].D_matrix,copy=False),
        Stage(stages[1].A_matrix@Us,stages[1].B_matrix,\
                stages[1].C_matrix@Us,stages[1].D_matrix,copy=False)
    ]
    return stages_n,s

def transform_input_normal_anticausal(stages_anti,epsilon = 1e-16):
    """
        Transforms input normal:

        Args:
            stages (List[Stage]): List with two stages to transform
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two transformed stages
            s (List[float]):    Singular values
    """
    U,s_a,Vt= np.linalg.svd(np.hstack([stages_anti[1].A_matrix,stages_anti[1].B_matrix]),full_matrices=False)
    n = np.count_nonzero(s_a>epsilon)
    Us=U[:,:n]*s_a[:n]
    s_a = s_a[:n]

    stages_anti_n=[
        Stage(stages_anti[0].A_matrix@Us,stages_anti[0].B_matrix,\
              stages_anti[0].C_matrix@Us,stages_anti[0].D_matrix,copy=False),
        Stage(Vt[:n,:stages_anti[1].A_matrix.shape[1]],Vt[:n,stages_anti[1].A_matrix.shape[1]:],\
              stages_anti[1].C_matrix,stages_anti[1].D_matrix,copy=False)
    ]
    return stages_anti_n,s_a

def transform_output_normal_causal(stages,epsilon = 1e-16):
    """
        Transforms output normal:

        Args:
            stages (List[Stage]): List with two stages to transform
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two transformed stages
            s (List[float]):    Singular values
    """
    U,s,Vt= np.linalg.svd(np.vstack([stages[1].A_matrix,stages[1].C_matrix]),full_matrices=False)
    n = np.count_nonzero(s>epsilon)
    sVt=s[:n].reshape(-1,1)*Vt[:n,:]
    s = s[:n]
    stages_n=[
        Stage(sVt@stages[0].A_matrix,sVt@stages[0].B_matrix,\
            stages[0].C_matrix,stages[0].D_matrix,copy=False),
        Stage(U[:stages[1].A_matrix.shape[0],:n],stages[1].B_matrix,\
              U[stages[1].A_matrix.shape[0]:,:n],stages[1].D_matrix,copy=False)
    ]
    return stages_n,s

def transform_output_normal_anticausal(stages_anti,epsilon = 1e-16):
    """
        Transforms output normal:

        Args:
            stages_anti (List[Stage]): List with two stages to transform
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two transformed stages
            s_a (List[float]):    Singular values
    """

    U,s_a,Vt= np.linalg.svd(np.vstack([stages_anti[0].A_matrix,stages_anti[0].C_matrix]),full_matrices=False)
    n = np.count_nonzero(s_a>epsilon)
    sVt=s_a[:n].reshape(-1,1)*Vt[:n,:]
    s_a = s_a[:n]
    stages_anti_n=[
        Stage(U[:stages_anti[0].A_matrix.shape[0],:n],stages_anti[0].B_matrix,\
              U[stages_anti[0].A_matrix.shape[0]:,:n],stages_anti[0].D_matrix,copy=False),
        Stage(sVt@stages_anti[1].A_matrix,sVt@stages_anti[1].B_matrix,\
              stages_anti[1].C_matrix,stages_anti[1].D_matrix,copy=False)
    ]
    return stages_anti_n,s_a

def move_right_causal(stages,m_r,d_add,epsilon = 1e-16):
    """
        Move right:
        function that moves the boundary to the right by m steps and preserves input normal

        Args:
            stages (List[Stage]): List with two stages, boundary inbetween will be moved
            m_r (int):            Distance to move
            d_add (Matrix):       Matrix to be attached
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two stages with moved boundaries,
            s_r (List[float]):    Singular values
    """
    b = stages[1].B_matrix[:,:m_r]
    d = stages[1].D_matrix[:,:m_r]
    #d_add = np.zeros((stages[0].D_matrix.shape[0],1))
    #d_add = stages_anticausal[i-1].C_matrix@stages_anticausal[i].B_matrix[:,:m_r]


    U,s_r,Vt= np.linalg.svd(np.block([[stages[1].A_matrix,b],
                                  [stages[1].C_matrix,d]]),full_matrices=False)
    n = np.count_nonzero(s_r>epsilon)
    Us=U[:,:n]*s_r[:n]
    s_r = s_r[:n]
    stages_r=[
     #Here the A and B are more complicated as we have to stack them
     Stage(Vt[:n,:]@(np.vstack([stages[0].A_matrix,np.zeros((m_r,stages[0].A_matrix.shape[1]))])),
          Vt[:n,:]@(np.block([[stages[0].B_matrix,np.zeros((stages[0].B_matrix.shape[0],m_r))],
                       [np.zeros((m_r,stages[0].B_matrix.shape[1])),np.eye(m_r)]])),
          stages[0].C_matrix,np.hstack([stages[0].D_matrix,d_add]),copy=False),

    Stage(Us[:stages[1].A_matrix.shape[0],:],stages[1].B_matrix[:,m_r:],\
         Us[stages[1].A_matrix.shape[0]:,:],stages[1].D_matrix[:,m_r:],copy=False)
         ]
    return stages_r,s_r


def move_right_anticausal(stages_anti,m_r,epsilon = 1e-16):
    """
        Move right:
        function that moves the boundary to the right by m steps and preserves output normal

        Args:
            stages_anti (List[Stage]): List with two stages, boundary inbetween will be moved
            m_r (int):            Distance to move
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two stages with moved boundaries,
            s_ar (List[float]):    Singular values
    """
    b = stages_anti[1].B_matrix[:,:m_r]
    U,s_ar,Vt= np.linalg.svd(np.hstack([stages_anti[1].A_matrix,stages_anti[1].B_matrix[:,m_r:]]),full_matrices=False)
    n = np.count_nonzero(s_ar>epsilon)
    sVt=s_ar[:n].reshape(-1,1)*Vt[:n,:]
    s_ar=s_ar[:n]
    stages_anti_r = [
        Stage(stages_anti[0].A_matrix@U[:,:n],np.hstack((stages_anti[0].B_matrix,stages_anti[0].A_matrix@b)),\
          stages_anti[0].C_matrix@U[:,:n],np.hstack((np.zeros((stages_anti[0].D_matrix.shape[0],m_r)),stages_anti[0].D_matrix)),copy=False),
          #D is here 0, instead we have d_add at the causal system. Insted we could use stages_anti_n[0].C_matrix@b
        Stage(sVt[:,:stages_anti[1].A_matrix.shape[1]],sVt[:,stages_anti[1].A_matrix.shape[1]:],\
            stages_anti[1].C_matrix,stages_anti[1].D_matrix[:,:-m_r],copy=False)
        ]
    return stages_anti_r,s_ar

def transform_rl(stages_causal,stages_anticausal,cost,m=1,dir_preset = -1,epsilon=1e-15,
                sigmas_causal=None,sigmas_anticausal=None,cost_global=False,print_progress=False):
    k = len(stages_causal)
    if sigmas_causal is None:
        sigmas_causal = [np.array([np.nan])]*(k-1)
    else:
        sigmas_causal=sigmas_causal.copy()
    if sigmas_anticausal is None:
        sigmas_anticausal = [np.array([np.nan])]*(k-1)
    else:
        sigmas_anticausal=sigmas_anticausal.copy()
    for i in range(1,len(stages_causal)):#loop over inices of causal states

        #no move-> only make R_k input normal
        stages_n,s = transform_input_normal_causal([stages_causal[i-1],stages_causal[i]])
        stages_anti_n,s_a = transform_output_normal_anticausal([stages_anticausal[i-1],stages_anticausal[i]])
        #move left:
        if m>stages_causal[i-1].B_matrix.shape[1]:
            m_l=stages_causal[i-1].B_matrix.shape[1]
        else:
            m_l=m
        if m_l>0:
            stages_l,s_l=move_left_causal([stages_causal[i-1],stages_causal[i]],m_l)
            d = stages_causal[i-1].D_matrix[:,-m_l:]
            stages_anti_l,s_al=move_left_anticausal([stages_anticausal[i-1],stages_anticausal[i]],m_l,d)
        else:
            stages_l=stages_n
            s_l=s
            stages_anti_l=stages_anti_n
            s_al = s_a

        #move right -> base on non move
        if m>stages_n[1].B_matrix.shape[1]:
            m_r=stages_n[1].B_matrix.shape[1]
        else:
            m_r=m
        if m_r>0:
            d_add = stages_anticausal[i-1].C_matrix@stages_anticausal[i].B_matrix[:,:m_r]
            stages_r,s_r=move_right_causal(stages_n,m_r,d_add)
            stages_anti_r,s_ar =move_right_anticausal(stages_anti_n,m_r)
        else:
            stages_r=stages_n
            s_r=s
            stages_anti_r = stages_anti_n
            s_ar = s_a


        dims_in = [stage.D_matrix.shape[1] for stage in stages_causal]
        dims_out = [stage.D_matrix.shape[0] for stage in stages_causal]
        if cost_global:
            #create a vector to modify the output dims
            move_l = np.zeros_like(dims_in)
            move_l[i-1] = move_l[i-1]-m_l
            move_l[i] = move_l[i]+m_l

            move_r = np.zeros_like(dims_in)
            move_r[i-1] = move_r[i-1]+m_r
            move_r[i] = move_r[i]-m_r
            costs = np.array([cost(sigmas_causal,sigmas_anticausal,dims_in,dims_out),\
                            cost(sigmas_causal[:i-1]+[s_l]+sigmas_causal[i:],sigmas_anticausal[:i-1]+[s_al]+sigmas_anticausal[i:],dims_in+move_l,dims_out),\
                            cost(sigmas_causal[:i-1]+[s_r]+sigmas_causal[i:],sigmas_anticausal[:i-1]+[s_ar]+sigmas_anticausal[i:],dims_in+move_r,dims_out)])

        else:
            costs = np.array([cost(s,s_a),\
                            cost(s_l,s_al),\
                            cost(s_r,s_ar)])
        #print("costs_lnr:",costs)
        if dir_preset==-1: #mainly for debugging and testing
            direction = np.argmin(costs)
        else:
            direction = dir_preset
            print("Move",direction)

        if direction == 1:
            stages_causal[i-1]= stages_l[0]
            stages_causal[i]= stages_l[1]
            stages_anticausal[i-1]= stages_anti_l[0]
            stages_anticausal[i]= stages_anti_l[1]
            sigmas_causal[i-1]=s_l
            sigmas_anticausal[i-1]=s_al
        if direction == 0:
            stages_causal[i-1]= stages_n[0]
            stages_causal[i]= stages_n[1]
            stages_anticausal[i-1]= stages_anti_n[0]
            stages_anticausal[i]= stages_anti_n[1]
            sigmas_causal[i-1]=s
            sigmas_anticausal[i-1]=s_a
        if direction == 2:
            stages_causal[i-1]= stages_r[0]
            stages_causal[i]= stages_r[1]
            stages_anticausal[i-1]= stages_anti_r[0]
            stages_anticausal[i]= stages_anti_r[1]
            sigmas_causal[i-1]=s_r
            sigmas_anticausal[i-1]=s_ar
        if print_progress:
            print("=",end = "")
    if print_progress:
        print("")
    return sigmas_causal,sigmas_anticausal


def move_up_causal(stages,m_u,d_add,epsilon = 1e-16):
    """
        Move right:
        function that moves the boundary to the right by m steps and preserves input normal

        Args:
            stages (List[Stage]): List with two stages, boundary inbetween will be moved
            m_u (int):            Distance to move
            d_add (Matrix):       Matrix to be attached
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two stages with moved boundaries,
            s_r (List[float]):    Singular values
    """
    c = stages[0].C_matrix[-m_u:,:]
    d = stages[0].D_matrix[-m_u:,:]
    #d_add = np.zeros((1,stages_n[1].D_matrix.shape[1]))
    #d_add = stages_anticausal[i-1].C_matrix[-m_u:,:]@stages_anticausal[i].B_matrix


    U,s_u,Vt= np.linalg.svd(np.block([[c,d],
                                  [stages[0].A_matrix,stages[0].B_matrix]]),full_matrices=False)
    n = np.count_nonzero(s_u>epsilon)
    sVt=s_u[:n].reshape(-1,1)*Vt[:n,:]
    s_u = s_u[:n]
    stages_u=[
        Stage(sVt[:,:stages[0].A_matrix.shape[1]],sVt[:,stages[0].A_matrix.shape[1]:],\
            stages[0].C_matrix[:-m_u,:],stages[0].D_matrix[:-m_u,:],copy=False),
            #Here the A and C are more complicated as we have to stack them
        Stage(np.hstack([np.zeros((stages[1].A_matrix.shape[0],m_u)),stages[1].A_matrix])@U[:,:n],
          stages[1].B_matrix,\
          np.block([[np.eye(m_u),np.zeros((m_u,stages[1].C_matrix.shape[1]))],
                    [np.zeros((stages[1].C_matrix.shape[0],m_u)),stages[1].C_matrix]])@U[:,:n],
          np.vstack([d_add,stages[1].D_matrix]),copy=False)
        ]
    return stages_u,s_u

def move_up_anticausal(stages_anti,m_u,epsilon = 1e-16):
    """
        Move right:
        function that moves the boundary to the right by m steps and preserves input normal

        Args:
            stages_anti (List[Stage]): List with two stages, boundary inbetween will be moved
            m_u (int):            Distance to move
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two stages with moved boundaries,
            s_r (List[float]):    Singular values
    """
    c = stages_anti[0].C_matrix[-m_u:,:]
    U,s_au,Vt= np.linalg.svd(np.vstack([stages_anti[0].A_matrix,stages_anti[0].C_matrix[:-m_u,:]]),full_matrices=False)
    n = np.count_nonzero(s_au>epsilon)
    Us=U[:,:n]*s_au[:n]
    s_au =s_au[:n]

    stages_anti_u = [
        #D is here 0, instead we have d_add at the causal system. Insted we could use stages_anti[0].C_matrix@b
        Stage(Us[:stages_anti[0].A_matrix.shape[0],:],stages_anti[0].B_matrix,\
          Us[stages_anti[0].A_matrix.shape[0]:,:],stages_anti[0].D_matrix[:-m_u,:],copy=False),
        Stage(Vt[:n,:]@stages_anti[1].A_matrix,Vt[:n,:]@stages_anti[1].B_matrix,\
          np.vstack((c@stages_anti[1].A_matrix,stages_anti[1].C_matrix)),\
          np.vstack((np.zeros((m_u,stages_anti[1].D_matrix.shape[1])),stages_anti[1].D_matrix)),copy=False)
        ]
    return stages_anti_u,s_au


def move_down_causal(stages,m_d,epsilon = 1e-16):
    """
        Move left:
        function that moves the boundary down by m steps and transform it to output normal

        Args:
            stages (List[Stage]): List with two stages, boundary inbetween will be moved
            m_d (int):            Distance to move
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two stages with moved boundaries,
            s_d (List[float]):    Singular values
    """
    c = stages[1].C_matrix[:m_d,:]
    U,s_d,Vt= np.linalg.svd(np.vstack([stages[1].A_matrix,stages[1].C_matrix[m_d:,:]]),full_matrices=False)
    n = np.count_nonzero(s_d>epsilon)
    sVt=s_d[:n].reshape(-1,1)*Vt[:n,:]
    s_d = s_d[:n]
    stages_d = [
        Stage(sVt@stages[0].A_matrix,sVt@stages[0].B_matrix,\
            np.vstack([stages[0].C_matrix,c@stages[0].A_matrix]),
            np.vstack([stages[0].D_matrix,c@stages[0].B_matrix])),
        Stage(U[:stages[1].A_matrix.shape[0],:n],stages[1].B_matrix,\
          U[stages[1].A_matrix.shape[0]:,:n],stages[1].D_matrix[m_d:,:])
        ]
    return stages_d,s_d


def move_down_anticausal(stages_anti,m_d,d,epsilon = 1e-16):
    """
        Move left:
        function that moves the boundary down by m steps and transform it to output normal

        Args:
            stages (List[Stage]): List with two stages, boundary inbetween will be moved
            m_d (int):            Distance to move
            d (Matrix):       Matrix to be attached
            epsilon (float):      epsilon to cut states

        Returns:
            stages (List[Stage]): List with two stages with moved boundaries,
            s_d (List[float]):    Singular values
    """
    c = stages_anti[1].C_matrix[:m_d,:]
    #d = stages_causal[i].D_matrix[:m_d,:]
    d_add = np.zeros((m_d,stages_anti[0].D_matrix.shape[1]))
    #d_add = stages_causal[i].C_matrix@stages_causal[i-1].B_matrix[:,-1:]

    U,s_ad,Vt= np.linalg.svd(np.block([[stages_anti[1].A_matrix,stages_anti[1].B_matrix],
                                   [c,d]]),full_matrices=False)
    n = np.count_nonzero(s_ad>epsilon)
    Us=U[:,:n]*s_ad[:n]
    s_ad = s_ad[:n]
    stages_anti_d=[
     #Here the A and B are more complicated as we have to stack them
        Stage((np.hstack([stages_anti[0].A_matrix,np.zeros((stages_anti[0].A_matrix.shape[0],m_d))]))@Us,
          stages_anti[0].B_matrix,
          np.block([[stages_anti[0].C_matrix,np.zeros((stages_anti[0].C_matrix.shape[0],m_d))],
                    [np.zeros((m_d,stages_anti[0].C_matrix.shape[1])),np.eye(m_d)]])@Us,
          np.vstack([stages_anti[0].D_matrix,d_add]),copy=False),
        Stage(Vt[:n,:stages_anti[1].A_matrix.shape[1]],Vt[:n,stages_anti[1].A_matrix.shape[1]:],\
          stages_anti[1].C_matrix[m_d:,:],stages_anti[1].D_matrix[m_d:,:],copy=False)
        ]
    return stages_anti_d,s_ad

def transform_ud(stages_causal,stages_anticausal,cost,m=1,dir_preset = -1,epsilon=1e-15,
            sigmas_causal=None,sigmas_anticausal=None,cost_global=False,print_progress=False):
    k = len(stages_causal)
    if sigmas_causal is None:
        sigmas_causal = [np.array([np.nan])]*(k-1)
    else:
        sigmas_causal=sigmas_causal.copy()
    if sigmas_anticausal is None:
        sigmas_anticausal = [np.array([np.nan])]*(k-1)
    else:
        sigmas_anticausal=sigmas_anticausal.copy()

    for i in range(k-1, 0,-1):
        #no move-> only make O_k normal

        stages_n,s = transform_output_normal_causal([stages_causal[i-1],stages_causal[i]])
        stages_anti_n,s_a = transform_input_normal_anticausal([stages_anticausal[i-1],stages_anticausal[i]])
        #move down:
        if m>stages_causal[i].C_matrix.shape[0]:
            m_d=stages_causal[i].C_matrix.shape[0]
        else:
            m_d=m
        if m_d>0:
            stages_d,s_d = move_down_causal([stages_causal[i-1],stages_causal[i]],m_d)
            d = stages_causal[i].D_matrix[:m_d,:]
            stages_anti_d,s_ad = move_down_anticausal([stages_anticausal[i-1],stages_anticausal[i]],m_d,d)
        else:
            stages_d=stages_n
            s_d = s
            stages_anti_d=stages_anti_n
            s_ad = s_a

        #move up
        if m>stages_n[0].C_matrix.shape[0]:
            m_u=stages_n[0].C_matrix.shape[0]
        else:
            m_u=m
        if m_u>0:
            d_add = stages_anticausal[i-1].C_matrix[-m_u:,:]@stages_anticausal[i].B_matrix
            stages_u,s_u = move_up_causal(stages_n,m_u,d_add)
            stages_anti_u,s_au = move_up_anticausal(stages_anti_n,m_u)
        else:
            stages_u=stages_n
            s_u = s
            stages_anti_u=stages_anti_n
            s_au = s_a

        dims_in = [stage.D_matrix.shape[1] for stage in stages_causal]
        dims_out = [stage.D_matrix.shape[0] for stage in stages_causal]
        if cost_global:
            #create a vector to modify the output dims
            move_d = np.zeros_like(dims_out)
            move_d[i-1] = move_d[i-1]+m_d
            move_d[i] = move_d[i]-m_d

            move_u = np.zeros_like(dims_out)
            move_u[i-1] = move_u[i-1]-m_u
            move_u[i] = move_u[i]+m_u
            costs = np.array([cost(sigmas_causal,sigmas_anticausal,dims_in,dims_out),\
                            cost(sigmas_causal[:i-1]+[s_d]+sigmas_causal[i:],sigmas_anticausal[:i-1]+[s_ad]+sigmas_anticausal[i:],dims_in,dims_out+move_d),\
                            cost(sigmas_causal[:i-1]+[s_u]+sigmas_causal[i:],sigmas_anticausal[:i-1]+[s_au]+sigmas_anticausal[i:],dims_in,dims_out+move_u)])
        else:
            costs = np.array([cost(s,s_a),\
                            cost(s_d,s_ad),\
                            cost(s_u,s_au)])
        #print("costs_dnu:",costs)
        if dir_preset==-1:
            direction = np.argmin(costs)
        else:
            direction = dir_preset
            print("Move",direction)

        if direction == 1:
            stages_causal[i-1]= stages_d[0]
            stages_causal[i]= stages_d[1]
            stages_anticausal[i-1]= stages_anti_d[0]
            stages_anticausal[i]= stages_anti_d[1]
            sigmas_causal[i-1]=s_d
            sigmas_anticausal[i-1]=s_ad
        if direction == 0:
            stages_causal[i-1]= stages_n[0]
            stages_causal[i]= stages_n[1]
            stages_anticausal[i-1]= stages_anti_n[0]
            stages_anticausal[i]= stages_anti_n[1]
            sigmas_causal[i-1]=s
            sigmas_anticausal[i-1]=s_a
        if direction == 2:
            stages_causal[i-1]= stages_u[0]
            stages_causal[i]= stages_u[1]
            stages_anticausal[i-1]= stages_anti_u[0]
            stages_anticausal[i]= stages_anti_u[1]
            sigmas_causal[i-1]=s_u
            sigmas_anticausal[i-1]=s_au
        if print_progress:
            print("=",end = "")
    if print_progress:
        print("")
    return sigmas_causal,sigmas_anticausal

def move(system,N,cost,m_in = 1,m_out=1,cost_global= False,return_sigmas=False,print_progress = False):
    """

    parameters:

    N: number of iterations
    cost: function that calculates a cost term for the sigmas

    """

    """prepare system:
    get an input normal causal system
    get an output normal anticausal system and caculate the epsilons
    """

    if N is None:
        N = len(m_in)
    approx =Approximation(system)
    sys_move = approx.get_approxiamtion(1e-15)

    sigmas_causal = approx.sigmas_causal
    sigmas_anticausal = approx.sigmas_anticausal

    input_dims=np.zeros((len(sys_move.causal_system.stages),N+1))
    output_dims=np.zeros((len(sys_move.causal_system.stages),N+1))
    input_dims[:,0] = sys_move.dims_in
    output_dims[:,0] = sys_move.dims_out
    costs = np.zeros(N+1)
    if cost_global:
        costs[0]= cost(sigmas_causal,sigmas_anticausal,sys_move.dims_in,sys_move.dims_out)
    else:
        costs[0] = np.sum([cost(s,s_a)\
                    for (s,s_a) in zip(sigmas_causal,sigmas_anticausal)])
    try:
        for n in range(N):
            if type(m_in)==int:
                m_in_ = m_in
            else:
                m_in_ = m_in[n]
            if type(m_out)==int:
                m_out_ = m_out
            else:
                m_out_ = m_out[n]

            if print_progress:
                print("Starting iteration ",n+1)
            sigmas_causal,sigmas_anticausal=transform_ud(sys_move.causal_system.stages,
                                                     sys_move.anticausal_system.stages,cost,m=m_out_,cost_global= cost_global,
                                                     sigmas_causal=sigmas_causal,sigmas_anticausal=sigmas_anticausal,print_progress=print_progress)

            sigmas_causal,sigmas_anticausal=transform_rl(sys_move.causal_system.stages,
                                                     sys_move.anticausal_system.stages,cost,m=m_in_,cost_global= cost_global,
                                                     sigmas_causal=sigmas_causal,sigmas_anticausal=sigmas_anticausal,print_progress=print_progress)

            #compute objective function
            if cost_global:
                costs[n+1]= cost(sigmas_causal,sigmas_anticausal,sys_move.dims_in,sys_move.dims_out)
            else:
                costs[n+1] = np.sum([cost(s,s_a)\
                            for (s,s_a) in zip(sigmas_causal,sigmas_anticausal)])

            input_dims[:,n+1] = sys_move.dims_in
            output_dims[:,n+1] = sys_move.dims_out
            print("Dims_in: ",sys_move.dims_in)
            print("Dims_out:",sys_move.dims_out)
    except:
         traceback.print_exc()
    if return_sigmas:
        return sys_move,input_dims,output_dims,costs,(sigmas_causal,sigmas_anticausal)
    else:
        return sys_move,input_dims,output_dims,costs


def test_moves(system,m,epsilon=1e-15):
    approx =Approximation(system)
    sys_move = approx.get_approxiamtion(1e-15)

    assert sys_move.causal_system.is_input_normal(),"Causal system is not input normal"
    assert sys_move.anticausal_system.is_output_normal(),"Anticausal system is not input normal"

    sigmas_causal_ref = approx.sigmas_causal
    sigmas_anticausal_ref = approx.sigmas_anticausal

    #check if the inputs of the cost are corect. for this we comare the sigmas
    mat = sys_move.to_matrix()
    def cost_test(sigmas_causal_c,sigmas_anticausal_c,dims_in,dims_out):
        print("Dims:",dims_in,dims_out)
        (sigmas_causal_refer,sigmas_anticausal_refer) = math.extract_sigmas(mat, dims_in,dims_out)
        for i in range(len(sigmas_causal_refer)):
            sig_causal = np.zeros_like(sigmas_causal_refer[i])
            sig_anticausal = np.zeros_like(sigmas_anticausal_refer[i])
            sig_causal[:len(sigmas_causal_c[i])]=sigmas_causal_c[i]
            sig_anticausal[:len(sigmas_anticausal_c[i])]=sigmas_anticausal_c[i]
            assert np.allclose(sig_causal,sigmas_causal_refer[i]),\
                    "Causal sigmas do not match"+str(i)+str(sigmas_causal_c[i])+str(sigmas_causal_refer[i])
            assert np.allclose(sig_anticausal,sigmas_anticausal_refer[i]),\
                    "Anticausal sigmas do not match"+str(i)+str(sigmas_anticausal_c[i])+str(sigmas_anticausal_refer[i])
            #print("compare_sigmas",sig_causal,sigmas_causal_refer[i])
            #print("compare_sigmas_anti",sig_anticausal,sigmas_anticausal_refer[i])

        #print(sigmas_causal)
        #print(sigmas_anticausal)
        #print(np.abs(sig_causal_ref-sig_causal))
        #print(np.abs(sig_anticausal_ref-sig_anticausal))
        return 0

    input_dims = sys_move.dims_in
    output_dims = sys_move.dims_out

    print("testing move d")
    sys_move_d = sys_move.copy()
    sigmas_causal,sigmas_anticausal=transform_ud(sys_move_d.causal_system.stages,
                                                 sys_move_d.anticausal_system.stages,cost_test,m=m,dir_preset=1,epsilon=epsilon,cost_global=True,
                                                 sigmas_causal=sigmas_causal_ref,sigmas_anticausal=sigmas_anticausal_ref)
    assert utils.check_dims(sys_move_d),"dims incorrect"
    print(np.max(np.abs(sys_move.to_matrix()-sys_move_d.to_matrix())))
    assert np.allclose(sys_move.to_matrix(),sys_move_d.to_matrix()), "Marix is different"

    print("testing move u")
    sys_move_u = sys_move.copy()
    sigmas_causal,sigmas_anticausal=transform_ud(sys_move_u.causal_system.stages,
                                                 sys_move_u.anticausal_system.stages,cost_test,m=m,dir_preset=2,epsilon=epsilon,cost_global=True,
                                                 sigmas_causal=sigmas_causal_ref,sigmas_anticausal=sigmas_anticausal_ref)
    assert utils.check_dims(sys_move_u),"dims incorrect"
    print(np.max(np.abs(sys_move.to_matrix()-sys_move_u.to_matrix())))
    assert np.allclose(sys_move.to_matrix(),sys_move_u.to_matrix()), "Marix is different"

    print("testing move none")
    sys_move_n = sys_move.copy()
    sigmas_causal,sigmas_anticausal=transform_ud(sys_move_n.causal_system.stages,
                                                 sys_move_n.anticausal_system.stages,cost_test,m=m,dir_preset=0,epsilon=epsilon,cost_global=True,
                                                 sigmas_causal=sigmas_causal_ref,sigmas_anticausal=sigmas_anticausal_ref)
    assert utils.check_dims(sys_move_n),"dims incorrect"
    print(np.max(np.abs(sys_move.to_matrix()-sys_move_n.to_matrix())))
    assert np.allclose(sys_move.to_matrix(),sys_move_n.to_matrix()), "Marix is different"

    print("compare_sigmas")
    for i in range(len(sigmas_causal)):
        print(np.max(np.abs(sigmas_causal[i]-sigmas_causal_ref[i]),initial=0))
        print(np.max(np.abs(sigmas_anticausal[i]-sigmas_anticausal_ref[i]),initial=0))

    print("testing_move l")
    sys_move_l = sys_move_n.copy()
    sigmas_causal,sigmas_anticausal=transform_rl(sys_move_l.causal_system.stages,
                                                 sys_move_l.anticausal_system.stages,cost_test,m=m,dir_preset=1,epsilon=epsilon,cost_global=True,
                                                 sigmas_causal=sigmas_causal_ref,sigmas_anticausal=sigmas_anticausal_ref)
    assert utils.check_dims(sys_move_l),"dims incorrect"
    print(np.max(np.abs(sys_move.to_matrix()-sys_move_l.to_matrix())))
    assert np.allclose(sys_move.to_matrix(),sys_move_l.to_matrix()), "Marix is different"

    print("testing_move r")
    sys_move_r = sys_move_n.copy()
    sigmas_causal,sigmas_anticausal=transform_rl(sys_move_r.causal_system.stages,
                                                 sys_move_r.anticausal_system.stages,cost_test,m=m,dir_preset=2,epsilon=epsilon,cost_global=True,
                                                 sigmas_causal=sigmas_causal_ref,sigmas_anticausal=sigmas_anticausal_ref)
    assert utils.check_dims(sys_move_r),"dims incorrect"
    print(np.max(np.abs(sys_move.to_matrix()-sys_move_r.to_matrix())))
    assert np.allclose(sys_move.to_matrix(),sys_move_r.to_matrix()), "Marix is different"


    print("testing_move nn")
    sys_move_nn = sys_move_n.copy()
    sigmas_causal,sigmas_anticausal=transform_rl(sys_move_nn.causal_system.stages,
                                                 sys_move_nn.anticausal_system.stages,cost_test,m=m,dir_preset=0,epsilon=epsilon,cost_global=True,
                                                 sigmas_causal=sigmas_causal_ref,sigmas_anticausal=sigmas_anticausal_ref)
    assert utils.check_dims(sys_move_n),"dims incorrect"
    print(np.max(np.abs(sys_move.to_matrix()-sys_move_n.to_matrix())))
    assert np.allclose(sys_move.to_matrix(),sys_move_n.to_matrix()), "Marix is different"
