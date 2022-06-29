import numpy as np
import tvsclib.utils as utils
import Split
import matplotlib.pyplot as plt
from tvsclib.strict_system import StrictSystem

from tvsclib.approximation import Approximation

import torchvision.models as models
import torch
import scipy.stats

import graphs

import scipy.linalg as linalg

def get_permutations(s_c,s_r):
    p_col = np.hstack([np.nonzero(s_c),np.nonzero(~s_c)]).reshape(-1)
    p_row = np.hstack([np.nonzero(~s_r),np.nonzero(s_r)]).reshape(-1)
    i_in =np.count_nonzero(s_c)
    i_out=np.count_nonzero(~s_r)
    return p_col,p_row,i_in,i_out

def permute_stage(stage,p_col,p_row):
    stage.B_tilde = stage.B_tilde[:,p_col]
    stage.C_tilde = stage.C_tilde[p_row,:]
    stage.D_matrix = stage.D_matrix[:,p_col][p_row]

def collect_permutations(P_col,P_row,k,p_col,p_row,system):
    """
    Function to collect the permutations in P_col and P_row

    P_col:    total permutation of columns
    P_row:    total permutation of columns
    l:        index of stage
    p_col:    new collumn permuation
    p_row:    new row permutation
    """

    dims_in = system.dims_in
    dims_out = system.dims_out

    I = np.sum(dims_in[:k]).astype(int)
    P_col[I:I+dims_in[k]]=P_col[I:I+dims_in[k]][p_col]
    I = np.sum(dims_out[:k]).astype(int)
    P_row[I:I+dims_out[k]]=P_row[I:I+dims_out[k]][p_row]


def segment_matrix_rank(stage_causal,stage_anticausal,N=70,initla_spectral=True,\
    gamma = 1e6):

    f_reg_row = lambda x: -gamma*x**3
    f_reg_col = lambda x: -gamma*x**3

    Ac = stage_causal.A_matrix
    Bc = stage_causal.B_matrix
    Cc = stage_causal.C_matrix

    Aa = stage_anticausal.A_matrix
    Ba = stage_anticausal.B_matrix
    Ca = stage_anticausal.C_matrix

    D = stage_causal.D_matrix

    #regularization vector
    v_reg_col = f_reg_col(np.linspace(-1,1,D.shape[1]))
    v_reg_row = f_reg_row(np.linspace(-1,1,D.shape[0]))

    #dims of states
    (d_out_c,d_in_c) = Ac.shape
    (d_out_a,d_in_a) = Aa.shape

    #setup matrix
    X = np.block([[np.zeros((d_out_a,d_in_c)),Ba,Aa ],
                  [Cc,D,Ca],
                  [Ac,Bc,np.zeros((d_out_c,d_in_a))]

    ])

    if initla_spectral:
        #get initial using spectral
        s_c = graphs.segment_matrix(X)
        if np.count_nonzero(~s_c[:d_in_c])+np.count_nonzero(s_c[X.shape[1]-d_in_a:]) > (d_in_a+d_in_c)/2:
            # if more fixed nodes are incorrect flip
            s_c = ~s_c

        #set the fixed
        s_c[:d_in_c]=1
        s_c[X.shape[1]-d_in_a:]=0

        s_r = graphs.segment_matrix(X.T)
        if np.count_nonzero(s_r[:d_out_a])+np.count_nonzero(~s_r[X.shape[0]-d_out_c:]) > (d_out_a+d_out_c)/2:
            # if more fixed nodes are incorrect flip
            s_r = ~s_r
        s_r[:d_out_a]=0
        s_r[X.shape[0]-d_out_c:]=1
    else:
        #initialize segmentation
        s_c = np.zeros(X.shape[1],dtype=bool)
        s_c[:d_in_c+D.shape[1]//2]=1
        s_r = np.zeros(X.shape[0],dtype=bool)
        s_r[d_out_a+D.shape[0]//2:]=1


    #fs = np.zeros(N+1)
    s_cols=np.zeros((N+1,X.shape[1]),dtype=bool)
    s_rows=np.zeros((N+1,X.shape[0]),dtype=bool)


    s_cols[0]=s_c
    s_rows[0]=s_r
    #fs[0]=f(X,s_c,s_r)

    for n in range(N):
        #for test
        normalize ="F"

        #columns:
        X_t = X[~s_r]
        X_b = X[s_r]

        n_xt =np.linalg.norm(X_t,axis=0)
        n_xb =np.linalg.norm(X_b,axis=0)

        #weights unnormalized
        W_t = n_xt.reshape(-1,1)*n_xt.reshape(1,-1)-np.abs(X_t.T@X_t)
        W_b = n_xb.reshape(-1,1)*n_xb.reshape(1,-1)-np.abs(X_b.T@X_b)

        #nomalize
        if normalize =="F": #Forbenius norm
            W_t = W_t/(n_xt.reshape(-1,1)*n_xt.reshape(1,-1))
            W_b = W_b/(n_xb.reshape(-1,1)*n_xb.reshape(1,-1))
        elif normalize =="L": #length
            W_t = W_t/(np.count_nonzero(~s_r)**2)
            W_b = W_b/(np.count_nonzero(s_r)**2)
        elif normalize =="M": #mixed: reference with norm, current with length
            W_t = W_t/(n_xt.reshape(-1,1)*np.count_nonzero(~s_r))#
            W_b = W_b/(n_xb.reshape(-1,1)*np.count_nonzero(s_r))

        #rows:
        X_r = X[:,~s_c]
        X_l = X[:,s_c]

        n_xr = np.linalg.norm(X_r,axis=1)
        n_xl = np.linalg.norm(X_l,axis=1)

        #Weights unnormalized
        W_r =n_xr.reshape(-1,1)*n_xr.reshape(1,-1)-np.abs(X_r@X_r.T)
        W_l =n_xl.reshape(-1,1)*n_xl.reshape(1,-1)-np.abs(X_l@X_l.T)

        #normlaize
        if normalize =="F": #Forbenius norm
            W_r = W_r/(n_xr.reshape(-1,1)*n_xr.reshape(1,-1))
            W_l = W_l/(n_xl.reshape(-1,1)*n_xl.reshape(1,-1))
        elif normalize =="L": #length
            W_r = W_r/(np.count_nonzero(~s_c)**2)#
            W_l = W_l/(np.count_nonzero(s_c)**2)
        elif normalize =="M": #mixed: reference with norm, current with length
            W_r = W_r/(n_xr.reshape(-1,1)*np.count_nonzero(~s_c))#
            W_l = W_l/(n_xl.reshape(-1,1)*np.count_nonzero(s_c))

        S_col = np.sum(W_t[~s_c],axis=0) -np.sum(W_b[s_c],axis=0)
        S_row = np.sum(W_r[~s_r],axis=0) -np.sum(W_l[s_r],axis=0)

        #ord_c = d_in_c +np.argsort(S_col[d_in_c:X.shape[1]-d_in_a])
        #ord_r = d_out_a+np.argsort(S_row[d_out_a:X.shape[0]-d_out_c])


        #only the ones we can change
        s_c_int = s_c[d_in_c:X.shape[1]-d_in_a]
        s_r_int = s_r[d_out_a:X.shape[0]-d_out_c]

        S_col_int = S_col[d_in_c:X.shape[1]-d_in_a]
        S_row_int = S_row[d_out_a:X.shape[0]-d_out_c]

        i_n = -1
        i_p = -1
        v_reg_ac = gamma*(((np.count_nonzero(s_c_int)+1)/len(s_c_int)-0.5)**2\
                                -((np.count_nonzero(s_c_int))/len(s_c_int)-0.5)**2)
        v_reg_ca = gamma*(((np.count_nonzero(s_c_int)-1)/len(s_c_int)-0.5)**2\
                                -((np.count_nonzero(s_c_int))/len(s_c_int)-0.5)**2)
        if np.any(S_col_int[s_c_int]+v_reg_ca<0):
            i_n= np.nonzero(s_c_int)[0][np.argmin(S_col_int[s_c_int])]
            s_c[d_in_c+i_n]=0
        if np.any(S_col_int[~s_c_int]-v_reg_ac>0):
            i_p= np.nonzero(~s_c_int)[0][np.argmax(S_col_int[~s_c_int])]
            s_c[d_in_c+i_p]=1
        if i_n==-1 and i_p==-1:
            print("flip n =",n)
            i_n= np.nonzero(s_c_int)[0][np.argmin(S_col_int[s_c_int])]
            s_c[d_in_c+i_n]=0
            i_p= np.nonzero(~s_c_int)[0][np.argmax(S_col_int[~s_c_int])]
            s_c[d_in_c+i_p]=1

        i_n = -1
        i_p = -1
        v_reg_ac = gamma*(((np.count_nonzero(s_r_int)+1)/len(s_r_int)-0.5)**2\
                                -((np.count_nonzero(s_r_int))/len(s_r_int)-0.5)**2)
        v_reg_ca = gamma*(((np.count_nonzero(s_r_int)-1)/len(s_r_int)-0.5)**2\
                                -((np.count_nonzero(s_r_int))/len(s_r_int)-0.5)**2)
        if np.any(S_row_int[s_r_int]+v_reg_ca<0):
            i_n = np.nonzero(s_r_int)[0][np.argmin(S_row_int[s_r_int])]
            s_r[d_out_a+i_n]=0
        if np.any(S_row_int[~s_r_int]-v_reg_ac>0):
            i_p = np.nonzero(~s_r_int)[0][np.argmax(S_row_int[~s_r_int])]
            s_r[d_out_a+i_p]=1
        if i_n==-1 and i_p==-1:
            print("flip n =",n)
            i_n = np.nonzero(s_r_int)[0][np.argmin(S_row_int[s_r_int])]
            s_r[d_out_a+i_n]=0
            i_p = np.nonzero(~s_r_int)[0][np.argmax(S_row_int[~s_r_int])]
            s_r[d_out_a+i_p]=1

        #v_c = ord_c[v_reg_col<S_col[ord_c]]
        #v_r = ord_r[v_reg_row<S_row[ord_r]]

        #s_c[d_in_c:s_c.size-d_in_a]=0
        #s_r[d_out_a:s_r.size-d_out_c]=0

        #s_c[v_c] = 1
        #s_r[v_r] = 1


        #fs[n+1] = f(X,s_c,s_r)
        s_cols[n+1]=s_c
        s_rows[n+1]=s_r

        if np.any(np.logical_and(np.all(s_cols[:n+1]==s_c,axis=1),np.all(s_rows[:n+1]==s_r,axis=1))):
            print("converged at n=",n)
            break

    report ={"s_cols":s_cols[:n+2],"s_rows":s_rows[:n+2],"X":X}

    return s_c[d_in_c:X.shape[1]-d_in_a],s_r[d_out_a:X.shape[0]-d_out_c],report

def segment_matrix_frob(stage_causal,stage_anticausal,N=70,gamma = 1e1): #minimize frobenius norm
    Ac = stage_causal.A_matrix**2
    Bc = stage_causal.B_matrix**2
    Cc = stage_causal.C_matrix**2

    Aa = stage_anticausal.A_matrix**2
    Ba = stage_anticausal.B_matrix**2
    Ca = stage_anticausal.C_matrix**2

    D = stage_causal.D_matrix**2


    (d_out_c_i,d_in_c_i) = Ac.shape
    (d_out_a_i,d_in_a_i) = Aa.shape

    Ac = np.array([[np.sum(Ac)]])
    Bc = np.sum(Bc,axis=0)
    Cc = np.sum(Cc,axis=1)
    Aa = np.array([[np.sum(Aa)]])
    Ba = np.sum(Ba,axis=0)
    Ca = np.sum(Ca,axis=1)

    s = np.sum(Cc)
    if s ==0:
        Cc = np.zeros((Cc.size,1))
    else:
        Cc = Cc.reshape(-1,1)

    s = np.sum(Ca)
    if s ==0:
        Ca = np.zeros((Ca.size,1))
    else:
        Ca = Ca.reshape(-1,1)

    s = np.sum(Bc)
    if s ==0:
        Bc = np.zeros((1,Bc.size))
    else:
        Bc = Bc.reshape(1,-1)

    s = np.sum(Ba)
    if s ==0:
        Ba = np.zeros((1,Ba.size))
    else:
        Ba = Ba.reshape(1,-1)

    D = D*(D.size/np.sum(D))

    #dims of states
    (d_out_c,d_in_c) = Ac.shape
    (d_out_a,d_in_a) = Aa.shape

    #setup matrix
    Xs = np.block([[np.zeros((1,1)),Ba,Aa ],
                  [Cc,D,Ca],
                  [Ac,Bc,np.zeros((1,1))]

    ])

    s_c = np.zeros(Xs.shape[1],dtype=bool)
    s_r = np.zeros(Xs.shape[0],dtype=bool)

    #initialize based on the Bs and Cs
    order = np.argsort(np.linalg.norm(Bc,axis=0)-np.linalg.norm(Ba,axis=0))
    s_c[d_in_c+order[:len(order)//2]]=1
    order = np.argsort(np.linalg.norm(Cc,axis=1)-np.linalg.norm(Ca,axis=1))
    s_r[d_out_a+order[:len(order)//2]]=1


    #set the fixed
    s_c[:d_in_c]=1
    s_c[Xs.shape[1]-d_in_a:]=0
    s_r[:d_out_a]=0
    s_r[Xs.shape[0]-d_out_c:]=1



    fs = np.zeros(N+1)
    s_cols=np.zeros((N+1,Xs.shape[1]),dtype=bool)
    s_rows=np.zeros((N+1,Xs.shape[0]),dtype=bool)


    #Xs = X**2

    s_cols[0]=s_c
    s_rows[0]=s_r
    fs[0]=np.sum(Xs[s_r][:,s_c])+ np.sum(Xs[~s_r][:,~s_c])

    q = int(np.ceil(min(D.shape)/1e2))
    n_restart = -1e5

    gamma_prime = np.mean(D.shape)*gamma
    #gamma_prime = D.size*gamma
    print("g_prime =",gamma_prime)
    print("gamma = ",gamma)

    for n in range(N):

        #columns:
        n_xt = np.sum(Xs[~s_r],axis=0)
        n_xb = np.sum(Xs[s_r],axis=0)

        #rows:
        n_xr = np.sum(Xs[:,~s_c],axis=1)
        n_xl = np.sum(Xs[:,s_c],axis=1)

        #only the ones we can change
        s_c_int = s_c[d_in_c:Xs.shape[1]-d_in_a]
        s_r_int = s_r[d_out_a:Xs.shape[0]-d_out_c]

        S_col = n_xt -n_xb
        S_row = n_xr -n_xl

        S_col_int = S_col[d_in_c:Xs.shape[1]-d_in_a]
        S_row_int = S_row[d_out_a:Xs.shape[0]-d_out_c]



        if q ==1:
            i_n = -1
            i_p = -1
            v_reg_ac = gamma_prime*(((np.count_nonzero(s_c_int)+1)/len(s_c_int)-0.5)**2\
                                    -((np.count_nonzero(s_c_int))/len(s_c_int)-0.5)**2)
            v_reg_ca = gamma_prime*(((np.count_nonzero(s_c_int)-1)/len(s_c_int)-0.5)**2\
                                    -((np.count_nonzero(s_c_int))/len(s_c_int)-0.5)**2)
            if np.any(S_col_int[s_c_int]+v_reg_ca<0):
                i_n= np.nonzero(s_c_int)[0][np.argmin(S_col_int[s_c_int])]
                s_c[d_in_c+i_n]=0
            if np.any(S_col_int[~s_c_int]-v_reg_ac>0):
                i_p= np.nonzero(~s_c_int)[0][np.argmax(S_col_int[~s_c_int])]
                s_c[d_in_c+i_p]=1
            if i_n==-1 and i_p==-1:
                print("flip n =",n)
                i_n= np.nonzero(s_c_int)[0][np.argmin(S_col_int[s_c_int])]
                s_c[d_in_c+i_n]=0
                i_p= np.nonzero(~s_c_int)[0][np.argmax(S_col_int[~s_c_int])]
                s_c[d_in_c+i_p]=1

            i_n = -1
            i_p = -1
            v_reg = gamma_prime*(np.count_nonzero(s_r_int)/len(s_r_int)-0.5)**3
            v_reg_ac = gamma_prime*(((np.count_nonzero(s_r_int)+1)/len(s_r_int)-0.5)**2\
                                    -((np.count_nonzero(s_r_int))/len(s_r_int)-0.5)**2)
            v_reg_ca = gamma_prime*(((np.count_nonzero(s_r_int)-1)/len(s_r_int)-0.5)**2\
                                    -((np.count_nonzero(s_r_int))/len(s_r_int)-0.5)**2)
            if np.any(S_row_int[s_r_int]+v_reg_ca<0):
                i_n = np.nonzero(s_r_int)[0][np.argmin(S_row_int[s_r_int])]
                s_r[d_out_a+i_n]=0
            if np.any(S_row_int[~s_r_int]-v_reg_ac>0):
                i_p = np.nonzero(~s_r_int)[0][np.argmax(S_row_int[~s_r_int])]
                s_r[d_out_a+i_p]=1
            if i_n==-1 and i_p==-1:
                print("flip n =",n)
                i_n = np.nonzero(s_r_int)[0][np.argmin(S_row_int[s_r_int])]
                s_r[d_out_a+i_n]=0
                i_p = np.nonzero(~s_r_int)[0][np.argmax(S_row_int[~s_r_int])]
                s_r[d_out_a+i_p]=1


        else:
            v_reg = gamma_prime*(np.count_nonzero(s_c_int)/len(s_c_int)-0.5)**3

            #if np.any(S_col_int[s_c_int]<v_reg):
            i_n= np.arange(len(s_c_int))[s_c_int][np.argsort(S_col_int[s_c_int])[:q]]#arange ist trick to recover index
            i_nf = i_n[S_col_int[i_n]<v_reg]
            s_c[d_in_c+i_nf]=0
            #if np.any(S_col_int[~s_c_int]>v_reg):
            i_p= np.arange(len(s_c_int))[~s_c_int][np.argsort(S_col_int[~s_c_int])[-q:]]
            i_pf = i_p[S_col_int[i_p]>v_reg]
            s_c[d_in_c+i_pf]=1

            if len(i_nf)==0 and len(i_pf)==0:
                print("flip")
                s_c[d_in_c+i_n[0]]=0
                s_c[d_in_c+i_p[-1]]=1

            v_reg = gamma_prime*(np.count_nonzero(s_r_int)/len(s_r_int)-0.5)**3
            #if np.any(S_row_int[s_r_int]<v_reg):
            i_n = np.arange(len(s_r_int))[s_r_int][np.argsort(S_row_int[s_r_int])[:q]]
            i_nf = i_n[S_row_int[i_n]<v_reg]
            s_r[d_out_a+i_nf]=0
            #if np.any(S_row_int[~s_r_int]>v_reg):
            i_p = np.arange(len(s_r_int))[~s_r_int][np.argsort(S_row_int[~s_r_int])[-q:]]
            i_pf = i_p[S_row_int[i_p]>v_reg]
            s_r[d_out_a+i_pf]=1

            if len(i_nf)==0 and len(i_pf)==0:
                print("flip n =",n)
                s_r[d_out_a+i_n[0]]=0
                s_r[d_out_a+i_p[-1]]=1


        f = np.sum(Xs[s_r][:,s_c])+ np.sum(Xs[~s_r][:,~s_c])
        if False:#f > fs[0] and n > n_restart + 50: #worse than initial -> do restart with other initial
            print("restart at n=",n)
            n_restart= n
            i_min = np.argmin(fs[:n+1])
            s_c = s_cols[i_min].copy()
            s_r = s_rows[i_min].copy()
            v = np.random.randint(d_in_c,X.shape[1]-d_in_a,3*q)
            s_c[v] = ~s_c[v]
            print(v)
            v = np.random.randint(d_out_a,X.shape[0]-d_out_c,3*q)
            print(v)
            s_r[v] = ~s_r[v]

        fs[n+1] = np.sum(Xs[s_r][:,s_c])+ np.sum(Xs[~s_r][:,~s_c])
        s_cols[n+1]=s_c
        s_rows[n+1]=s_r

        if np.any(np.logical_and(np.all(s_cols[:n+1]==s_c,axis=1),np.all(s_rows[:n+1]==s_r,axis=1))):
            print("converged at n=",n)
            break

    #get minimum f
    i_min = np.argmin(fs[:n+2])
    s_c = s_cols[i_min]
    s_r = s_rows[i_min]
    print("frac cols:",np.count_nonzero(s_c[d_in_c:Xs.shape[1]-d_in_a])/D.shape[1])
    print("frac rows:",np.count_nonzero(s_r[d_out_a:Xs.shape[0]-d_out_c])/D.shape[0])

    report ={"s_cols":s_cols[:n+2],"s_rows":s_rows[:n+2],"X":0,"f":fs[:n+2],"q":q}

    return s_c[d_in_c:Xs.shape[1]-d_in_a],s_r[d_out_a:Xs.shape[0]-d_out_c],report




def identification_split_permute(T,N,epsilon=1e-12,canonical_form=None,compute_sigmas=False,\
    strategy = "rank",opts = {}):
    """identify creates a mixed system to represent the matrix T

    Args:
        T (np.ndarray):                     Transfer Operator
        N (int):                            number of splittings. Final system will have 2^N stages
        epsilon (float,optional):           epsilon for balanceed truncation, Default is 1e-15
        canonical_form (tvsclib.CanonicalForm,optional):  Cannonical form of system, Default is None
        compute_sigmas (bool,optional):     If True, the function returns the Hankel singular values, default is False
        strategy  "rank" or "fro":          Solver to use for clustering
        opts  (dict):                       options for clustering algorithm
    Returns:
        system (StrictSystem):              Time varying system
        P_in:  (np.ndarray):                Permuations of inputs
        P_out: (np.ndarray):                Permuatations of outputs
        reports: ([dicts]):                 array with reports form clustering algorithm
        sigmas ():                          Tupels with sigmas for causal and anticasual Hankel matrices
    """
    sys = Split.initial_sigmas_mixed(T)
    P_col = np.arange(np.sum(sys.dims_in) ,dtype=int)
    P_row = np.arange(np.sum(sys.dims_out),dtype=int)

    Ps_col =np.zeros((N+1,P_col.size),dtype=int)
    Ps_row =np.zeros((N+1,P_row.size),dtype=int)
    Ps_col[0]=P_col
    Ps_row[0]=P_row
    reports = []
    for n in range(N):
        print(n)
        if type(opts) == list:
            opt = opts[n]
        else:
            opt = opts
        for k in range(len(sys.causal_system.stages)-1,-1,-1): #reverse ordering makes indexing easier
            stage_c=sys.causal_system.stages[k]
            stage_a=sys.anticausal_system.stages[k]
            if strategy == "rank":
                s_c,s_r,report = segment_matrix_rank(stage_c,stage_a,**opt)
            elif strategy == "fro":
                s_c,s_r,report = segment_matrix_frob(stage_c,stage_a,**opt)
            else:
                raise ValueError('Unknown strategy use "rank" or "fro"')
            reports.append(report)

            assert len(s_c)==stage_c.dim_in ,"dims_in causal do not match s_c"
            assert len(s_r)==stage_c.dim_out,"dims_out causal do not match s_r"
            assert len(s_c)==stage_a.dim_in ,"dims_in antic do not match s_c"
            assert len(s_r)==stage_a.dim_out,"dims_out antic do not match s_r"
            p_col,p_row,i_in,i_out = get_permutations(s_c,s_r)
            permute_stage(stage_c,p_col,p_row)
            permute_stage(stage_a,p_col,p_row)
            collect_permutations(P_col,P_row,k,p_col,p_row,sys)

            Split.split_sigmas_mixed(sys,k,i_in,i_out)
        #save the Permutations collected for all stages
        Ps_col[n+1]=P_col
        Ps_row[n+1]=P_row

    return sys,Ps_col,Ps_row,reports
