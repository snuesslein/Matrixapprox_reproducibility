import numpy as np

def calculate_Laplacian(A,normalize = True):
    L = -np.abs(A.T@A) 
    # Set diagonal
    np.fill_diagonal(L,0)
    np.fill_diagonal(L,-np.sum(L,axis=0))

    if normalize:
        #normalize L
        d = np.sqrt(1/np.diag(L))
        L = d.reshape(1,-1)*L*d.reshape(-1,1)
    return L


def segment_matrix(A,normalize = True,print_info=False):
    """
    Matrix 
    
    returns
        s: boolena vector, True if collumn in second part 
    """
    
    L = calculate_Laplacian(A,normalize = normalize)

    w, v = np.linalg.eig(L)

    #order eigenvalues
    o = np.argsort(w.real)
    v = v[:,o]
    if print_info:
        print("Eigenvalues:",w[o])
        print("Fiedler-Vector",v[:,1])
        
    return v[:,1].real>0