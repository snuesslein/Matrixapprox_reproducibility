import numpy as np
import matplotlib.pyplot as plt


def invert_permutation(P):
    P_inv = np.zeros_like(P)
    for i in range(len(P)):
        P_inv[P[i]]=i
    return P_inv

def invert_permutations(Ps):
    Ps_inv = np.zeros_like(Ps)
    for (j,i), x in np.ndenumerate(Ps):
        Ps_inv[j,Ps[j,i]]=i
    return Ps_inv


def connection_plot(P1_inv,P2_inv,start=0,end=1,colors = None,flipxy=False,linewidth=None,ax=None,N=100):
    #cosine
    #v = np.linspace(0,1)
    #y = (np.cos(np.pi*v)*0.5)+0.5
    #x = v*(end-start)+start

    #bezier
    t = np.linspace(0,1,N)
    a =-t**3+3*t**2-3*t+1
    b =3*t**3-6*t**2+3*t
    c =-3*t**3+3*t**2
    d =t**3

    x = 0*a +0.5*b +0.5*c +1*d
    y = 1*a +  1*b +  0*c +0*d

    fraction_straight = 0.05

    x = np.hstack([0,((1-2*fraction_straight)*x)+fraction_straight,1])
    y = np.hstack([1,y,0])

    x = x*(end-start)+start


    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if linewidth is None:
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        if flipxy:
            linewidth=width/P1_inv.shape[0]*fig.dpi
        else:
            linewidth=height/P1_inv.shape[0]*fig.dpi

    if colors is None:
        #cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']
        #colors = cycler*int(np.ceil(len(P1_inv)/len(cycler)))
        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0,1,len(P1_inv)))
    elif type(colors)==str:
        cmap = plt.cm.get_cmap(colors)
        colors = cmap(np.linspace(0,1,len(P1_inv)))
        #colors = [colors]*len(P1_inv)
    for p1,p2,c in zip(P1_inv,P2_inv,colors):
        if flipxy:
            ln, =ax.plot(y*(p1-p2)+p2,x,color = c,linewidth=linewidth,linestyle='-')
        else:
            ln, =ax.plot(x,y*(p1-p2)+p2,color = c,linewidth=linewidth,linestyle='-')
        ln.set_solid_capstyle('butt')

def multiple_connection_plot(Ps_inv,start=0,end=1,ax=None,**kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    N = Ps_inv.shape[0]
    d = (end-start)/(N-1)
    for n in range(N-1):
        connection_plot(Ps_inv[n],Ps_inv[n+1],start=start+d*n,end=start+d*(n+1),ax=ax,**kwargs)
