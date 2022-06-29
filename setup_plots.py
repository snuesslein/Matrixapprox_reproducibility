import matplotlib.pyplot as plt

from cycler import cycler


markers = ["4", "2", "3", "1", "+", "x", "."]
textwidth = 369.88588 * 0.0138 

def setup():
    line_cycler   = (cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color']) +
                 cycler(linestyle=["-", "--", ":", "-.", "-", "--", "-.", ":", "-", "--",]))
    plt.rc("axes", prop_cycle=line_cycler)

    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}\usepackage{helvet}")
    #\usepackage{helvet}\usepackage[cm]{sfmath}
    plt.rc("font", family="sans-serif", size=10.)
    plt.rc("savefig", dpi=200)
    plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
    plt.rc("lines", linewidth=2, markersize=8, markeredgewidth=2)
    plt.rcParams['figure.dpi'] = 100
