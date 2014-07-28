# -*- coding: utf8 -*-
from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import factorial
import sys
import math
import re
import datafile
import pulse
from matplotlib import rc
from plot import info_box

#rc('text', usetex=True)
#rc('font', family='serif')

event_filter = lambda dt, amplitude: abs(amplitude) > 0.0

def import_file(file):
    name = None
    data = {}
    for line in map(str.strip, file):
        if not line:
            continue
        split = map(str.strip, line.split("\t"))
        if len(split) == 1:
            name = split[0]
            data[name] = []
        else:
            data[name].append(map(float, split))
    return data

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data_set = import_file(f)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"Temperatur / $\mathrm{{}^\circ\!C}$")
    ax.set_ylabel(r"Ereignissfrequenz / $\mathrm{us}^{-1}$")
    style = ["b^", "go", "rx", "bv"]
    for nth_plot, (label, data) in enumerate(data_set.iteritems()):
        T, rate, sigma_P, sigma_var = map(np.array, zip(*data))
        #ax.errorbar(T - 273.15, total_rate, fmt=style[nth_plot], label=label, yerr=sample_sigma)
        ax.plot(T - 273.15, rate, style[nth_plot], label=label)
    #ax.logy()
    #ax.grid(which="minor", axis="y")
    ax.legend()
    ax.set_xbound(10, 22)
    fig.savefig("countrate_temperaturabhaengigkeit.pdf")
