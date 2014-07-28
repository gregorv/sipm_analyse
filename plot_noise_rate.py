# -*- coding: utf8 -*-
from matplotlib import pyplot as plt
import numpy as np
import sys
import math

if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 12))
    num_plots = 2
    ax1 = fig.add_axes([0.1, 0.1/num_plots,
                        0.8, 0.8/num_plots])
    ax1.set_xlabel(u"Zeitabstand (ns)")
    ax1.set_ylabel(u"Zählrate")
    ax1.set_yscale("log")
    ax2 = fig.add_axes([0.1, 0.1/num_plots + 1.0/num_plots,
                        0.8, 0.8/num_plots])
    ax2.set_xlabel(u"Zeitabstand (ns)")
    ax2.set_ylabel(u"Zählrate")
    ax2.set_yscale("log")
    #ax3 = fig.add_axes([0.1, 0.1/3 + 2.0/3, 0.8, 0.8/3 + 2.0/3])
    #ax3.set_xlabel(u"Zeitabstand (ns)")
    #ax3.set_ylabel(u"Rel. Häufigkeit")
    #ax3.set_yscale("log")
    ##n_bins = 10**int(math.log10(len(data)) - 1)
    for filename in sys.argv[1:]:
        data = []
        with open(filename) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                if float(line.split()[1]) > -15.0:
                    continue
                if float(line.split()[0]) > 1100:
                    continue
                if float(line.split()[0]) < 50:
                    continue
                data.append(float(line.split()[0]))
        n_bins = 1000 if len(data) > 3000 else 50
        hist, bin_start, = np.histogram(data, bins=n_bins)
        hist = np.array(hist) * (np.array(bin_start[1:]) - np.array(bin_start[:-1]))
        ax1.plot(bin_start[:-1], hist, "x", label=filename)
        ax2.plot(bin_start[:-1], hist, "x", label=filename)
    ax1.set_xbound(lower=0, upper=200)
    #ax.set_xbound(lower=1100, upper=1300)
    ax1.legend(loc=3)
    fig.savefig("time_difference_histogram.pdf")
