# -*- coding: utf8 -*-
from matplotlib import pyplot as plt
import numpy as np
import sys
import math

if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 6))
    num_plots = 1
    ax1 = fig.add_axes([0.1, 0.1/num_plots,
                        0.8, 0.8/num_plots])
    ax1.set_xlabel(u"Ereignisse (µs⁻¹)")
    ax1.set_ylabel(u"Häufigkeit")
    for filename in sys.argv[1:]:
        data = []
        with open(filename) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                data_tuple = map(float, line.split())
                data.append(data_tuple[1]/data_tuple[2]*1000)
                #data.append(data_tuple[1]/data_tuple[2])
        n_bins = 30
        hist, bin_start, = np.histogram(data, bins=n_bins, normed=1)
        hist = np.array(hist) * (np.array(bin_start[1:]) - np.array(bin_start[:-1]))
        ax1.plot(bin_start[:-1], hist, "x--", label=filename)
    #ax.set_xbound(lower=1100, upper=1300)
    ax1.legend(loc=1)
    fig.savefig("count_rate.pdf")
