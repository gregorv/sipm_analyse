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

if __name__ == "__main__":
    fname2temps = dict((fname,
                        datafile.userheader2dict(datafile.import_csv_header(fname))["T_soll_f"])
                       for fname in sys.argv[1:])
    print fname2temps
    num_plots = len(sys.argv)-1
    fig = plt.figure(figsize=(8, 6*num_plots))
    all_ax = []
    max_rate = 0
    for nth_plot, filename in enumerate(sorted(sys.argv[1:],
                                               key=lambda x: fname2temps[x])):
        ax1 = fig.add_subplot(num_plots, 1, nth_plot+1)
        ax1.set_xlabel(u"Amplitude (mV)")
        ax1.set_ylabel(u"Rel. Häufigkeit")
        n_bins = 300
        data = [-a for _, a, _ in datafile.import_raw_csv(filename)]
        #data = [1]
        max_amplitude = max(data)
        hist, bin_start, = np.histogram(data, bins=n_bins, normed=True)
        hist = np.array(hist) * (np.array(bin_start[1:]) - np.array(bin_start[:-1]))
        ax1.step(bin_start[:-1], hist, "-",
                 where="post",
                 label=u"{0}°C".format(fname2temps[filename]-273.15))
        #ax1.set_ybound(lower=0, upper=1.0)
        #ax1.legend(loc=1)
        all_ax.append(ax1)
        
        textstr = "$T_\mathrm{{soll}} = {0:.0f}\,^\degree\!\mathrm{{C}}$\n$N = {1}$\n$N_\mathrm{{bins}}={2}$".format(fname2temps[filename]-273.15, len(data), n_bins)
        info_box(ax1, textstr, x=0.75)
    for ax in all_ax:
        ax.set_xbound(0, max_amplitude)
    fig.savefig("amplitude_histogram.pdf")
