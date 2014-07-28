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

def only_first_pulse(data):
    old_idx = -1
    for a, _, idx in data:
        #if idx > old_idx and a < 1000:
            #old_idx = idx
            yield a

if __name__ == "__main__":
    fname2temps = dict((fname,
                        datafile.userheader2dict(datafile.import_csv_header(fname))["T_soll_f"])
                       for fname in sys.argv[1:])
    print fname2temps
    num_plots = len(sys.argv)-1
    fig = plt.figure()
    max_rate = 0
    ax = fig.add_subplot(111)
    ax.set_xlabel(u"Zeitdifferenz (mV)")
    ax.set_ylabel(u"Wahrscheinlichkeitsdichte")
    ax.semilogy()
    for nth_plot, filename in enumerate(sorted(sys.argv[1:],
                                               key=lambda x: fname2temps[x])):
        print filename
        #ax1 = fig.add_subplot(num_plots, 1, nth_plot+1)
        temperature = fname2temps[filename]
        data = np.array(list(only_first_pulse(datafile.import_raw_csv(filename))))
        n_bins = 100
        n_bins = int(max(data) / min(data))
        print "n_bins", n_bins
        density = False
        hist, bin_start, = np.histogram(data, bins=n_bins, density=density)
        if not density:
            hist = np.array(map(lambda x: float(x)/len(data), hist))
        #hist = np.array(hist) / (np.array(bin_start[1:]) - np.array(bin_start[:-1]))
        ax.step(bin_start[:-1], hist,
                where="post",
                label=u"$T_\mathrm{{Soll}} = {0}\\mathrm{{\\degree C}}$".format(temperature-273.15))
    ax.legend(loc=8)
    fig.savefig("Dt_histogram.pdf")
