# -*- coding: utf8 -*-
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import re
import datafile
import pulse

event_filter = lambda dt, amplitude: amplitude > 5.0

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
        ax1 = fig.add_subplot(num_plots, 1, nth_plot)
        ax1.set_xlabel(u"Ereignisse (µs^{-1})")
        ax1.set_ylabel(u"Rel. Häufigkeit")
        expected_value = 0
        n_bins = 0
        data = []
        for data_tuple in pulse.pulse_data_to_countrates(filename,
                                                         event_filter):
            rate = data_tuple[1]/data_tuple[2]*1e3
            data.append(rate)
            expected_value += rate
            n_bins = max(n_bins, data_tuple[1])
            max_rate = max(max_rate, rate)
            data.append(rate)
        expected_value /= len(data)
        hist, bin_start, = np.histogram(data, bins=n_bins, normed=1)
        hist = np.array(hist) * (np.array(bin_start[1:]) - np.array(bin_start[:-1]))
        ax1.plot(bin_start[:-1], hist, "x--", label=u"{0}°C".format(fname2temps[filename]-273.15))
        ax1.set_ybound(lower=0, upper=0.7)
        ax1.axvline(expected_value)
        ax1.legend(loc=1)
        all_ax.append(ax1)
    for ax in all_ax:
        ax.set_xbound(0, max_rate)
    fig.savefig("count_rate.pdf")
