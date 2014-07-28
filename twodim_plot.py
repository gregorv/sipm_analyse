# -*- coding: utf8 -*-
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
import math


def read_data(file):
    delta_list = []
    amplitude_list = []
    num_events = 0
    cur_frame_idx = -1
    for line in file:
        if line.strip().startswith("#") or not line.strip():
            continue
        delta, amplitude, frame_idx = map(float, map(str.strip, line.split()))
        if delta < 0 or amplitude >= 0:
            continue
        #if delta < 0 or delta > 1100 or amplitude > 100:
            #continue
        #if delta > 280 and delta < 320:
            #continue
        if delta > 1000:
            continue
        if cur_frame_idx != frame_idx:
            num_events += 1
            cur_frame_idx = frame_idx
        delta_list.append(delta)
        amplitude_list.append(-amplitude)
    return delta_list, amplitude_list, num_events

if __name__ == "__main__":
    delta_list = []
    amplitude_list = []
    try:
        with open(sys.argv[1]) as f:
            delta_list, amplitude_list, num_events = read_data(f)
    except IOError:
        sys.stderr.write("Cannot open file {0}\n".format(sys.argv[1]))
        sys.exit(1)
    except IndexError:
        delta_list, amplitude_list, num_events = read_data(sys.stdin)
    
    max_amplitude = max(amplitude_list)
    max_delta = max(delta_list)
    amplitude_list = np.array(amplitude_list)
    delta_list = np.array(delta_list)
    n_tbins = int(max_delta / min(delta_list))
    n_ybins = int(max_amplitude / min(amplitude_list))
    n_ybins = n_tbins*5
    print "n_bins", n_tbins, "x", n_ybins
        
    print len(delta_list)
    fig = plt.figure(figsize=(7, 6))
    ax1 = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    ax1.set_xlabel(u"Zeitabstand / ($\mathrm{ns}$)")
    ax1.set_ylabel(u"Amplitude / ($\mathrm{mV}$)")
    H, xedges, yedges = np.histogram2d(amplitude_list, delta_list, bins=(n_ybins, n_tbins))
    H /= (max_amplitude/n_ybins) * (max_delta/n_tbins*1e-3) * num_events
    #H = np.log(H)/np.log(10)
    #extent = [0, 10, 45, 50]
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    ax1.set_aspect("equal")
    print ax1.get_aspect()
    im = ax1.imshow(H, extent=extent,
                    origin="low",
                    aspect="auto",
                    #cmap="gist_rainbow",
                    interpolation='nearest',
                    norm=LogNorm()
                    )
    print ax1.get_aspect()
    colorbar = fig.colorbar(im)
    #colorbar.set_log()
    colorbar.set_label(u"Sekundärereignisse/Puls / ($\mathrm{\mu s}^{-1}\mathrm{mV}^{-1}$)")
    fig.savefig("amplitude_histogram.pdf")
