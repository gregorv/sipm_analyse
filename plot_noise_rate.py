# -*- coding: utf8 -*-
from matplotlib import pyplot as plt
import sys
import math

if __name__ == "__main__":
    data = []
    with open(sys.argv[1]) as f:
        for line in f:
            data.append(float(line.strip()))
    fig = plt.figure()

    plt.xlabel(u"Zeitabstand (ns)")
    plt.ylabel(u"Zählrate")
    #n_bins = 10**int(math.log10(len(data)) - 1)
    n_bins = 10000 if len(data) > 1000 else 50
    plt.title("n={0}, n_bins={1}".format(len(data), n_bins))
    plt.hist(data, bins=n_bins, log=True)
    plt.savefig("time_difference_histogram.png")
