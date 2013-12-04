# -*- coding: utf-8 -*-

import platform
if platform.python_implementation() == "PyPy":
    import numpypy
else:
    import matplotlib.pyplot as plt
    from scipy.optimize import leastsq
    fig = plt.figure()
import numpy as np
#import scipy
import datafile
import sys
import os
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("-d", "--dataset", default=None, metavar="dataset")
    parser.add_argument("-o", "--outfile", default='datafile.pdf', metavar="outfile")
    parser.add_argument("data", nargs="+", type=str)
    args = parser.parse_args()
    if args.dataset:
        dataset = set()
        groups = map(str.strip, args.dataset.split(","))
        for g in groups:
            if ":" in g:
                low, high = map(int, g.split(":"))
                dataset.update(range(low, high+1))
            else:
                dataset.add(int(g))
    else:
        sys.stdout("Please specify dataset")
        sys.exit(0)
    fig = plt.figure(figsize=(6,3*len(dataset)))
    for i, (num_frames, frame_idx, t, signal) in enumerate(datafile.get_all_data(args.data,
                                                                  dataset)):
        plt.subplot(num_frames, 1, i)
        plt.plot(t, signal, label="Frame {0}".format(frame_idx))
        plt.legend(loc=4)
    fig.savefig(args.outfile)
