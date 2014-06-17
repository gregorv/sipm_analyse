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
from pulse import flankenize, collapse_flanks, integrate_signal, time_string


def get_events(t, sig):
    a = filter(lambda x: (x[1]-x[0]) > 2 or not x[2], flankenize(t, sig))
    #for x in collapse_flanks(a):
        #print not x[2], sig[x[1]] - sig[x[0]]
    b = filter(lambda x: not x[2] and (sig[x[1]] - sig[x[0]]) < -5, collapse_flanks(a))
    #plt.plot(t, sig, "x-")
    #for start_idx, end_idx, pos in b:
        #print [t[start_idx], t[end_idx]], [sig[start_idx], sig[end_idx]]
        #plt.plot([t[start_idx], t[end_idx]],
                 #[sig[start_idx], sig[end_idx]], color="red")
    #fig.savefig("test.pdf")
    #fig.clf()
    return [t[end_idx] for start_idx, end_idx, pos in b]


def process_frame(number, t, sig):
    _, sig_smooth = integrate_signal(t, sig)
    ev = get_events(t, sig_smooth)
    return len(ev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("-d", "--dataset", default=None, metavar="dataset")
    parser.add_argument("-o", "--outfile", default='', metavar="outfile")
    parser.add_argument("-s", "--silent", default=False, action="store_true")
    parser.add_argument("data", type=str)
    args = parser.parse_args()
    if args.outfile:
        try:
            f = open(args.outfile, "r")
            f.close()
            while 1:
                sys.stderr.write("Target file '{0}' already exists. Overwrite and continue? [y/N] ".format(args.outfile))
                answer = raw_input()
                if answer.lower() == "y":
                    break
                elif answer.lower() in ("n", ""):
                    sys.exit(0)
        except IOError:
            # file does not exist
            pass
        outfile = open(args.outfile, "w")
    else:
        outfile = sys.stdout
    try:
        if args.dataset:
            dataset = set()
            groups = args.dataset.split(",")
            for g in groups:
                if ":" in g:
                    low, high = map(int, g.split(":"))
                    dataset.update(range(low, high+1))
                else:
                    dataset.add(int(g))
        else:
            dataset = None
        num_frames, num_frames_processed = 0, 0
        clock_start = 0
        clock_end = 0

        def print_status():
            #elapsed_time = (time.clock() if clock_end == 0 else clock_end) - clock_start
            elapsed_time = time.clock() - clock_start
            time_remaining = elapsed_time/num_frames_processed * (num_frames - num_frames_processed)
            sys.stderr.write("\33[2K\rProcess frame {0} of {1}, {2} elapsed, {3:.1f}ms/frame, about {4} remaining"
                            .format(num_frames_processed,
                                    num_frames,
                                    time_string(elapsed_time),
                                    elapsed_time/num_frames_processed*1000.0,
                                    time_string(time_remaining)
                                    )
                            )
            outfile.flush()
        if args.data[0].endswith("/"):
            dirname = os.path.basename(args.data[0][:-1])
        else:
            dirname = os.path.basename(args.data[0])
        info = datafile.get_dta_info(args.data)
        if not info["auto_trigger"]:
            sys.stderr.write("WARNING: Input file was not auto triggered. Analysis on this file is meaningless.\nExiting.\n")
            sys.exit(0)
        output_name = dirname+".csv"
        clock_start = time.clock()
        outfile.write("#Frame Idx\tNum Events\tTimespan(ns)\n")
        for key, val in datafile.get_dta_userheader(args.data).iteritems():
            outfile.write("# {0} = {1}\n".format(key, val))
        total_time = 0
        total_events = 1
        timespan = None
        for num_frames, frame_idx, t, signal in datafile.get_all_data([args.data],
                                                                      dataset):
            try:
                if 1 and (num_frames_processed % 50) == 0:
                    try:
                        print_status()
                        sys.stdout.flush()
                    except ZeroDivisionError:
                        pass
                if not timespan:
                    timespan = t[-1]-t[0]
                num_events = process_frame(frame_idx, t, signal)
                outfile.write("{0}\t{1}\t{2}\n".format(frame_idx,
                                                       num_events,
                                                       timespan))
                total_time += timespan
                total_events += num_events
                num_frames_processed += 1
            except KeyboardInterrupt:
                clock_end = time.clock()
                if not args.silent:
                    print_status()
                    sys.stderr.write("\nKeyboard interrupt, save recent datapoints\n")
                break
        else:
            clock_end = time.clock()
            if not args.silent:
                print_status()
                sys.stderr.write("\nAll frames processed\n")
        outfile.write("# Total: {0}/{1}ns = {2}s^-1\n".format(total_events,
                                                          total_time,
                                                          total_events/total_time*1e9))
    finally:
        outfile.close()
    #print "Results will be written to", output_name
    #if not 0:
        #print "No events? O.o"
    #elif len(time_differences) < 2:
        #print "Not enough events"
    ##else:
        #fig = plt.figure()
        #plt.xlabel(u"Zeitabstand (ns)")
        #plt.ylabel(u"Zählrate")
        #plt.text("Hallo")
        #plt.hist(time_differences, bins=30, log=True)
        #plt.savefig("time_difference_histogram.pdf")
        #del fig
