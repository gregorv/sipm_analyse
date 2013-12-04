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
from pulse import *


filter_kernel = None


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
    t, smooth_signal = integrate_signal(t, sig)
    # Get events
    events = get_events(t, smooth_signal)
    # Fit exp. curve to get each events amplitude
    result = []
    t_0 = None
    for i, t_ev in enumerate(events):
        try:
            t_ev_next = events[i+1]
        except IndexError:
            t_ev_next = None
        t_ev_fit, amplitude = fit_decay(t, sig, t_ev, t_ev_next)
        #print "x_1", x
        if t_ev_fit-200 < 0.0 or t_ev_fit > t[-1]:
            continue
        if amplitude > 1000.0:
            sys.stderr.write("\nSorted out! Frame idx {0}\n".format(number))
            return []
        if t_0 is None:
            t_0 = t_ev_fit
        else:
            result.append((t_ev_fit-t_0, amplitude))
    # prepare return set
    return result


def process_frame_quick(number, t, sig):
    t_start = 200
    t, smooth_signal = integrate_signal(t, sig)
    plt.plot(t, smooth_signal)
    plt.savefig("test.png")
    plt.clf()
    # Get events
    a = filter(lambda x: (x[1]-x[0]) > 2 or not x[2], flankenize(t, smooth_signal))
    b = filter(lambda x: not x[2] and (smooth_signal[x[1]] - smooth_signal[x[0]]) < -5, collapse_flanks(a))
    c = filter(lambda x: int(t_start/(t[-1] - t[0])*len(t)) < x[0], b)
    events = []
    for start_idx, end_idx, _ in c:
        end_idx -= 3  # empiric offset
        events.append((t[end_idx],
                       sig[end_idx] - sig[start_idx]))
    if len(events) > 1:
        return [(t_ev-events[0][0], amplitude)
                for t_ev, amplitude
                in events[1:]]
    else:
        return []


def integrate_signal(time, signal):
    new_sig = []
    half_window_size = 5
    for i in xrange(len(signal)):
        used_samples = 0
        sig = 0
        for j in xrange(i-half_window_size, i+half_window_size):
            try:
                sig += signal[j]
                used_samples += 1
            except IndexError:
                pass
        sig /= used_samples
        new_sig.append(sig)
    return time, new_sig


def time_string(seconds):
    s = ""
    seconds = int(seconds)
    if (seconds / 86400) > 0:
        s = ''.join((s, str(seconds / 86400), "d "))
        seconds %= 86400
    if (seconds / 3600) > 0:
        s = ''.join((s, str(seconds / 3600), "h "))
        seconds %= 3600
    if (seconds / 60) > 0:
        s = ''.join((s, str(seconds / 60), "m "))
        seconds %= 60
    return ''.join((s, str(seconds), "s"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("-d", "--dataset", default=None, metavar="dataset")
    parser.add_argument("-o", "--outfile", default='', metavar="outfile")
    parser.add_argument("-s", "--silent", default=False, action="store_true")
    parser.add_argument("-q", "--quick-and-dirty", default=False, action="store_true")
    parser.add_argument("data", nargs="+", type=str)
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

        try:
            info = datafile.get_dta_info(args.data[0])
            if info:
                header = datafile.get_dta_userheader(args.data[0])
                out_header = "# "+"\n# ".join(header)+"\n" if len(header) > 0 else ""
                outfile.write("#Source dta-file: {0}\n## Original user header\n{1}##\n\n".format(args.data[0], out_header))
        except Exception:
            raise

        if args.quick_and_dirty:
            process_function = process_frame_quick
        else:
            process_function = process_frame

        def print_status():
            #elapsed_time = (time.clock() if clock_end == 0 else clock_end) - clock_start
            elapsed_time = time.clock() - clock_start
            if num_frames_processed > 0:
                time_remaining = elapsed_time/num_frames_processed * (num_frames - num_frames_processed)
            sys.stderr.write("\33[2K\rProcess frame {0} of {1}, {2} elapsed, {3:.1f}ms/frame, about {4} remaining"
                            .format(num_frames_processed,
                                    num_frames,
                                    time_string(elapsed_time),
                                    elapsed_time/num_frames_processed*1000.0 if num_frames_processed > 0 else 0.0,
                                    time_string(time_remaining) if num_frames_processed > 0 else '?'
                                    )
                            )
            outfile.flush()
        if args.data[0].endswith("/"):
            dirname = os.path.basename(args.data[0][:-1])
        else:
            dirname = os.path.basename(args.data[0])
        output_name = dirname+".csv"
        clock_start = time.clock()
        outfile.write("# Source data: "+", ".join(args.data))
        outfile.write("\n#Delta T\tAmplitude\tFrame Idx\n")
        for num_frames, frame_idx, t, signal in datafile.get_all_data(args.data,
                                                                      dataset):
            #if dataset is not None and frame_idx not in dataset:
                #continue
            try:
                if 1 and (num_frames_processed % 50) == 0:
                    try:
                        print_status()
                        sys.stdout.flush()
                    except ZeroDivisionError:
                        pass
                event_list = process_function(frame_idx, t, signal)
                for delta_t, amplitude in event_list:
                    outfile.write("{0}\t{1}\t{2}\n".format(delta_t, amplitude, frame_idx))
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
