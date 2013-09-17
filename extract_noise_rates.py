# -*- coding: utf8 -*-

import platform

if platform.python_implementation() == "pypy":
    import numpypy
    import numpy as np
else:
    import numpy as np
    from matplotlib import pyplot as plt
import sys
import os
import struct


def import_frame(filename):
    with open(filename, "rb") as f:
        magic = f.read(5)
        if magic == "#BIN\n":
            return import_frame_bin(f)
        elif magic == "#TXT\n":
            return import_frame_txt(f)
        elif magic == "#NPY\n":
            assert("Not implemented!")
        else:
            f.seek(0)
            return import_frame_txt(f)


def import_frame_bin(f):
    data = []
    while True:
        read = f.read(256*4)
        if not len(read):
            break
        data.extend(struct.unpack("f"*256, read))
    assert (len(data) == 4096 or len(data) == 2048),\
        "Expected 4096 or 2048 floats in input file"
    return np.array(data[:len(data)/2]), np.array(data[len(data)/2:])


def import_frame_txt(f):
    time = []
    signal = []
    for l in f:
        l = l.split(" ")
        time.append(float(l[0]))
        signal.append(float(l[1]))
    return np.array(time), np.array(signal)


def neg_slope_deadtime_events(time, signal):
    slope_thresh = -1  # -13.2/7.3
    depth_thresh = 13
    dead_time = 03

    def get_events():
        last_event = -100
        start_signal = 0
        start_time = 0
        prev_positive = True
        for t_a, t_b, s_a, s_b in zip(time[1:], time[:-1],
                                      signal[1:], signal[:-1]):
            t = (t_a + t_b) / 2
            slope = (s_a - s_b) / (t_a - t_b)
            s = (s_a + s_b) / 2
            if prev_positive and slope < 0:
                start_signal = s_b
                start_time = t_b
                prev_positive = False
            elif not prev_positive and slope > 0:
                prev_positive = True
            depth = start_signal - s_a
            total_slope = depth/(start_time - t_a)
            #if slope < 0:
                #print t_a, slope, total_slope, depth, start_signal, start_time, s_a
            if(total_slope < slope_thresh and
               last_event + dead_time < t and
               depth > depth_thresh and
               s_a < -depth_thresh):
                last_event = t
                yield t
    return get_events()


def neg_slope_events(time, signal):
    slope_thresh = 0  # -13.2/10.3
    slopes = ((s_a - s_b)/(t_a - t_b)
              for s_a, s_b, t_a, t_b
              in zip(signal[1:], signal[:-1],
                     time[1:], time[:-1]))
    depth = (s_a - s_b
             for s_a, s_b, t_a, t_b
             in zip(signal[1:], signal[:-1],
                    time[1:], time[:-1]))
    events = ((t_a + t_b)/2
              for t_a, t_b, s, depth
              in zip(time[1:], time[:-1], slopes, depth)
              if s < slope_thresh and abs(depth) > 10)
    return events


def thresh_neg_slope_events(time, signal):
    thresh = -10
    events = ((t_a + t_b)/2
              for t_a, t_b, s_a, s_b
              in zip(time[1:], time[:-1],
                     signal[1:], signal[:-1])
              if s_a < thresh and s_b > thresh)
    return events


def integrate_signal(time, signal):
    new_sig = []
    half_window_size = 4
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


def extract_event_times(time, signal):
    events = neg_slope_deadtime_events(time, signal)
    return list(events)


if __name__ == "__main__":
    time_differences = []
    files_to_process = set()
    files_processed = set()
    directories = []
    output_name = None

    def discover_files():
        for d in directories:
            for fname in os.listdir(d):
                if fname.endswith(".dat") or fname.endswith(".csv"):
                    fname = os.path.join(d, fname)
                    if fname not in files_processed:
                        files_to_process.add(fname)

    if len(sys.argv) == 1:
        print "HELP"
    else:
        for path in sys.argv[1:]:
            if os.path.isdir(path):
                directories.append(path)
                if not output_name:
                    output_name = os.path.basename(path)+".csv"
            elif path.endswith(".dat") or path.endswith(".csv"):
                files_to_process.add(path)
    discover_files()

    if not output_name:
        output_name = "time_differences.csv"
    while len(files_to_process):
        filename = files_to_process.pop()
        files_processed.add(filename)
        if len(files_to_process) == 0:
            discover_files()
        #print filename
        if 1 and (len(files_processed) % 10) == 0:
            sys.stdout.write("\33[2K\rProcess frame {0} of {1}, {2} data points so far"
                             .format(len(files_processed),
                                     len(files_processed) +
                                     len(files_to_process),
                                     len(time_differences)))
            sys.stdout.flush()
        time, signal = import_frame(filename)
        time, signal = integrate_signal(time, signal)
        events = extract_event_times(time, signal)
        time_differences.extend(a - b
                                for a, b
                                in zip(events[1:], events[:-1]))
        if 0 and platform.python_implementation() == "cpython":
            for e in events:
                plt.axvline(e, linestyle="--", color="grey")
            plt.plot(time, signal, linestyle="-")
            plt.savefig(os.path.basename(filename)+".png")
            plt.clf()
    sys.stdout.write("\33[2K\rAll frames processed")
    with open(output_name, "w") as f:
        for delta in time_differences:
            f.write(str(delta)+"\n")
    if not len(time_differences):
        print "No events? O.o"
    elif len(time_differences) < 2:
        print "Not enough events"
    #else:
        #fig = plt.figure()
        #plt.xlabel(u"Zeitabstand (ns)")
        #plt.ylabel(u"Zählrate")
        #plt.text("Hallo")
        #plt.hist(time_differences, bins=30, log=True)
        #plt.savefig("time_difference_histogram.pdf")
        #del fig
