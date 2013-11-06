# -*- coding: utf8 -*-

import platform

if platform.python_implementation() == "PyPy":
    import numpypy
    import numpy as np
else:
    import numpy as np
    from matplotlib import pyplot as plt
import sys
import os
import struct
import time as time_mod
import datafile


def neg_slope_deadtime_events(time, signal):
    slope_thresh = 0  # -13.2/7.3
    depth_thresh = 4
    dead_time = 0

    def get_events():
        last_event = -100
        start_signal = 0
        start_time = 0
        prev_positive = True
        prev_datapoint = None
        for n, (t_a, t_b, s_a, s_b) \
            in enumerate(zip(time[1:], time[:-1],
            signal[1:], signal[:-1])):
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
               depth > depth_thresh):# and
              # s_a < -depth_thresh):
                last_event = t
                yield (n, t, depth)
            else:
                if prev_datapoint:
                    yield prev_datapoint
                    prev_datapoint = None
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
    thresh = -16
    events = ((n, (t_a + t_b)/2)
              for n, (t_a, t_b, s_a, s_b)
              in enumerate(zip(time[1:], time[:-1],
                     signal[1:], signal[:-1]))
              if s_a < thresh and s_b > thresh)
    return events


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


def extract_event_times(time, signal):
    events = neg_slope_deadtime_events(time, signal)
    return list(events)


def filter_relevant_events(events):
    try:
        #ev = list(sorted(filter(lambda x: x > 257+50, events)))[0]
        #return [257.36, ev]
        events = list(events)
        ev_trig = sorted(filter(lambda x: x > 215, events))[:2]
        if ev_trig[0] > 270:
            return []
        return ev_trig
    except IndexError:
        return []


def get_peak_height(time, signal, smooth_signal, n):
    try:
        delta_t = time[1] - time[0]
        #n = int(event_time/delta_t)
        max_sig = min_sig = smooth_signal[n]
        max_idx = min_idx = n
        #import pdb; pdb.set_trace()
        #print "max"
        for i in xrange(n-1, n-30, -1):
            #print max_sig, smooth_signal[i]
            if max_sig <= smooth_signal[i]:
                max_sig = smooth_signal[i]
            else:
                max_idx = i
                break
        #print "min"
        for i in xrange(n+1, n+30, 1):
            #print min_sig, smooth_signal[i]
            if min_sig >= smooth_signal[i]:
                min_sig = smooth_signal[i]
            else:
                min_idx = i
                break
        min_sig = signal[min_idx]
        max_sig = signal[max_idx]
        return max_sig - min_sig
    except Exception:
        return 0.0


def process_signal_a(time, signal):
    time, signal = integrate_signal(time, signal)
    events = extract_event_times(time, signal)
    events = filter_relevant_events(events)
    return time, signal, list(events), [0 for i in xrange(len(events)-1)]


def process_signal_b(time, signal):
    time, smooth_signal = integrate_signal(time, signal)
    event_indices, events = zip(*thresh_neg_slope_events(time, smooth_signal))
    event = filter_relevant_events(events)
    if len(events) != 2:
        events = []
        amplitude = []
    elif len(events) > 0:
        amplitude = [get_peak_height(time, signal, smooth_signal, events[1])]
    else:
        amplitude = []
    #print amplitude
    return time, signal, list(events), amplitude

def process_signal_c(time, signal):
    #print "new frame"
    time, smooth_signal = integrate_signal(time, signal)
    try:
        event_indices, events, amplitude = zip(*filter(lambda x: x[1]> 215,
                                                       neg_slope_deadtime_events(time, smooth_signal)))
    except ValueError:
        return time, signal, [], []
    #events = sorted(filter(lambda x: x > 215, events))
    #if len(events) > 1:
        #amplitude = [get_peak_height(time, signal, smooth_signal, index) for index, ev in zip(event_indices[1:], events[1:])]
        ##for ev, am in zip(events[1:], amplitude):
            ##print "amplitude for {0}: {1}".format(ev, am)
    #else:
        #amplitude = []
    #print amplitude
    return time, signal, list(events), amplitude[1:]


if __name__ == "__main__":
    time_differences = []
    amplitudes = []
    num_frames, num_frames_processed = 0, 0
    if len(sys.argv) == 1:
        print "HELP"
        sys.exit(0)
    clock_start = 0
    clock_end = 0

    def print_status():
        sys.stdout.write("\33[2K\rProcess frame {0} of {1}, {2} data points so far, {3:.0f}% efficency, {4:.1f}s elapsed"
                         .format(num_frames_processed,
                                 num_frames,
                                 len(time_differences),
                                 100.0*len(time_differences)/num_frames_processed,
                                 (time_mod.clock() if clock_end == 0 else clock_end) - clock_start
                                 )
                         )
    if sys.argv[1].endswith("/"):
        dirname = os.path.basename(sys.argv[1][:-1])
    else:
        dirname = os.path.basename(sys.argv[1])
    output_name = dirname+".csv"
    clock_start = time_mod.clock()
    for num_frames, num_frames_processed, time, signal in datafile.get_all_data(sys.argv):
        try:
            if 1 and (num_frames_processed % 50) == 0:
                try:
                    print_status()
                    sys.stdout.flush()
                except ZeroDivisionError:
                    pass
            #time, signal, events, ampl = process_signal_a(time, signal)
            time, signal, events, ampl = process_signal_c(time, signal)
            time_differences.extend(ev - events[0]
                                    for ev
                                    in events[1:])
            amplitudes.extend(ampl)
            num_frames_processed += 1
            if 1 and platform.python_implementation() == "CPython":
                for e in events:
                    plt.axvline(e, linestyle="--", color="grey")
                plt.plot(time, signal, linestyle="-")
                plt.savefig("sample_{0}.png".format(num_frames_processed))
                plt.clf()
        except KeyboardInterrupt:
            clock_end = time_mod.clock()
            print_status()
            sys.stdout.write("\nKeyboard interrupt, save recent datapoints\n")
            break
    else:
        clock_end = time_mod.clock()
        print_status()
        sys.stdout.write("\nAll frames processed\n")
    print "Results will be written to", output_name
    with open(output_name, "w") as f:
        f.write("# Num Datapoints {0}\n".format(len(time_differences)))
        f.write("# Processed Frames {0}\n".format(num_frames_processed))
        f.write("# Total processing time {0}s\n".format(clock_end - clock_start))
        f.write("# Time per frame {0}ms\n".format(float(clock_end - clock_start)/num_frames_processed*1000.0))
        for delta, ampl in zip(time_differences, amplitudes):
            f.write(str(delta)+" "+str(ampl)+"\n")
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
