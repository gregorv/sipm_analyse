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


def function_value(x, y, x_0):
    #idx = int((x[-1] - x[0])*len(x)*)
    idx = int((x_0 - x[0])/(x[-1] - x[0])*len(x))
    return y[idx]

filter_kernel = None

plot_count = 0


def flankenize(t, sig):
    deltas = map(lambda x: x[0]-x[1], zip(sig[1:], sig[:-1]))
    start_idx = 0
    positive = True
    for i, s in enumerate(deltas):
        if (s > 0.0) != positive:
            yield start_idx, i-1, positive
            start_idx = i
            positive = (s > 0.0)
    yield start_idx, len(deltas)-1, positive


def collapse_flanks(flanks):
    prev_flank = [0, 0, True]
    for cur_flank in flanks:
        # flank polarity change
        if prev_flank[2] != cur_flank[2]:
            yield tuple(prev_flank)
            prev_flank = list(cur_flank)
        # flank gap threshold not reached
        elif cur_flank[0] - prev_flank[1] < 3:
            prev_flank[1] = cur_flank[1]
        else:
            yield tuple(prev_flank)
            prev_flank = list(cur_flank)
    if prev_flank != [0, 0, True]:
        yield tuple(prev_flank)


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


def fit_decay(t, sig, t_ev, t_ev_next=None):
    global plot_count
    t_ev_idx = int((t_ev - t[0])/(t[-1] - t[0])*len(t))
    #min_idx = max(t_ev_idx-10, 0)
    min_idx = max(t_ev_idx-10, 0)
    max_idx = min(t_ev_idx+50, len(t))
    if t_ev_next:
        t_ev_next_idx = int((t_ev_next - t[0])/(t[-1] - t[0])*len(t))
        #print "next-check", max_idx, t_ev_next_idx-4
        max_idx = min(max_idx, t_ev_next_idx-4)
    t_sub = t[min_idx:max_idx]
    sig_sub = sig[min_idx:max_idx]
    tau = 23  # ns
    sig_0 = function_value(t, sig, t_ev)
    x_0 = (sig_0, t_ev, 0, 0, 0)
    #print ""
    #print "x_0", x_0
    #print min_idx, max_idx

    def func_to_fit(t, sig, param):
        t_break_idx = max(int((param[1] - t[0])/(t[-1] - t[0])*len(t)), 0)
        t_lin_break_idx = max(0, t_break_idx-3)
        t_lin = t[:t_lin_break_idx]
        t_exp = t[t_break_idx:]
        linear = param[3]*t_lin + param[4]
        exp = param[0]*np.exp(-(t_exp-param[1])/tau) + param[2]
        #print t_lin_break_idx, t_break_idx, sig[t_lin_break_idx:t_break_idx]
        return np.concatenate([linear, sig[t_lin_break_idx:t_break_idx], exp])

    #print exp(x_0)
    x, cov_x, infodict, mesg, ier = leastsq(lambda x: sig_sub - func_to_fit(t_sub, sig_sub, x),
                                            x_0, full_output=True)
    t_ev = x[1]
    amplitude = abs(x[0]*np.exp((t_ev-x[1])/tau)+x[2] - x[3]*t_ev - x[4])
    ##x = x_0
    ##print mesg
    ##plt.plot(t, sig, "x-")
    #plt.plot(t_sub, sig_sub, "x-")
    ###t_sub = np.arange(t_ev - 50, t_ev + 150)
    #plt.plot(t_sub, func_to_fit(t_sub, sig_sub, x), color="red")
    #plt.plot(t_sub, func_to_fit(t_sub, sig_sub, x_0), color="black")
    #fig.text(0.05, 0.97, "A={0:.1f}, t_ev={1:.0f}".format(amplitude, t_ev))
    #fig.text(0.05, 0.91, str(x))
    #fig.savefig("test_{0}.png".format(plot_count))
    #plot_count += 1
    #fig.clf()
    return t_ev, amplitude


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
