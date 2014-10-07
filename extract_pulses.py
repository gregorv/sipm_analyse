# -*- coding: utf-8 -*-
import argparse
import numpy as np
import sys
import time
from lib.filetypes import openDatafile, hasRootSupport

if hasRootSupport():
    from ROOT import TFile, TTree, TBranch, TObjString, TGraph


class RootOutput:
    def __init__(self, root_file):
        self.file = root_file

    def write(self, frame_idx, pulse_list):
        #print(frame_idx, pulse_list)
        pass

    def close(self):
        # do nothing! We do not control file object!!
        pass

    def flush(self):
        pass


class FileOutput:
    def __init__(self, filename):
        try:
            f = open(filename, "r")
            f.close()
            while 1:
                sys.stderr.write("Target file '{0}' already exists."
                                 "Overwrite and continue? [y/N] "
                                 .format(filename))
                answer = raw_input()
                if answer.lower() == "y":
                    break
                elif answer.lower() in ("n", ""):
                    sys.exit(0)
        except IOError:
            # file does not exist
            pass
        if filename == "--":
            self.file = sys.stdout
        else:
            self.file = open(filename, "w")
        self.file.write("\n#Frame Idx;Delta T(ns),Amplitude(mV);Delta T,Amp;Delta T,Amp;Delta T,Amp\n")

    def write(self, frame_idx, pulse_list):
        # outfile.write("# Source data: "+", ".join(args.data))
        # print(frame_idx, pulse_list)
        pulse_number = 0
        while True:
            pulse = [None,]*4
            for ch, pulse_channel in enumerate(pulse_list):
                if not pulse_channel:
                    continue
                try:
                    pulse[ch] = pulse_channel[pulse_number]
                except IndexError:
                    pass
            if all(x is None for x in pulse):
                break
            self.file.write("{0};{1};{2};{3};{4}\n".format(
                frame_idx,
                *tuple(("" if x is None else "{0},{1}".format(*x)) for x in pulse)
            ))
            pulse_number += 1
        self.file.write("\n")

    def close(self):
        self.file.close()

    def flush(self):
        self.file.flush()


def flankenize(t, sig):
    """
    Convert time-domain signal to flank-train.

    Each flank is a tuple of a start index (of the time-array),
    the end index and the flank polarity.
    """
    deltas = map(lambda x: x[0]-x[1], zip(sig[1:], sig[:-1]))
    start_idx = 0
    positive = True
    for i, s in enumerate(deltas):
        if (s > 0.0) != positive:
            yield start_idx, i-1, positive
            start_idx = i
            positive = (s > 0.0)
    yield start_idx, len(deltas)-1, positive


def collapseFlanks(flanks, gap_threshold=3):
    """
    Join consecutive flanks with matchin polarity if
    the distance between the adjacent flanks is smaller
    than the gap_threshold (in "index units").
    """
    prev_flank = [0, 0, True]
    for cur_flank in flanks:
        # flank polarity change
        if prev_flank[2] != cur_flank[2]:
            yield tuple(prev_flank)
            prev_flank = list(cur_flank)
        # flank gap threshold not reached
        elif cur_flank[0] - prev_flank[1] < gap_threshold:
            prev_flank[1] = cur_flank[1]
        else:
            yield tuple(prev_flank)
            prev_flank = list(cur_flank)
    if prev_flank != [0, 0, True]:
        yield tuple(prev_flank)


def smoothSignal(time, signal, half_window_size=5):
    """
    Moving average time-domain filter.
    """
    new_sig = []
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


def processFrame(number, t, sig):
    t_start = 200
    t, smooth_signal = smoothSignal(t, sig)
    # Get events
    a = filter(lambda x: (x[1]-x[0]) > 2 or not x[2], flankenize(t, smooth_signal))
    b = filter(lambda x: not x[2] and (smooth_signal[x[1]] - smooth_signal[x[0]]) < -5, collapseFlanks(a))
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


def processFrameFull(number, t, sig):
    t, smooth_signal = smoothSignal(t, sig)
    a = filter(lambda x: (x[1]-x[0]) > 2 or
               not x[2], flankenize(t, smooth_signal))
    b = filter(lambda x: not x[2] and
               (smooth_signal[x[1]] - smooth_signal[x[0]]) < -5,
               collapseFlanks(a))
    return [(t[end_idx],
             sig[end_idx] - sig[start_idx])
            for start_idx, end_idx, pos in b]


def timeString(seconds):
    s = ""
    if seconds > 0 and seconds < 1:
        return "{0} ms".format(int(seconds*1000))
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
    parser.add_argument("-o", "--outfile", default=None, metavar="outfile",
                        help="Specify -- for output on stdout. If outfile is"
                        "not specified and input is a ROOT file, new result"
                        "tables are created. If the input was not a ROOT file,"
                        "you have to specify an output file, otherwise an"
                        "error is raised.")
    parser.add_argument("-s", "--silent", default=False, action="store_true")
    parser.add_argument("-S", "--full-spectrum", default=False,
                        action="store_true",
                        help="If set, *all* pulses in the frame are placed"
                        "in output. The default is to write only the pulses"
                        "after a certain initial pulse that occured in a"
                        "certain time-range. The -S output is used for"
                        "countrates, without it, it is usefull for"
                        "time-structure analysis (after-pulsing etc.)")
    parser.add_argument("-c", "--channels", default="1",
                        help="Comma separated list of channel numbers to"
                        "analyze. By default, only the first channel"
                        "is analyzed.")
    parser.add_argument("data", nargs="+", type=str)
    args = parser.parse_args()

    channels = [int(x)-1 for x in args.channels.split(",")]

    rw_mode = args.outfile is None
    datafile = openDatafile(args.data[0], rw_mode)
    if rw_mode:
        try:
            root = datafile.getRootFile()
            outfile = RootOutput(root)
        except AttributeError:
            sys.stderr.write("{0}: Cannot write results in non ROOT input"
                             "file. Please specify output file."
                             .format(sys.argv[0]))
    else:
        outfile = FileOutput(args.outfile)
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
            # elapsed_time = (time.clock() if clock_end == 0 else clock_end) - clock_start
            elapsed_time = time.clock() - clock_start
            if num_frames_processed > 0:
                time_remaining = elapsed_time/num_frames_processed * (num_frames - num_frames_processed)
            sys.stderr.write("\33[2K\rProcess frame {0} of {1}, {2} elapsed, {3:.1f}ms/frame, about {4} remaining"
                             .format(num_frames_processed,
                                     num_frames,
                                     timeString(elapsed_time),
                                     elapsed_time/num_frames_processed*1000.0
                                     if num_frames_processed > 0 else 0.0,
                                     timeString(time_remaining)
                                     if num_frames_processed > 0 else '?'
                                     )
                             )
            outfile.flush()
        clock_start = time.clock()
        if args.full_spectrum:
            processFunction = processFrameFull
        else:
            processFunction = processFrame
        for num_frames, frame_idx, t, signals in datafile.frames(dataset):
            frames_per_display_refresh = min(50, max(1, num_frames/50))
            try:
                if 1 and (num_frames_processed % frames_per_display_refresh) == 0:
                    try:
                        print_status()
                        sys.stdout.flush()
                    except ZeroDivisionError:
                        pass
                event_list = [None,]*4
                for ch in channels:
                    if signals[ch] is None:
                        continue
                    event_list[ch] = processFunction(frame_idx, t, signals[ch])
                outfile.write(frame_idx, event_list)
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
