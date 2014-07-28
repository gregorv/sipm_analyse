# -*- coding: utf8 -*-
import sys
import math
from math import sqrt
import re
import datafile
import pulse
from matplotlib import rc

#rc('text', usetex=True)
#rc('font', family='serif')

event_filter = lambda dt, amplitude: abs(amplitude) > 0.0

if __name__ == "__main__":
    named_set = {}
    name = None
    for filename in sys.argv[1:]:
        if name is None and filename != "--":
            name = filename
            named_set[name] = []
        elif filename == "--":
            name = None
        else:
            named_set[name].append(filename)
    all_filenames = []
    for f_list in named_set.itervalues():
        all_filenames.extend(f_list)
    fname2temps = dict((fname,
                        datafile.userheader2dict(datafile.import_csv_header(fname))["T_soll_f"])
                       for fname in all_filenames)
    nth_plot = 1
    outfile = open("countrate_temperature.csv", "w")
    testfile = open("test.csv", "w")
    for label, filenames in named_set.iteritems():
        outfile.write(label)
        outfile.write("\n")
        for filename in sorted(filenames,
                               key=lambda x: fname2temps[x]):
            print "# Processing data {0} of {1}".format(nth_plot, len(all_filenames))
            count = 0
            total_time = 0
            rate = 0
            sqr_rate = 0
            num_frames = 0
            for idx, num_events, frame_length in pulse.pulse_data_to_countrates(filename):
                #if num_frames >= 9999:
                    #break
                #if num_events > 10:
                    #continue
                frame_length *= 1e-3;
                num_frames += 1
                count += num_events
                #print num_events
                rate += num_events/frame_length
                sqr_rate += (num_events/frame_length)**2
                total_time += frame_length
                testfile.write("{0}\t{1}\t{2}\n".format(num_frames, rate/num_frames, sqrt(sqr_rate/num_frames - (rate/num_frames)**2)))
            testfile.write("\n\n")
            #print ""
            rate /= num_frames
            sqr_rate /= num_frames
            sample_sigma = sqrt(sqr_rate - rate**2)
            poisson_sigma = sqrt(count) / total_time
            #countrate = sum(data) / len(data) * 1e3
            T = fname2temps[filename]
            nth_plot += 1
            outfile.write("{0}\t{1}\t{2}\t{3}\n".format(T, rate, poisson_sigma, sample_sigma))
        outfile.write("\n\n")
    outfile.close()
    testfile.close()
