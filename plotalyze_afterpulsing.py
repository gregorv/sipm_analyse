
from ROOT import gROOT, TH1F, TF1, TApplication, TFile, TCanvas, TMinuit, Long, Double, TString

import sys
import math
import re
import datafile
from numpy import array

event_filter = lambda dt, amplitude: abs(amplitude) > 0.0

def only_first_pulse(data, amplitude_threshold):
    old_idx = -1
    for dt, a, idx in data:
        if -a < 20:
            continue
        if idx > old_idx and a < 1000:
            old_idx = idx
            yield dt, a

def get_1pe_amplitude(root_file, temperature, filename):
    """
    Get the amplitude of a 1 photon-event pulse.
    """
    hist = TH1F("amplitude_hist_{0:.0f}K".format(temperature),
                "Amplitudenhistogramm, T={0}K".format(temperature),
                200, 0, 200)
    for _, amplitude, idx in datafile.import_raw_csv(filename):
        if idx > 100000: break
        hist.Fill(-amplitude)
    pe_ampl = 40
    for _ in xrange(4):
        hist.Fit("gaus", "", "", pe_ampl-10, pe_ampl+10)
        gaus = hist.GetFunction("gaus")
        pe_ampl = gaus.GetParameter("Mean")
        print pe_ampl
    hist.Write()
    return pe_ampl
        

def get_afterpulse_data(root_file, temperature, filename):
    print "Get 1pe amplitude"
    pe_amplitude = get_1pe_amplitude(root_file, temperature, filename)
    
    print "Build time-difference histogram"

    data = array([dt for dt, _ in only_first_pulse(datafile.import_raw_csv(filename),
                                                  0.1 * pe_amplitude)])
    print data
    max_dt = max(data)
    min_dt = min(data)
    n_bins = int(max_dt/min_dt)
    hist = TH1F("timediff_hist_{0:.0f}K".format(temperature),
                "Zeitdifferenz-Histogramm, T={0}K".format(temperature),
                n_bins, min_dt, max_dt)
    for dt in data:
        hist.Fill(dt)
    print "Get Afterpulsing Rate"
    hist.Write()
    return 0, 0, 0

if __name__ == "__main__":
    fname2temps = dict((fname,
                        datafile.userheader2dict(datafile.import_csv_header(fname))["T_soll_f"])
                       for fname in sys.argv[1:])
    root_file = TFile("timediff.root", "RECREATE")
    try:
        for nth_plot, filename in enumerate(sorted(sys.argv[1:],
                                                key=lambda x: fname2temps[x])):
            print filename
            temperature = fname2temps[filename]
            n_05e, n15e, n_ap = get_afterpulse_data(root_file, temperature, filename)
    finally:
        root_file.Close()