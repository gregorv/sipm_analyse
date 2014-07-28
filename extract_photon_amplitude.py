
from ROOT import gROOT, TH1F, TF1, TApplication, TFile, TCanvas, TMinuit, Long, Double, TString

import sys
import math
import re
import datafile
from numpy import array

event_filter = lambda dt, amplitude: abs(amplitude) > 0.0

def get_1pe_amplitude(root_file, temperature, filename):
    """
    Get the amplitude of a 1 photon-event pulse.
    """
    hist = TH1F("amplitude_hist_{0:.0f}K".format(temperature),
                filename,
                200, 0, 200)
    for _, amplitude, idx in datafile.import_raw_csv(filename):
        if idx > 10000: break
        hist.Fill(-amplitude)
    pe_ampl = 40
    for _ in xrange(3):
        hist.Fit("gaus", "", "", pe_ampl-10, pe_ampl+10)
        gaus = hist.GetFunction("gaus")
        pe_ampl = gaus.GetParameter("Mean")
        print pe_ampl
    hist.Write()
    return pe_ampl


if __name__ == "__main__":
    fname2temps = dict((fname,
                        datafile.userheader2dict(datafile.import_csv_header(fname))["T_soll_f"])
                       for fname in sys.argv[1:])
    root_file = TFile("pe_amplitude.root", "RECREATE")
    try:
        with open("pe_amplitude.csv", "w") as f:
            for nth_plot, filename in enumerate(sorted(sys.argv[1:],
                                                    key=lambda x: fname2temps[x])):
                temperature = fname2temps[filename]
                pe_amplitude = get_1pe_amplitude(root_file, temperature, filename)
                f.write("{0}\t{1}\t{2}\n".format(filename, temperature, pe_amplitude))
    finally:
        root_file.Close()