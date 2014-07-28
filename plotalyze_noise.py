
from ROOT import gROOT, TH1F, TF1, TApplication, TFile, TCanvas, TMinuit, Long, Double, TString, TGraph, TGraphErrors

import sys
import math
import re
import datafile
from numpy import array, sqrt


def approximate_afterpulsing(root_file, name, data):
    clean_name = "_".join(name.split()).lower()
    T, rate, sigma_poisson, sigma_sample = map(array, zip(*data))
    
    graph = TGraph(len(T), T, rate)
    graph.Fit("pol3")
    graph.Write(clean_name)
    f1 = graph.GetFunction("pol3")
    chi2ndf = f1.GetChisquare() / f1.GetNDF()
    sigma = math.sqrt(chi2ndf)
    error = array([sigma]*len(T))
    print rate, sigma, error
    
    c1 = TCanvas("c1")
    grapherror = TGraphErrors(len(T), T, rate, array([0.0]*len(T)), error)
    grapherror.SetTitle(name)
    grapherror.SetMarkerStyle(7)
    grapherror.Write(clean_name+"_error")
    c1.SetLogy()
    grapherror.Draw("AP")
    c1.SaveAs("noise_error_{0}.pdf".format(clean_name))
    print "N_\\mathrm{{Ap}} = \\num{{{0}}}".format(sigma_sample/sigma)
    print "N_\\mathrm{{Ap}} = \\num{{{0}}}".format(sigma_sample/sigma)


if __name__ == "__main__":
    filename = sys.argv[1]
    root_file = TFile("noise.root", "RECREATE")
    try:
        data = []
        name = ""
        with open(filename) as f:
            for line in f:
                if not line.strip():
                    if len(data) > 0:
                        approximate_afterpulsing(root_file, name, data)
                    data = []
                    continue
                try:
                    data.append(map(float, line.split()))
                except ValueError:
                    name = line
    finally:
        root_file.Close()