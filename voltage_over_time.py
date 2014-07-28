# -*- coding: utf-8 -*-

import platform
import sys
if platform.python_implementation() == "PyPy":
    print "PyPy analysis not supported, sorry! =("
    sys.exit(0)
else:
    from ROOT import TMinuit, Long, Double, TString
    import matplotlib.pyplot as plt
    from scipy.optimize import leastsq
    fig = plt.figure()
import numpy as np
from array import array
#import scipy
import os
import time
import argparse
import datafile
import math
import re


class FitErrorless:
    """
    """
    def __init__(self, func, n_pars):
        self.minuit = TMinuit(n_pars)
        self.n_pars = n_pars
        self.func = func
        self.minuit.SetFCN(self.fcn)
        self._fit_range = None

    def chisq(self, par):
        chisq = 0
        #for x, y in zip(self.data_x, self.data_y):
            #if self._fit_range:
                #if not self._fit_range[0] <= x <= self._fit_range[1]:
                    #continue
            #fval = self.func(par, x)
            #chisq += (y - fval)**2
        #windowed_x = np.array(filter(lambda x: (par[2]-par[0]/2) < x < (par[2]+par[0]/2), self.data_x))
        chisq = sum((self.data_y - self.func(par, self.data_x))**2)/par[0]
        return chisq

    def fcn(self, npar, gin, f, xval, iflag):
        f[0] = self.chisq([xval[i] for i in xrange(npar[0])])

    def setParameterDefs(self, par_id, name, init, step, min, max):
        """
        [(par_id, name, init, step, min, max), ]
        """
        ierflg = Long()
        self.minuit.mnparm(par_id, name, init, step, min, max, ierflg)

    def fit(self, x, y, fit_range=None):
        self.data_x = x
        self.data_y = y
        self._fit_range = fit_range
        arglist = array("d", (1, 1))
        ierflg = Long()

        self.minuit.mnexcm("SET ERR", arglist, 1, ierflg)
        arglist = array("d", (1000., .1))
        self.minuit.mnexcm("MIGRAD", arglist, 2, ierflg)
        amin, edm, errdef = Double(), Double(), Double()
        nvpar, nparx, icstat = Long(), Long(), Long()
        self.minuit.mnstat(amin, edm, errdef, nvpar, nparx, icstat)

    def getParameterData(self, par_id):
        par, error, bound_min, bound_max = Double(), Double(), Double(), Double()
        name = TString()
        ierflg = Long()
        self.minuit.mnpout(par_id, name, par, error,
                           bound_min, bound_max, ierflg)
        return par, error, name, bound_max, bound_min

    def getParameter(self, par_id):
        return self.getParameterData(par_id)[0]

    def getParameterError(self, par_id):
        return self.getParameterData(par_id)[1]


class FitWindowedGaus(FitErrorless):
    """
    """
    def __init__(self):
        func = (lambda par, x: par[1]*np.exp(-(x-par[2])**2 * par[3]**2))
        FitErrorless.__init__(self, func, 4)
        self.setParameterDefs(0, "window", 40, 0.1, 30, 100)
        self.setParameterDefs(1, "N", 0.07, 0.001, 0, 500)
        self.setParameterDefs(2, "E", 43, 0.1, 20, 1000)
        self.setParameterDefs(3, "inv_sigma", 0.1, 0.1, 0, 0)

    def chisq(self, par):
        chisq = 0
        #for x, y in zip(self.data_x, self.data_y):
            #if self._fit_range:
                #if not self._fit_range[0] <= x <= self._fit_range[1]:
                    #continue
            #fval = self.func(par, x)
            #chisq += (y - fval)**2
        windowed = filter(lambda p: (par[2]-par[0]/2) <
                          p[0] < (par[2]+par[0]/2),
                          zip(self.data_x, self.data_y))
        if len(windowed) == 0:
            return 1e9
        x, y = map(np.array, zip(*windowed))
        chisq = sum((y - self.func(par, x))**2)/par[0]
        return chisq


def bin_dataset(event_list):
    relevant_amplitudes = map(lambda x: abs(x[1]),
                              filter(lambda x: x[0] > 100,
                                     event_list
                                     )
                              )
    return np.histogram(relevant_amplitudes, bins=200)


def plot_hist2d(ax, hist, xedges, yedges):
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    #im = ax.imshow(H, extent=extent, origin="low", interpolation='nearest')
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, hist.transpose())
    ax.set_aspect('auto')

fileno = 0


def get_first_level(filename, ev_list):
    global fileno
    t, amp, _ = zip(*filter(lambda x: 300 < x[0] < 1000 and
                            -1000 < x[1] < -15.0, ev_list))
    n_bins = int(max(np.abs(amp))/3)
    hist, edges = np.histogram(np.abs(amp), bins=n_bins)
    hist = hist.astype(float) / float(len(amp))

    # guess initial parameters
    N_0 = np.max(hist)
    E_0 = [(edges[i]+edges[i+1])/2.0
           for i, N
           in enumerate(hist)
           if N == N_0][0]
    fit = FitWindowedGaus()
    fit.setParameterDefs(1, "N", N_0, 0.001, 0, 500)
    fit.setParameterDefs(2, "E", E_0, 0.1, 20, 1000)
    par_0 = [fit.getParameter(i) for i in xrange(4)]
    fit.fit(edges[:-1], hist)
    par = [fit.getParameter(i) for i in xrange(4)]
    fig.clf()
    #fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, aspect="auto")
    x_0 = np.arange(par_0[2]-par_0[0]/2, par_0[2]+par_0[0]/2, 1)
    x = np.arange(par[2]-par[0]/2, par[2]+par[0]/2, 1)
    ax.set_xlim(min(x[0], x_0[0]), max(x[-1], x_0[-1]))
    ax.plot(edges[:-1], hist, label=filename)
    #t_fit = np.arange(fit_range[0], fit_range[1], (fit_range[1] - fit_range[0])/100)
    ax.plot(x, fit.func(par, x), "r-.", label="Fit")
    ax.plot(x_0, fit.func(par_0, x_0), "r--", label="Initial Parameters")
    ax.legend()
    fig.savefig("fit_file_{0:04d}_curve.png".format(fileno))
    fileno += 1
    return par[2], fit.getParameterError(2), 1.0/par[3], fit.getParameterError(3)/(par[3]**2)


def Heavyside(x):
    return 0.5*(np.sign(x)+1.0)


def get_levels(filename, ev_list):
    global fileno
    t, amp, _ = zip(*filter(lambda x: 300 < x[0] < 1000 and
                            -1000 < x[1] < -15.0, ev_list))
    n_bins = int(max(np.abs(amp))/2)
    hist, edges = np.histogram(np.abs(amp), bins=n_bins)
    hist = hist.astype(float) / float(len(amp))
    #func_i = (lambda par, i, x: par[1]*np.exp(-(x-par[2]*i)**2 * par[3]**2))
    func = (lambda par, x: (par[2]*np.exp(-(x-par[0])**2 * par[1]**2) +
                            par[3]*np.exp(-(x-par[0]*2)**2 * par[1]**2) +
                            par[4]*np.exp(-(x-par[0]*3)**2 * par[1]**2)
                            #+ Heavyside(x-par[7])*par[5]*np.exp(par[6]*x)))
                            ))
    fit = FitErrorless(func, 4)
    fit.setParameterDefs(0, "V_1", 44, 0.1, 10, 200.0)
    fit.setParameterDefs(1, "inv_sigma", 0.07, 0.001, -500, 500)
    fit.setParameterDefs(2, "count_1", 0.006, 0.0001, 0.0, 5.0)
    fit.setParameterDefs(3, "count_2", 0.0045, 0.0001, 0.0, 5.0)
    fit.setParameterDefs(4, "count_3", 0.003, 0.0001, 0.0, 5.0)
    fit.setParameterDefs(5, "offset", 5, 0.1, 0.0, 2000.0)
    fit.setParameterDefs(6, "lambda", -0.005, 0.001, -1000.0, 0.0)
    fit.setParameterDefs(7, "start", 60, 10, 0.0, 1000.0)
    #fit.fit(edges[:-1], hist)
    par = [fit.getParameter(i) for i in xrange(8)]
    fig.clf()
    #fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, aspect="auto")
    x = np.arange(edges[0], edges[-1], (edges[-1]-edges[0])/1000)
    ax.plot(x, hist, label=filename)
    ##t_fit = np.arange(fit_range[0], fit_range[1], (fit_range[1] - fit_range[0])/100)
    ax.plot(x, func(par, x), "r-.")
    ax.legend()
    #ax.semilogy()
    fig.savefig("fit_file_{0}_curve.png".format(fileno))
    fileno += 1
    return par[0]


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
    parser.add_argument("event_files", nargs="+", type=str)
    args = parser.parse_args()
    n_files = len(args.event_files)
    x_tiles = int(math.ceil(math.sqrt(n_files)))
    y_tiles = int(math.ceil(n_files/x_tiles)+1)
    #fig = plt.figure(figsize=(x_tiles*6, y_tiles*4))
    fig = plt.figure(figsize=(12, 8))
    outfile = open("levels.csv", "w")
    for i, filename in enumerate(sorted(args.event_files)):
        try:
            #if 1 and (num_frames_processed % 50) == 0:
                #try:
                    #print_status()
                    #sys.stdout.flush()
                #except ZeroDivisionError:
                    #pass
            #header = datafile.get_dta_userheader(filename)
            #if "V_bias" not in header:
                #sys.stderr.write("Warning: Bias-voltage not specified in file {0}, data worthless\n".format(filename))
                #continue
            #voltage = header["V_bias"]
            event_list = datafile.import_raw_csv(filename)
            header = datafile.import_csv_header(filename)
            voltage = None
            for entry in header:
                try:
                    key, value = map(str.strip, entry.split("="))
                except ValueError:
                    continue
                if key == "V_bias_f":
                    voltage = float(value)
            if not voltage:
                sys.stderr.write("Warning: Bias-voltage not specified in file {0}, data worthless\n".format(filename))
                continue
            #H, edges = bin_dataset(event_list)
            amplitude, amplitude_err, sigma, sigma_err = map(str, get_first_level(filename, event_list))
            outfile.write("{0} {1} {2} {3} {4}\n".format(str(voltage),
                                                         amplitude, amplitude_err,
                                                         sigma, sigma_err))
            outfile.flush()
            #ax = fig.add_subplot(y_tiles, x_tiles, i+1)
            #ax.autoscale(axis='y')
            #ax.set_title(os.path.basename(filename))
            #ax.plot((edges[1:]+edges[:-1])/2, H)
            #ax.set_xlim(0, 500)
            #ax.semilogy()
        except KeyboardInterrupt:
            #clock_end = time.clock()
            #if not args.silent:
                #print_status()
            sys.stderr.write("Keyboard interrupt\n")
            break
    else:
        sys.stderr.write("All event sets processed\n")
    outfile.close()
    fig.savefig("test.png")

