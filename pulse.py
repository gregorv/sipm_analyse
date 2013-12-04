
if platform.python_implementation() == "PyPy":
    import numpypy
else:
    #import matplotlib.pyplot as plt
    from scipy.optimize import leastsq
    fig = plt.figure()
import numpy as np


def function_value(x, y, x_0):
    #idx = int((x[-1] - x[0])*len(x)*)
    idx = int((x_0 - x[0])/(x[-1] - x[0])*len(x))
    return y[idx]


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


def fit_decay(t, sig, t_ev, t_ev_next=None):
    #global plot_count
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

