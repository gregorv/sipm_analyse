
import pulse
import datafile
import sys
from matplotlib import pyplot as plt

if __name__ == "__main__":
    filename, dataset = sys.argv[1:3]
    print filename, dataset
    dataset = int(dataset)
    t, sig, _ = next(datafile.import_dta(filename, [dataset]))
    fig = plt.figure()

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(r"Zeit / ($\mathrm{ns}$)")
    ax.set_ylabel(r"Spannung / ($\mathrm{mV}$)")
    ax.plot(t, sig)
    fig.savefig("pulse_test_0.pdf")

    _, sig_smooth = pulse.integrate_signal(t, sig)
    fig.clf()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(r"Zeit / ($\mathrm{ns}$)")
    ax.set_ylabel(r"Spannung / ($\mathrm{mV}$)")
    ax.plot(t, sig_smooth)
    fig.savefig("pulse_test_1.pdf")


    a = list(filter(lambda x: (x[1]-x[0]) > 2 or not x[2], pulse.flankenize(t, sig_smooth)))
    fig.clf()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(r"Zeit / ($\mathrm{ns}$)")
    ax.set_ylabel(r"Spannung / ($\mathrm{mV}$)")
    for start, end, positive in a:
        ax.plot([t[start], t[end]],
                [sig_smooth[start], sig_smooth[end]],
                "g" if positive else "r")
    fig.savefig("pulse_test_2.pdf")


    b = filter(lambda x: not x[2] and (sig_smooth[x[1]] - sig_smooth[x[0]]) < -5, pulse.collapse_flanks(a))
    fig.clf()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(r"Zeit / ($\mathrm{ns}$)")
    ax.set_ylabel(r"Spannung / ($\mathrm{mV}$)")
    for start, end, positive in b:
        ax.plot([t[start], t[end]],
                [sig_smooth[start], sig_smooth[end]],
                "g" if positive else "r")
    fig.savefig("pulse_test_3.pdf")

    t_start = 200
    c = filter(lambda x: int(t_start/(t[-1] - t[0])*len(t)) < x[0], b)
    fig.clf()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(r"Zeit / ($\mathrm{ns}$)")
    ax.set_ylabel(r"Spannung / ($\mathrm{mV}$)")
    for start, end, positive in c:
        ax.plot([t[start], t[end]],
                [sig_smooth[start], sig_smooth[end]],
                "g" if positive else "r")
    fig.savefig("pulse_test_4.pdf")


    c = filter(lambda x: int(t_start/(t[-1] - t[0])*len(t)) < x[0], b)
    fig.clf()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(r"Zeit / ($\mathrm{ns}$)")
    ax.set_ylabel(r"Spannung / ($\mathrm{mV}$)")
    ax.plot(t, sig, "grey")
    for start, end, positive in c:
        ax.plot([t[start], t[end]],
                [sig[start], sig[end]],
                "g" if positive else "r",
                linewidth=2)
    fig.savefig("pulse_test_5.pdf")