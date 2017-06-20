# coding: utf-8

import sys
sys.path.insert(0, '..')

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from run_nengo import error_lims

sns.set_style('white')


class Results(object):
    def __init__(self, loadpath, dstart, dend, nstarts=None):
        self.dstart = dstart
        self.dend = dend

        self.loadname = os.path.splitext(os.path.split(loadpath)[1])[0]
        objs = np.load(loadpath)
        self.dt, self.pt, self.labels, self.t, self.y = [
            objs[k] for k in ['dt', 'pt', 'labels', 't', 'y']]
        ct_starts = dstart * np.arange(int(self.pt / dstart))
        if nstarts:
            ct_starts = ct_starts[:nstarts]
        self.ct_starts = ct_starts

        self.compute_results()

        self.error5 = False

    def compute_results(self):
        pt = self.pt
        dend = self.dend

        results = []
        for ct_start in self.ct_starts:
            ct_ends = dend * np.arange(ct_start / dend + 1, pt / dend + 1)
            errors1 = np.zeros(len(ct_ends))
            errors5 = np.zeros(len(ct_ends))
            for i, ct_end in enumerate(ct_ends):
                e1, e5, _ = error_lims(
                    self.dt, pt, self.labels, self.t, self.y,
                    ct_start=ct_start/pt, ct_end=ct_end/pt)
                errors1[i] = e1.mean()
                errors5[i] = e5.mean()
            results.append((ct_start, ct_ends, errors1, errors5))

        self.results = results

    def print_results(self):
        for ct_start, ct_ends, e1, e5 in self.results:
            errors = e5 if self.error5 else e1
            i = np.argmin(errors)
            print("Min (start/end/error): %0.2f, %0.2f, %0.2f" % (
                ct_start, ct_ends[i], 100*errors[i]))
            print("End (start/end/error): %0.2f, %0.2f, %0.2f" % (
                ct_start, ct_ends[-1], 100*errors[-1]))

            ct_end = 0.08
            i = np.argmin(np.abs(ct_ends - ct_end))
            print("Trg (start/end/error): %0.2f, %0.2f, %0.2f" % (
                ct_start, ct_ends[i], 100*errors[i]))

            # print(ct_start, errors.min())
            # ct_end = 0.06
            # print(ct_start, ct_end, errors[np.argmin(np.abs(ct_ends - ct_end))])
            # print(ct_start, ct_ends[-1], errors[-1])

    def plot_results(self, yticks=None):
        plt.figure(figsize=(5,4))
        ax = plt.gca()
        error_min = []
        for ct_start, ct_ends, e1, e5 in self.results:
            errors = e5 if self.error5 else e1
            plt.semilogy(ct_ends * 1000, errors * 100, label='%d' % (ct_start*1000,))
            error_min.append(errors.min())

        error_min = min(error_min)
        plt.xlim([0., self.pt*1000])
        plt.ylim([80*error_min, 100])

        # yticks = [10, 20, 50, 100]
        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(tick) for tick in yticks])

        plt.xlabel('classification time [ms]')
        plt.ylabel('error [%]')
        plt.legend(loc=3, title='start time [ms]')

        sns.set(context='paper', style='ticks', palette='dark')
        sns.despine()
        plt.tight_layout()

        plt.savefig(self.loadname + '-classplot.pdf')


def plot_cifar10_long():
    if 0:
        loadpath = '../checkpoints/cifar10-lif-1628-80ms_pt-0ms_alpha.npz'
        dstart = 0.01
        dend = 0.002
    elif 0:
        loadpath = '../checkpoints/cifar10-lif-1628-200ms_pt.npz'
        dstart = 0.02
        dend = 0.005
    else:
        loadpath = '../checkpoints/cifar10-lifalpharc-1556-150ms_pt-3ms_alpha.npz'
        dstart = 0.02
        dend = 0.005

    nstarts = 6
    results = Results(loadpath, dstart, dend, nstarts=nstarts)
    results.print_results()

    yticks = [10, 20, 50, 100]
    results.plot_results(yticks=yticks)

    plt.show()


def plot_cifar10_short():
    # loadpath = '../checkpoints/cifar10-lif-1628-80ms_pt-0ms_alpha.npz'
    # loadpath = '../checkpoints/cifar10-lifalpharc-1556-100ms_pt-0ms_alpha.npz'
    loadpath = '../checkpoints/cifar10-lifalpharc-1556-100ms_pt-1ms_alpha.npz'
    dstart = 0.01
    dend = 0.002
    nstarts = 6

    results = Results(loadpath, dstart, dend, nstarts=nstarts)
    results.print_results()

    yticks = [10, 20, 50, 100]
    results.plot_results(yticks=yticks)

    plt.show()


def plot_mnist():
    #loadpath = '../checkpoints/mnist-lif-0097-200ms_pt-5ms_alpha.npz'
    loadpath = '../checkpoints/mnist-lif-0097-100ms_pt-2ms_alpha.npz'
    #loadpath = '../checkpoints/mnist-lif-0097-60ms_pt-0ms_alpha.npz'

    # dstart = 0.005
    dstart = 0.01
    #dstart = 0.02
    dend = 0.002

    nstarts = 7

    results = Results(loadpath, dstart, dend, nstarts=nstarts)
    results.print_results()

    yticks = [1, 2, 5, 10, 20, 50, 100]
    results.plot_results(yticks=yticks)
    plt.show()


def plot_imagenet():
    if 1:
        loadpath = '../checkpoints/ilsvrc2012-lif-48-80ms_pt-0ms_alpha.npz'
        dstart = 0.01
        dend = 0.002
        nstarts = 7
    else:
        loadpath = '../checkpoints/ilsvrc2012-lif-48-200ms_pt-3ms_alpha.npz'
        dstart = 0.02
        dend = 0.005
        nstarts = None

    results = Results(loadpath, dstart, dend, nstarts=nstarts)
    results.error5 = True
    results.print_results()

    yticks = [20, 50, 100]
    results.plot_results(yticks=yticks)

    plt.show()


plot_cifar10_long()
# plot_cifar10_short()
# plot_mnist()
# plot_imagenet()
