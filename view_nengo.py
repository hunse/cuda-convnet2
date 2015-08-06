import argparse
# import sys

import numpy as np
from run_nengo import error, view

parser = argparse.ArgumentParser(description="View Nengo run results.")
parser.add_argument('loadfile', help="File to load")
parser.add_argument('--view', action='store_const', const=True, default=False)
parser.add_argument('--spikes', action='store_const', const=True, default=False)
args = parser.parse_args()

# assert len(sys.argv) == 2
# loadfile = sys.argv[1]
objs = np.load(args.loadfile)

kwargs = {}
for n in ['dt', 'images', 'labels', 'data_mean', 'label_names', 't', 'y', 'z']:
    kwargs[n] = objs[n]

errors, y, z = error(*[kwargs[n] for n in ('dt', 'labels', 't', 'y', 'z')])
print "Error: %f (%d samples)" % (errors.mean(), errors.size)
kwargs['y'] = y
kwargs['z'] = z

if args.view:
    view(**kwargs)

if args.spikes:
    import matplotlib.pyplot as plt

    t = objs['t']
    spikes = objs['spikes'].item()

    keys = sorted(spikes)
    values = [spikes[k] for k in keys]

    # sum spikes from all layers
    n_spikes = sum((s > 0).sum() for s in values)

    # print spikes.values()[0][1050:1090, :12] > 0

    n_neurons = sum(s.shape[1] for s in values)
    print n_neurons

    ttotal = t[-1] - t[0]

    print "Spike rate: %f spikes/s" % (float(n_spikes) / n_neurons / ttotal)


    # n_spikes = np.hstack([(s > 0).sum(axis=0) for s in spikes.values()])
    # print n_spikes.mean(), n_spikes.std()

    n_spikes = [(s > 0).sum(axis=0) / ttotal for s in values]
    n_neurons = [s.shape[1] for s in values]

    print keys
    print [s.mean() for s in n_spikes]
    print [s.std() for s in n_spikes]
    print n_neurons
