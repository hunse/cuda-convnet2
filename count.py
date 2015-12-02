import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_ocl

from run_core import load_network, SoftLIFRate

presentation_time = 0.15
get_ind = lambda t: int(t / presentation_time)


def hist_dist(hist, edges):
    p = np.zeros(edges.size, dtype=float)
    p[1:-1] = hist[:-1] + hist[1:]
    p /= p.sum()
    return nengo.dists.PDF(edges, p)


def count_layer(layer, counts, hist=None):
    name = layer['name']
    lcounts = counts.setdefault(name, {})

    if layer['type'] == 'cost.logreg':
        return

    if layer['type'] == 'data':
        lcounts['out_size'] = layer['outputs']
        return

    # single input layers
    if layer['type'] == 'neuron':
        lcounts['n_neurons'] = layer['outputs']
        # basename = name.rstrip('_neurons')
        # counts.setdefault(basename, {})['n_neurons'] = layer['outputs']
        return

    if layer['type'] == 'softmax':
        return

    if layer['type'] == 'dropout':
        return

    if layer['type'] == 'fc':
        assert len(layer['weights']) == 1
        weights = layer['weights'][0]
        lcounts['in_size'] = weights.shape[0]
        lcounts['out_size'] = weights.shape[1]

        lcounts['n_weights'] = weights.size
        lcounts['n_biases'] = layer['biases'].size
        lcounts['n_synapses'] = lcounts['n_weights']
        lcounts['n_full'] = lcounts['in_size'] * lcounts['out_size']
        return

    if layer['type'] == 'conv':
        assert len(layer['weights']) == 1
        lcounts['n_weights'] = layer['weights'][0].size
        lcounts['n_biases'] = layer['biases'].size

        assert layer['sharedBiases']
        assert layer['stride'][0] == 1

        c = layer['channels'][0]
        f = layer['filters']
        s = layer['filterSize'][0]
        n = int(np.sqrt(layer['numInputs'][0] / c))
        assert n == layer['modulesX']

        lcounts['in_size'] = layer['numInputs'][0]
        lcounts['in_shape'] = (c, n, n)
        lcounts['out_size'] = f * n * n
        lcounts['out_shape'] = (f, n, n)

        lcounts['n_synapses'] = n**2 * c * s**2 * f
        lcounts['n_full'] = lcounts['in_size'] * lcounts['out_size']
        return

    if layer['type'] == 'local':
        assert len(layer['weights']) == 1
        lcounts['n_weights'] = layer['weights'][0].size
        lcounts['n_biases'] = layer['biases'].size

        st = layer['stride'][0]
        assert st == 1

        c = layer['channels'][0]
        f = layer['filters']
        s = layer['filterSize'][0]

        n = int(np.sqrt(layer['numInputs'][0] / c))
        assert n == layer['modulesX']

        lcounts['in_size'] = c * n * n
        lcounts['in_shape'] = (c, n, n)
        lcounts['out_size'] = f * n * n
        lcounts['out_shape'] = (f, n, n)

        lcounts['n_synapses'] = n**2 * c * s**2 * f
        lcounts['n_full'] = lcounts['in_size'] * lcounts['out_size']
        return

    if layer['type'] == 'pool':
        assert layer['start'] == 0
        pooltype = layer['pool']
        s = layer['sizeX']
        st = layer['stride']
        c = layer['channels']
        nx = layer['imgSize']
        ny = (nx - 1) / st + 1
        # print(layer)

        lcounts['in_size'] = c * nx * nx
        lcounts['in_shape'] = (c, nx, nx)
        lcounts['out_size'] = c * ny * ny
        lcounts['out_shape'] = (c, ny, ny)
        return

    raise NotImplementedError(layer['type'])


def count_target_layer(target_key, layers, counts, hists={}):
    depth = counts.setdefault(target_key, {}).setdefault('depth', 0)

    # if outputs is None:
    #     outputs = {}
    # elif target_key in outputs:
    #     return outputs

    layer = layers[target_key]
    input_keys = layer.get('inputs', [])
    for input_key in input_keys:
        counts.setdefault(input_key, {})
        counts[input_key]['depth'] = max(
            counts[input_key].get('depth', 0), depth + 1)
        count_target_layer(input_key, layers, counts, hists=hists)

    # --- count this layer
    hist = hists[target_key] if target_key in hists else None
    count_layer(layer, counts, hist=hist)


def count(loadfile, histfile=None):
    layers, data = load_network(loadfile)
    hists = np.load(histfile) if histfile is not None else {}

    counts = {}
    count_target_layer('logprob', layers, counts, hists=hists)

    layer_keys = sorted(counts, key=lambda k: counts[k]['depth'], reverse=True)
    for key in layer_keys:
        print("%s:" % key)
        for k in sorted(counts[key]):
            if k in ['depth']:
                continue
            print("  %s: %s" % (k, counts[key][k]))

    print("totals:")
    keys = ['n_weights', 'n_synapses', 'n_full', 'n_neurons']
    for key in keys:
        entries = [counts[lk][key] for lk in layer_keys if key in counts[lk]]
        print("  %s: %s -> %s" % (key, entries, sum(entries)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Network statistics")
    parser.add_argument('loadfile', help="Checkpoint to load")
    # parser.add_argument('--histload', help="Layer histograms created by run_numpy")

    args = parser.parse_args()
    count(args.loadfile)
    # count(args.loadfile, histfile=args.histload)