import os

import numpy as np

os.environ['THEANO_FLAGS'] = 'floatX=float32'
import theano
import theano.tensor as tt
import theano.tensor.signal.pool  # noqa: 401
# dtype = theano.config.floatX

from run_core import load_network, SoftLIFRate, round_layer


def compute_layer(layer, inputs, data):
    assert isinstance(inputs, list)
    assert len(layer.get('inputs', [])) == len(inputs)
    print("Computing layer %s" % layer['name'])

    if layer['type'] == 'data':
        return data[layer['dataIdx']]

    if layer['type'] == 'cost.logreg':
        assert len(inputs) == 2
        labels, probs = inputs
        assert probs.ndim == 2
        assert labels.ndim == 1
        assert labels.shape[0] == probs.shape[0]
        cost = -np.log(probs)[np.arange(probs.shape[0]), labels].mean()
        inds = np.argsort(probs, axis=1)
        top1error = (inds[:, -1] != labels).mean()
        top5error = (inds[:, -5:] != labels[:, None]).all(axis=1).mean()
        return cost, top1error, top5error

    # single input layers
    assert len(inputs) == 1
    x = inputs[0]

    if layer['type'] == 'fc':
        weights = layer['weights'][0]
        return np.dot(x.reshape(x.shape[0], -1), weights) + layer['biases']
    if layer['type'] == 'neuron':
        neuron = layer['neuron']
        ntype = neuron['type']
        if ntype == 'ident':
            return x.copy()
        if ntype == 'logistic':
            return 1. / (1 + np.exp(-x))
        if ntype == 'relu':
            return np.maximum(0, x)
        if ntype.startswith('softlif'):  # includes softlifalpha and softlifalpharc
            params = neuron['params']
            if 't' not in params:
                print("WARNING: using default neuron params")
                tau_ref, tau_rc, alpha, amp = (0.001, 0.05, 0.825, 0.063)
                sigma = params.get('g', params.get('a', None))
            else:
                tau_ref, tau_rc, alpha, amp, sigma = [
                    params[k] for k in ['t', 'r', 'a', 'm', 'g']]

            lif = SoftLIFRate(sigma=sigma, tau_rc=tau_rc, tau_ref=tau_ref)
            bias = 1.
            r = amp * lif.rates(x, alpha, bias)
            # r = amp * lif.rates(x.astype(np.float32), np.float32(alpha), np.float32(bias))
            return r
        raise NotImplementedError(ntype)
    if layer['type'] == 'softmax':
        assert x.ndim == 2
        sx = tt.matrix()
        sy = tt.nnet.softmax(sx)
        f = theano.function([sx], sy)
        return f(x)
    if layer['type'] in ['dropout', 'dropout2']:
        return layer['keep'] * x  # scale all outputs by dropout factor

    # layers that need square inputs
    assert x.shape[-2] == x.shape[-1]

    if layer['type'] == 'conv':
        assert layer['sharedBiases']
        n = x.shape[0]
        nc = layer['channels'][0]
        nx = layer['imgSize'][0]
        ny = layer['modulesX']
        nf = layer['filters']
        s = layer['filterSize'][0]
        st = layer['stride'][0]
        p = -layer['padding'][0]  # Alex makes -ve in layer.py (why?)
        filters = layer['weights'][0].reshape(nc, s, s, nf)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases'].reshape(1, nf, 1, 1)
        assert x.shape == (n, nc, nx, nx)

        nx2 = (ny - 1) * st + s
        xpad = np.zeros((n, nc, nx2, nx2), dtype=x.dtype)
        xpad[:, :, p:p+nx, p:p+nx] = x
        x = xpad

        sx = tt.tensor4()
        sy = tt.nnet.conv2d(
            sx, filters, input_shape=x.shape, filter_shape=filters.shape,
            subsample=(st, st), border_mode='valid', filter_flip=False)
        sy = sy + biases
        f = theano.function([sx], sy)
        y = f(x)

        print("Abs filters (mean, std, max) %s %s %s" % (
            abs(filters).mean(), abs(filters).std(), abs(filters).max()))
        print("Abs biases (mean, std, max) %s %s %s" % (
            abs(biases).mean(), abs(biases).std(), abs(biases).max()))
        assert y.shape == (n, nf, ny, ny)

        return y

    if layer['type'] == 'local':
        n = x.shape[0]
        nc = layer['channels'][0]
        nxi = nxj = layer['imgSize'][0]
        nyi = nyj = layer['modulesX']
        nf = layer['filters']
        si = sj = layer['filterSize'][0]
        sti = stj = layer['stride'][0]
        pi = pj = -layer['padding'][0]  # Alex makes -ve in layer.py (why?)
        assert x.shape == (n, nc, nxi, nxj)

        filters = layer['weights'][0].reshape(nyi, nyj, nc, si, sj, nf)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases'].reshape(1, nf, nyi, nyj)

        y = np.zeros((n, nf, nyi, nyj), dtype=x.dtype)
        for i in range(nyi):
            for j in range(nyj):
                i0, j0 = i*sti - pi, j*stj - pj
                i1, j1 = i0 + si, j0 + sj
                sli = slice(max(-i0, 0), min(nxi + si - i1, si))
                slj = slice(max(-j0, 0), min(nxj + sj - j1, sj))
                w = filters[:, i, j, :, sli, slj].reshape(nf, -1)
                xij = x[:, :, max(i0, 0):min(i1, nxi), max(j0, 0):min(j1, nxj)]
                y[:, :, i, j] = np.dot(xij.reshape(n, -1), w.T)

        y += biases

        return y

    if layer['type'] == 'pool':
        assert layer['start'] == 0
        n = x.shape[0]
        nc = layer['channels']
        nxi = nxj = layer['imgSize']
        nyi = nyj = layer['outputsX']
        st, s = layer['stride'], layer['sizeX']
        mode = dict(max='max', avg='average_exc_pad')[layer['pool']]
        assert x.shape == (n, nc, nxi, nxj)

        sx = tt.tensor4()
        sy = tt.signal.pool.pool_2d(
            sx, (s, s), ignore_border=False, st=(st, st), mode=mode)
        f = theano.function([sx], sy)
        y = f(x)

        return y

    raise NotImplementedError(layer['type'])


def compute_target_layer(target_key, layers, data, outputs=None):
    if outputs is None:
        outputs = {}
    if target_key in outputs:
        return

    layer = layers[target_key]
    input_keys = layer.get('inputs', [])
    for input_key in input_keys:
        if input_key not in outputs:
            compute_target_layer(input_key, layers, data, outputs)

    inputs = [outputs[key] for key in input_keys]
    outputs[target_key] = compute_layer(layer, inputs, data)

    return outputs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run network in Numpy")
    parser.add_argument('loadfile', help="Checkpoint to load")
    parser.add_argument('--histsave', help="Save layer histograms")
    parser.add_argument('--n', type=int, help="Number of images to test")

    args = parser.parse_args()
    layers, data, dp = load_network(args.loadfile)

    if 0:
        # use fixed point weights
        for layer in layers.values():
            round_layer(layer, 2**8, clip_percent=0.1)

    inds = slice(0, args.n)
    data = [d[inds] for d in data]

    if 0:
        n = 10
        images = data[0]
        pimages = images[:n]
        pimages = (pimages + dp.data_mean.reshape(1, 3, 24, 24)) / 255.
        pimages = np.transpose(pimages, (0, 2, 3, 1))
        pimages = pimages.clip(0, 1)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(pimages[i], vmin=0, vmax=1)

        plt.show()

    outputs = compute_target_layer('logprob', layers, data)

    def print_acts(name):
        for parent in layers[name].get('inputs', []):
            print_acts(parent)

        output = outputs[name]
        print("%15s: %10.3f (%10.3f) [%10.3f %10.3f]" % (
            name, output.mean(), output.std(), output.min(), output.max()))

    print_acts('probs')
    print("logprob: %10.6f, top-1: %0.6f, top-5: %0.6f" % outputs['logprob'])

    if args.histsave is not None:
        hist_dict = {}
        def hist_acts(name):
            output = outputs[name]
            hist, edges = np.histogram(output.ravel(), bins=100)
            hist_dict[name] = (hist, edges)

            # compute parents
            for parent in layers[name].get('inputs', []):
                hist_acts(parent)

        hist_acts('probs')
        np.savez(args.histsave, **hist_dict)
        print("Saved %r" % args.histsave)
