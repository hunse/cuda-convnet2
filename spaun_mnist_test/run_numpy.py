import os

import numpy as np

os.environ['THEANO_FLAGS'] = 'floatX=float32'
import theano
import theano.tensor as tt
# dtype = theano.config.floatX

# from convnet import ConvNet
from run_core import load_network, SoftLIFRate, round_layer


def compute_layer(layer, inputs, data):
    assert isinstance(inputs, list)
    assert len(layer.get('inputs', [])) == len(inputs)

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
        if ntype == 'softlif':
            params = neuron['params']
            if 't' not in params:
                print("WARNING: using default neuron params")
                tau_ref = 0.001
                tau_rc = 0.05
                alpha = 0.825
                amp = 0.063
                sigma = params.get('g', params.get('a', None))
                noise = params.get('n', 0.0)
            else:
                tau_ref, tau_rc, alpha, amp, sigma, noise = [
                    np.array(params[k], dtype=np.float32)
                    for k in ['t', 'r', 'a', 'm', 'g', 'n']]

            lif = SoftLIFRate(sigma=sigma, tau_rc=tau_rc, tau_ref=tau_ref)
            bias = np.ones_like(alpha)
            r = amp * lif.rates(x.astype(np.float32), alpha, bias)
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
        assert x.shape[1:] == (nc, nx, nx)

        nx2 = (ny - 1) * st + s
        xpad = np.zeros((n, nc, nx2, nx2), dtype=x.dtype)
        xpad[:, :, p:p+nx, p:p+nx] = x
        x = xpad

        filters = layer['weights'][0].reshape(nc, s, s, nf)
        filters = np.rollaxis(filters, axis=-1, start=0)
        filters = filters[:, :, ::-1, ::-1]  # flip, since conv2d flips back
        biases = layer['biases'].reshape(1, nf, 1, 1)

        sx = tt.tensor4()
        sy = tt.nnet.conv2d(
            sx, filters, input_shape=x.shape, filter_shape=filters.shape,
            subsample=(st, st), border_mode='valid')
        sy = sy + biases
        f = theano.function([sx], sy)

        y = f(x)

        assert y.shape[1:] == (nf, ny, ny)
        return y

    if layer['type'] == 'local':
        n = x.shape[0]
        nc = layer['channels'][0]
        nx = layer['imgSize'][0]
        ny = layer['modulesX']
        nf = layer['filters']
        s = layer['filterSize'][0]
        st = layer['stride'][0]
        p = -layer['padding'][0]  # Alex makes -ve in layer.py (why?)
        assert x.shape[1:] == (nc, nx, nx)

        filters = layer['weights'][0].reshape(ny, ny, nc, s, s, nf)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases'][0].reshape(1, 1, 1, 1)

        y = np.zeros((n, nf, ny, ny), dtype=x.dtype)
        for i in range(ny):
            for j in range(ny):
                xi0, xj0 = st*i-p, st*j-p
                xi1, xj1 = xi0+s, xj0+s
                w = filters[:, i, j, :, max(-xi0, 0):min(nx+s-xi1, s),
                                        max(-xj0, 0):min(nx+s-xj1, s)]
                xij = x[:, :, max(xi0, 0):min(xi1, nx), max(xj0, 0):min(xj1, nx)]
                y[:, :, i, j] = np.dot(xij.reshape(n, -1), w.reshape(nf, -1).T)

        y += biases
        return y

    if layer['type'] == 'pool':
        assert x.shape[-2] == x.shape[-1]
        assert layer['start'] == 0
        pool = layer['pool']
        n = x.shape[0]
        nc = layer['channels']
        nx = layer['imgSize']
        ny = layer['outputsX']
        st, s = layer['stride'], layer['sizeX']
        assert x.shape == (n, nc, nx, nx)

        nx2 = ny * st
        y = x[:, :, 0:nx2:st, 0:nx2:st].copy()
        c = np.ones((ny, ny))
        assert y.shape == (n, nc, ny, ny)

        for i in range(0, s):
            for j in range(0, s):
                if i == 0 and j == 0:
                    continue

                nxi = min(nx2 + i, nx)
                nxj = min(nx2 + j, nx)
                nyi = (nx - i - 1) / st + 1
                nyj = (nx - j - 1) / st + 1
                xij, yij = x[:, :, i:nxi:st, j:nxj:st], y[:, :, :nyi, :nyj]
                if pool == 'max':
                    yij[...] = np.maximum(yij, xij)
                elif pool == 'avg':
                    yij += xij
                    c[:nyi, :nyj] += 1
                else:
                    raise NotImplementedError(pool)

        if pool == 'avg':
            y /= c

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
            # plt.imshow(pimages[i], vmin=0, vmax=1)

        plt.show()

    outputs = compute_target_layer('logprob', layers, data)

    def print_acts(name):
        for parent in layers[name].get('inputs', []):
            print_acts(parent)

        output = outputs[name]
        print("%15s: %10.3f (%10.3f) [%10.3f %10.3f]" % (
            name, output.mean(), output.std(), output.min(), output.max()))

    print_acts('probs')
    print(outputs['logprob'])

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
        # print(hist_dict)
        np.savez(args.histsave, **hist_dict)
        print("Saved %r" % args.histsave)
