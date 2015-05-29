import os

import numpy as np

os.environ['THEANO_FLAGS'] = 'floatX=float32'
import theano
import theano.tensor as tt
# dtype = theano.config.floatX

from convnet import ConvNet


def compute_layer(layer, inputs):
    assert isinstance(inputs, list)
    assert len(layer.get('inputs', [])) == len(inputs)
    print "Computing layer %s" % layer['name']

    if layer['type'] == 'cost.logreg':
        assert len(inputs) == 2
        labels, probs = inputs
        assert probs.ndim == 2
        assert labels.ndim == 1
        assert labels.shape[0] == probs.shape[0]
        cost = -np.log(probs)[np.arange(probs.shape[0]), labels].mean()
        error = (np.argmax(probs, axis=1) != labels).mean()
        return cost, error

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
        if ntype == 'relu':
            return np.maximum(0, x)
        if ntype == 'softlif':
            params = neuron['params']
            # tau_ref = 0.001
            # tau_rc = 0.05
            # alpha = 0.825
            # amp = 0.063
            tau_ref = params['t']
            tau_rc = params['r']
            alpha = params['a']
            amp = params['m']
            sigma = params['g']
            y = (alpha / sigma) * x
            j = sigma * np.where(y > 4.0, y, np.log1p(np.exp(y)))
            v = amp / (tau_ref + tau_rc * np.log1p(1. / j))
            return np.where(j > 0, v, 0.0)
        raise NotImplementedError(ntype)
    if layer['type'] == 'softmax':
        assert x.ndim == 2
        sx = tt.matrix()
        sy = tt.nnet.softmax(sx)
        f = theano.function([sx], sy)
        return f(x)

    # layers that need square inputs
    assert x.shape[-2] == x.shape[-1]

    if layer['type'] == 'conv':
        assert layer['sharedBiases']
        assert layer['stride'][0] == 1

        c = layer['channels'][0]
        f = layer['filters']
        s = layer['filterSize'][0]
        assert x.shape[1] == c

        filters = layer['weights'][0].reshape(c, s, s, f)
        filters = np.rollaxis(filters, axis=-1, start=0)
        filters = filters[:, :, ::-1, ::-1]  # flip
        biases = layer['biases'].reshape(1, f, 1, 1)

        sx = tt.tensor4()
        sy = tt.nnet.conv2d(
            sx, filters, image_shape=x.shape, filter_shape=filters.shape,
            border_mode='full')  # flips the kernel (performs actual conv)

        p = (s - 1) / 2
        sy = sy[:, :, p:-p, p:-p]
        sy = sy + biases

        f = theano.function([sx], sy)
        y = f(x)

        assert np.prod(y.shape[1:]) == layer['outputs']
        return y
    if layer['type'] == 'local':
        st = layer['stride'][0]
        assert st == 1

        N = x.shape[0]
        c = layer['channels'][0]
        f = layer['filters']
        s = layer['filterSize'][0]
        s2 = (s - 1) / 2
        nx = x.shape[-1]
        ny = layer['modulesX']

        filters = layer['weights'][0].reshape(ny, ny, c, s, s, f)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases'][0].reshape(1, 1, 1, 1)

        y = np.zeros((N, f, ny, ny), dtype=x.dtype)
        for i in range(ny):
            for j in range(ny):
                i0, i1 = i-s2, i+s2+1
                j0, j1 = j-s2, j+s2+1
                w = filters[:, i, j, :, max(-i0, 0):min(nx+s-i1, s),
                                        max(-j0, 0):min(nx+s-j1, s)]
                xij = x[:, :, max(i0, 0):min(i1, nx), max(j0, 0):min(j1, nx)]
                y[:, :, i, j] = np.dot(xij.reshape(N, -1), w.reshape(f, -1).T)

        y += biases
        return y
    if layer['type'] == 'pool':
        assert x.shape[-2] == x.shape[-1]
        assert layer['start'] == 0
        pool = layer['pool']
        st, s = layer['stride'], layer['sizeX']
        nx = x.shape[-1]
        ny = layer['outputsX']

        y = x[:, :, ::st, ::st].copy()
        c = np.ones((ny, ny))
        assert y.shape[-2] == y.shape[-1] and y.shape[-1] == ny

        for i in range(0, s):
            for j in range(0, s):
                if i == 0 and j == 0:
                    continue

                ni = (nx - i - 1) / st + 1
                nj = (nx - j - 1) / st + 1
                xij, yij = x[:, :, i::st, j::st], y[:, :, :ni, :nj]
                if pool == 'max':
                    yij[...] = np.maximum(yij, xij)
                elif pool == 'avg':
                    yij += xij
                    c[:ni, :nj] += 1
                else:
                    raise NotImplementedError(pool)

        if pool == 'avg':
            y /= c

        return y

    raise NotImplementedError(layer['type'])


def compute_target_layer(layers, target_key, output_dict):
    if target_key in output_dict:
        return

    layer = layers[target_key]
    if layer['type'] == 'data':
        raise ValueError("Put data layers into 'output_dict' first")

    input_keys = layer['inputs']
    for input_key in input_keys:
        if input_key not in output_dict:
            compute_target_layer(layers, input_key, output_dict)

    inputs = [output_dict[key] for key in input_keys]
    output_dict[target_key] = compute_layer(layer, inputs)


def compute_layers(layers, output_dict):
    for key in layers:
        if key not in output_dict:
            compute_target_layer(layers, key, output_dict)


if __name__ == '__main__':
    op = ConvNet.get_options_parser()
    op, load_dic = ConvNet.parse_options(op)
    net = ConvNet(op, load_dic)

    for k, v in net.layers.items():
        print k, v.get('inputs', None), v.get('outputs', None)



    dp = net.test_data_provider
    epoch, batchnum, [data, labels] = dp.get_next_batch()
    data = data.T
    data.shape = (data.shape[0], 3, 24, 24)
    labels.shape = (-1,)
    labels = labels.astype('int')

    # np.random.shuffle(labels)

    # data = data[:1000]
    # labels = labels[:1000]

    if 0:
        n = 10
        pdata = data[:n]
        pdata = (pdata + dp.data_mean.reshape(1, 3, 24, 24)) / 255.
        pdata = np.transpose(pdata, (0, 2, 3, 1))
        pdata = pdata.clip(0, 1)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(pdata[i], vmin=0, vmax=1)
            # plt.imshow(pdata[i], vmin=0, vmax=1)

        plt.show()

    output_dict = {}
    output_dict['data'] = data
    output_dict['labels'] = labels
    compute_layers(net.layers, output_dict)

    print output_dict['logprob']
