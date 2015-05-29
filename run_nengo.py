import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import nengo

# os.environ['THEANO_FLAGS'] = 'floatX=float32'
# import theano
# import theano.tensor as tt
# dtype = theano.config.floatX
# dtype = 'float32'

# from convnet import ConvNet
from convdata import DataProvider, CIFARDataProvider
from python_util.gpumodel import IGPUModel

presentation_time = 0.15
get_ind = lambda t: int(t / presentation_time)


def softrelu(x, sigma=1.):
    y = x / sigma
    z = np.array(x)
    z[y < 34.0] = sigma * np.log1p(np.exp(y[y < 34.0]))
    # ^ 34.0 gives exact answer in 32 or 64 bit but doesn't overflow in 32 bit
    return z


class SoftLIFRate(nengo.neurons.LIFRate):
    sigma = nengo.params.NumberParam(low=0)

    def __init__(self, sigma=1., **lif_args):
        super(SoftLIFRate, self).__init__(**lif_args)
        self.sigma = sigma

    @property
    def _argreprs(self):
        args = super(SoftLIFRate, self)._argreprs
        if self.sigma != 1.:
            args.append("sigma=%s" % self.sigma)
        return args

    def rates(self, x, gain, bias):
        J = gain * x + bias
        out = np.zeros_like(J)
        SoftLIFRate.step_math(self, dt=1, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Compute rates in Hz for input current (incl. bias)"""
        # x = (J - 1) / self.sigma
        # j = self.sigma * np.where(x > 4.0, x, np.log1p(np.exp(x)))
        j = softrelu(J - 1, sigma=self.sigma)
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = 1. / (
            self.tau_ref + self.tau_rc * np.log1p(1. / j[j > 0]))


def test_softlifrate():
    neurons = SoftLIFRate(sigma=0.002, tau_rc=0.02, tau_ref=0.002)
    x = np.linspace(-1, 1, 101)
    # j = 0.8 * x + 1.
    r = neurons.rates(x, 1., 1.)
    plt.plot(x, r)
    plt.show()


def build_layer(layer, inputs, data):
    assert isinstance(inputs, list)
    assert len(layer.get('inputs', [])) == len(inputs)
    name = layer['name']

    if layer['type'] == 'cost.logreg':
        labels, probs = inputs
        nlabels, nprobs = layer['numInputs']
        assert nlabels == 1

        def label_error(t, x, labels=labels):
            return np.argmax(x) != labels[get_ind(t)]

        u = nengo.Node(label_error, size_in=nprobs)
        nengo.Connection(probs, u, synapse=None)
        return u
        # return nengo.Probe(u)

    if layer['type'] == 'data':
        if name == 'data':
            def image_output(t, images=data[name]):
                return images[get_ind(t)].ravel()

            return nengo.Node(image_output)
        else:
            return data[name]  # just output the raw data

    # single input layers
    assert len(inputs) == 1
    input0 = inputs[0]

    if layer['type'] == 'neuron':
        neuron = layer['neuron']
        ntype = neuron['type']
        n = layer['outputs']

        e = nengo.Ensemble(n, 1)
        nengo.Connection(input0, e.neurons)

        if ntype == 'ident':
            e.neuron_type = nengo.Direct()
            return e.neurons
        if ntype == 'relu':
            e.neuron_type = nengo.RectifiedLinear()
            e.gain = 1 * np.ones(n)
            e.bias = 0 * np.ones(n)
            return e.neurons
        if ntype == 'softlif':
            params = neuron['params']
            # import pdb; pdb.set_trace()
            print neuron
            # tau_rc, tau_ref = neuron.params['r'], neuron.params

            e.neuron_type = SoftLIFRate(sigma=params['g'], tau_rc=params['r'], tau_ref=params['t'])
            # e.neuron_type = nengo.LIFRate(tau_rc=params['r'], tau_ref=params['t'])
            # e.neuron_type = nengo.LIF(tau_rc=params['r'], tau_ref=params['t'])
            e.gain = params['a'] * np.ones(n)
            e.bias = 1. * np.ones(n)
            u = nengo.Node(size_in=n)
            nengo.Connection(e.neurons, u, transform=params['m'], synapse=None)
            return u
        raise NotImplementedError(ntype)
    if layer['type'] == 'softmax':
        # do nothing for now
        return input0
    if layer['type'] == 'fc':
        weights = layer['weights'][0]
        biases = layer['biases'].ravel()
        u = nengo.Node(size_in=layer['outputs'])
        b = nengo.Node(output=biases)
        nengo.Connection(input0, u, transform=weights.T)
        nengo.Connection(b, u, synapse=None)
        return u
    if layer['type'] == 'conv':
        assert layer['sharedBiases']
        assert layer['stride'][0] == 1

        c = layer['channels'][0]
        f = layer['filters']
        s = layer['filterSize'][0]
        s2 = (s - 1) / 2
        nx = int(np.sqrt(layer['numInputs'][0] / c))
        ny = layer['modulesX']

        filters = layer['weights'][0].reshape(c, s, s, f)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases'].reshape(f, 1, 1)

        def conv(_, x, filters=filters, biases=biases):
            x = x.reshape(c, nx, nx)
            y = np.zeros((f, ny, ny))
            for i in range(ny):
                for j in range(ny):
                    i0, i1 = i-s2, i+s2+1
                    j0, j1 = j-s2, j+s2+1
                    w = filters[:, :, max(-i0, 0):min(nx+s-i1, s),
                                      max(-j0, 0):min(nx+s-j1, s)]
                    xij = x[:, max(i0, 0):min(i1, nx), max(j0, 0):min(j1, nx)]
                    y[:, i, j] = np.dot(xij.ravel(), w.reshape(f, -1).T)

            y += biases
            return y.ravel()

        u = nengo.Node(conv, size_in=layer['numInputs'][0])
        nengo.Connection(input0, u)
        return u
    if layer['type'] == 'local':
        st = layer['stride'][0]
        assert st == 1

        c = layer['channels'][0]
        f = layer['filters']
        s = layer['filterSize'][0]
        s2 = (s - 1) / 2
        nx = int(np.sqrt(layer['numInputs'][0] / c))
        ny = layer['modulesX']

        filters = layer['weights'][0].reshape(ny, ny, c, s, s, f)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases'][0].reshape(1, 1, 1)

        def local(_, x, filters=filters, biases=biases):
            x = x.reshape(c, nx, nx)
            y = np.zeros((f, ny, ny), dtype=x.dtype)
            for i in range(ny):
                for j in range(ny):
                    i0, i1 = i-s2, i+s2+1
                    j0, j1 = j-s2, j+s2+1
                    w = filters[:, i, j, :, max(-i0, 0):min(nx+s-i1, s),
                                            max(-j0, 0):min(nx+s-j1, s)]
                    xij = x[:, max(i0, 0):min(i1, nx), max(j0, 0):min(j1, nx)]
                    y[:, i, j] = np.dot(xij.ravel(), w.reshape(f, -1).T)

            y += biases
            return y.ravel()

        u = nengo.Node(local, size_in=layer['numInputs'][0])
        nengo.Connection(input0, u)
        return u
    if layer['type'] == 'pool':
        assert layer['start'] == 0
        pooltype = layer['pool']
        st = layer['stride']
        s = layer['sizeX']
        c = layer['channels']
        nx = layer['imgSize']
        ny = layer['outputsX']

        def pool(_, x):
            x = x.reshape(c, nx, nx)
            y = x[:, ::st, ::st].copy()
            n = np.zeros((ny, ny))
            assert y.shape[-2] == ny and y.shape[-1] == ny

            for i in range(0, s):
                for j in range(0, s):
                    ni = (nx - i - 1) / st + 1
                    nj = (nx - j - 1) / st + 1
                    xij, yij = x[:, i::st, j::st], y[:, :ni, :nj]
                    if pooltype == 'max':
                        yij[...] = np.maximum(yij, xij)
                    elif pooltype == 'avg':
                        yij += xij
                        n[:ni, :nj] += 1
                    else:
                        raise NotImplementedError(pool)

            if pooltype == 'avg':
                y /= n

            return y.ravel()

        u = nengo.Node(pool, size_in=layer['numInputs'][0])
        nengo.Connection(input0, u, synapse=None)
        return u

    raise NotImplementedError(layer['type'])


def build_target_layer(target_key, layers, data, network, outputs=None):
    if outputs is None:
        outputs = {}
    elif target_key in outputs:
        return outputs

    layer = layers[target_key]
    input_keys = layer.get('inputs', [])
    for input_key in input_keys:
        if input_key not in outputs:
            build_target_layer(input_key, layers, data, network, outputs)

    inputs = [outputs[key] for key in input_keys]
    with network:
        outputs[target_key] = build_layer(layer, inputs, data)

    return outputs


def load_network(loadfile, multiview=None):
    load_dic = IGPUModel.load_checkpoint(loadfile)
    layers = load_dic['model_state']['layers']
    op = load_dic['op']

    options = {}
    for o in load_dic['op'].get_options_list():
        options[o.name] = o.value

    dp_params = {}
    for v in ('color_noise', 'multiview_test', 'inner_size', 'scalar_mean', 'minibatch_size'):
        dp_params[v] = options[v]

    if multiview is not None:
        dp_params['multiview_test'] = multiview

    # for k, v in layers.items():
    #     print k, v.get('inputs', None), v.get('outputs', None)

    # print dp_params
    dp = DataProvider.get_instance(
        options['data_path'],
        batch_range=options['test_batch_range'],
        type=options['dp_type'],
        dp_params=dp_params, test=True)

    epoch, batchnum, [images, labels] = dp.get_next_batch()
    images = images.T
    images.shape = (images.shape[0], 3, 24, 24)
    labels.shape = (-1,)
    labels = labels.astype('int')

    if 1:
        rng = np.random.RandomState(8)
        i = rng.permutation(images.shape[0])
        images = images[i]
        labels = labels[i]

    data = {}
    data['data'] = images
    data['labels'] = labels
    data['data_mean'] = dp.data_mean
    data['label_names'] = dp.batch_meta['label_names']

    return layers, data


def run_layer(loadfile):
    from run_numpy import compute_target_layer
    from nengo.utils.numpy import rms

    layers, data = load_network(loadfile)

    network = nengo.Network()
    network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.005)

    outputs = build_target_layer('logprob', layers, data, network)

    # test a particular layer
    key = 'conv1'

    with network:
        yp = nengo.Probe(outputs[key])

    sim = nengo.Simulator(network)
    sim.run(0.01)

    outputs = {}
    outputs['data'] = images[:1]
    outputs['labels'] = labels[:1]
    compute_target_layer(layers, key, outputs)
    yref = outputs[key][0]
    y = sim.data[yp][-1].reshape(yref.shape)

    plt.figure()
    plt.subplot(121)
    plt.imshow(yref[0])
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(y[0])
    plt.colorbar()
    plt.show()

    print rms(y - yref)

    assert np.allclose(yref, y)


def run(loadfile, savefile=None, multiview=None):
    layers, data = load_network(loadfile, multiview)

    network = nengo.Network()
    network.config[nengo.Connection].synapse = nengo.synapses.Lowpass(0.0)
    # network.config[nengo.Connection].synapse = nengo.synapses.Lowpass(0.005)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.003)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.005)

    outputs = build_target_layer('logprob', layers, data, network)

    # test whole network
    with network:
        # xp = nengo.Probe(outputs['data'], synapse=None)
        yp = nengo.Probe(outputs['fc10'], synapse=None)
        zp = nengo.Probe(outputs['logprob'], synapse=None)

    sim = nengo.Simulator(network)
    # sim.run(3 * presentation_time)
    sim.run(20 * presentation_time)
    # sim.run(100 * presentation_time)
    # sim.run(1000 * presentation_time)

    dt = sim.dt
    t = sim.trange()
    y = sim.data[yp]
    z = sim.data[zp]

    # inds = slice(0, get_ind(t[-2]) + 1)
    inds = slice(0, get_ind(t[-1]) + 1)
    images = data['data'][inds]
    labels = data['labels'][inds]
    data_mean = data['data_mean']
    label_names = data['label_names']

    if savefile is not None:
        np.savez(savefile,
                 images=images, labels=labels,
                 data_mean=data_mean, label_names=label_names,
                 dt=dt, t=t, y=y, z=z)
        print("Saved '%s'" % savefile)

    # view(dt, images, labels, data_mean, label_names, t, y, z)
    errors, y, z = error(dt, labels, t, y, z)
    print("Error: %f" % errors.mean())


def error(dt, labels, t, y, z):
    s = nengo.synapses.Alpha(0.01)
    y = nengo.synapses.filtfilt(y, s, dt)

    if 0:
        z = nengo.synapses.filtfilt(z, s, dt)
    else:
        lt = labels[(t / presentation_time).astype(int)]
        z = (np.argmax(y, axis=1) != lt)

    # take 10 ms at end of each presentation
    blocks = z.reshape(-1, int(presentation_time / dt))[:, -10:]
    errors = blocks.mean(1) > 0.4
    return errors, y, z


def view(dt, images, labels, data_mean, label_names, t, y, z):

    # plot
    plt.figure()
    c, m, n = images.shape[1:]
    inds = slice(0, get_ind(t[-2]) + 1)
    imgs = images[inds]
    allimage = np.zeros((c, m, n * len(imgs)))
    for i, img in enumerate(imgs):
        img = (img + data_mean.reshape(1, c, m, n)) / 255.
        allimage[:, :, i * n:(i + 1) * n] = img.clip(0, 1)

    allimage = np.transpose(allimage, (1, 2, 0))

    rows, cols = 3, 1
    plt.subplot(rows, cols, 1)
    plt.imshow(allimage, vmin=0, vmax=1)

    plt.subplot(rows, cols, 2)
    plt.plot(t, y)
    plt.xlim([t[0], t[-1]])
    plt.legend(label_names, fontsize=8, loc=2)

    plt.subplot(rows, cols, 3)
    plt.plot(t, z)
    plt.xlim([t[0], t[-1]])
    plt.ylim([-0.1, 1.1])

    plt.show()


if __name__ == '__main__':
    # test_softlifrate()
    # assert False

    import argparse
    parser = argparse.ArgumentParser(description="Run network in Nengo")
    parser.add_argument('--multiview', action='store_const', const=1, default=None)
    parser.add_argument('loadfile', help="Checkpoint to load")
    parser.add_argument('savefile', nargs='?', default=None, help="Where to save output")

    args = parser.parse_args()
    savefile = args.loadfile.rstrip('/') + '.npz' if args.savefile is None else args.savefile
    run(args.loadfile, savefile, args.multiview)
