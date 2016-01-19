import matplotlib.pyplot as plt
import numpy as np
import nengo
nengo.log(level='info')
import nengo_ocl

from run_core import load_network, SoftLIFRate, round_layer


def hist_dist(hist, edges):
    p = np.zeros(edges.size, dtype=float)
    p[1:-1] = hist[:-1] + hist[1:]
    p /= p.sum()
    return nengo.dists.PDF(edges, p)


def build_layer(layer, inputs, data, hist=None, pt=None):
    assert isinstance(inputs, list)
    assert len(layer.get('inputs', [])) == len(inputs)
    name = layer['name']

    if layer['type'] == 'cost.logreg':
        assert pt is not None
        labels, probs = inputs
        nlabels, nprobs = layer['numInputs']
        assert nlabels == 1

        def label_error(t, x, labels=labels):
            return np.argmax(x) != labels[int(t / pt) % len(labels)]

        u = nengo.Node(label_error, size_in=nprobs)
        nengo.Connection(probs, u, synapse=None)
        return u

    if layer['type'] == 'data':
        if name == 'data':
            assert pt is not None
            images = data[name]
            images = images.reshape(images.shape[0], -1)
            return nengo.Node(nengo.processes.PresentInput(images, pt))
        else:
            return data[name]  # just output the raw data

    # single input layers
    assert len(inputs) == 1
    input0 = inputs[0]

    if layer['type'] == 'neuron':
        neuron = layer['neuron']
        ntype = neuron['type']
        n = layer['outputs']

        if hist is not None:
            # approximate function using NEF
            f = lambda x: np.maximum(x, 0)
            n_neurons = 5
            dist = hist_dist(*hist)

            if 0:
                # tune parameters
                e = nengo.Ensemble(n_neurons, 1, encoders=np.ones((n_neurons, 1)),
                                   intercepts=dist, eval_points=dist)
                nengo.utils.ensemble.tune_ens_parameters(e, function=f)

                earray = nengo.networks.EnsembleArray(n_neurons, n)
                for ei in earray.ensembles:
                    ei.encoders = e.encoders
                    ei.gain = e.gain
                    ei.bias = e.bias
                    ei.eval_points = ei.eval_points
            else:
                earray = nengo.networks.EnsembleArray(
                    n_neurons, n, eval_points=dist, intercepts=dist)

            nengo.Connection(input0, earray.input)
            if ntype == 'relu':
                earray.add_output('relu', f, synapse=None)
                return earray.relu
            raise NotImplementedError(ntype)

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
            if 't' not in params:
                print("Warning: Using default neuron params")
                tau_ref = 0.001
                tau_rc = 0.05
                alpha = 0.825
                amp = 0.063
                sigma = params.get('g', params.get('a', None))
                noise = params.get('n', 0.0)
            else:
                tau_ref, tau_rc, alpha, amp, sigma, noise = [
                    params[k] for k in ['t', 'r', 'a', 'm', 'g', 'n']]

            # e.neuron_type = SoftLIFRate(sigma=sigma, tau_rc=tau_rc, tau_ref=tau_ref)
            # e.neuron_type = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)
            e.neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
            e.gain = alpha * np.ones(n)
            e.bias = 1. * np.ones(n)
            u = nengo.Node(size_in=n)
            nengo.Connection(e.neurons, u, transform=amp, synapse=None)
            return u
        raise NotImplementedError(ntype)
    if layer['type'] == 'softmax':
        # do nothing for now
        return input0
    if layer['type'] == 'dropout':
        u = nengo.Node(size_in=layer['outputs'])
        nengo.Connection(input0, u, transform=layer['keep'])
        return u
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
        n = int(np.sqrt(layer['numInputs'][0] / c))
        assert n == layer['modulesX']

        filters = layer['weights'][0].reshape(c, s, s, f)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases']
        u = nengo.Node(nengo.processes.Conv2((c, n, n), filters, biases))
        nengo.Connection(input0, u)
        return u
    if layer['type'] == 'local':
        st = layer['stride'][0]
        assert st == 1

        c = layer['channels'][0]
        f = layer['filters']
        s = layer['filterSize'][0]
        s2 = (s - 1) / 2
        n = int(np.sqrt(layer['numInputs'][0] / c))
        assert n == layer['modulesX']

        filters = layer['weights'][0].reshape(n, n, c, s, s, f)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases'][0].reshape(1, 1, 1)
        u = nengo.Node(nengo.processes.Conv2((c, n, n), filters, biases))
        nengo.Connection(input0, u)
        return u
    if layer['type'] == 'pool':
        assert layer['start'] == 0
        pooltype = layer['pool']
        assert pooltype == 'avg'
        s = layer['sizeX']
        st = layer['stride']
        c = layer['channels']
        nx = layer['imgSize']

        u = nengo.Node(nengo.processes.Pool2((c, nx, nx), s, stride=st))
        nengo.Connection(input0, u, synapse=None)
        return u

    raise NotImplementedError(layer['type'])


def build_target_layer(target_key, layers, data, network, outputs=None, hists=None, pt=None):
    if outputs is None:
        outputs = {}
    elif target_key in outputs:
        return outputs

    layer = layers[target_key]
    input_keys = layer.get('inputs', [])
    for input_key in input_keys:
        if input_key not in outputs:
            build_target_layer(input_key, layers, data, network, outputs, hists=hists, pt=pt)

    inputs = [outputs[key] for key in input_keys]
    with network:
        hist = hists[target_key] if target_key in hists else None
        outputs[target_key] = build_layer(layer, inputs, data, hist=hist, pt=pt)

    return outputs


def run(loadfile, savefile=None, multiview=None, histload=None,
        count_spikes=False):
    assert not multiview

    layers, data = load_network(loadfile, multiview)
    hists = np.load(histload) if histload is not None else {}

    if 0:
        # use fixed point weights
        for layer in layers.values():
            round_layer(layer, 2**8, clip_percent=0)

    # --- build network in Nengo
    network = nengo.Network()

    # presentation_time = 0.02
    # presentation_time = 0.03
    # presentation_time = 0.04
    # presentation_time = 0.05
    presentation_time = 0.08
    # presentation_time = 0.1
    # presentation_time = 0.13
    # presentation_time = 0.15
    # presentation_time = 0.2

    network.config[nengo.Connection].synapse = nengo.synapses.Lowpass(0.0)
    # network.config[nengo.Connection].synapse = nengo.synapses.Lowpass(0.001)
    # network.config[nengo.Connection].synapse = nengo.synapses.Lowpass(0.005)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.001)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.002)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.003)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.004)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.005)

    outputs = build_target_layer('logprob', layers, data, network, hists=hists,
                                 pt=presentation_time)

    # test whole network
    with network:
        # xp = nengo.Probe(outputs['data'], synapse=None)
        yp = nengo.Probe(outputs['fc10'], synapse=None)
        zp = nengo.Probe(outputs['logprob'], synapse=None)

        if count_spikes:
            spikes_p = {}
            for name in layers:
                if layers[name]['type'] == 'neuron':
                    # node outputs scaled spikes
                    spikes_p[name] = nengo.Probe(outputs[name])

    if 0:
        # profile
        sim = nengo_ocl.Simulator(network, profiling=True)
        # sim.run(0.005)
        sim.run(10 * presentation_time)
        sim.print_profiling()
    else:
        # n = 3
        # n = 20
        # n = 100
        # n = 1000
        n = data['labels'].size  # test on all examples

        print("Running %d examples for %0.3f s each" % (n, presentation_time))
        sim = nengo_ocl.Simulator(network)
        sim.run(n * presentation_time)

    dt = sim.dt
    t = sim.trange()
    y = sim.data[yp]
    z = sim.data[zp]

    get_ind = lambda t: int(t / presentation_time)
    inds = slice(0, get_ind(t[-2]) + 1)
    images = data['data'][inds]
    labels = data['labels'][inds]
    data_mean = data['data_mean']
    label_names = data['label_names']

    spikes = None
    if count_spikes:
        trange = float(t[-1] - t[0])
        spikes = {k: sim.data[v] for k, v in spikes_p.items()}
        counts = [(v > 0).sum() / trange for v in spikes.values()]
        neurons = [v.shape[1] for v in spikes.values()]
        rates = [c / n for c, n in zip(counts, neurons)]
        print("Spike rates: {%s}" % ', '.join(
            "%s: %0.1f" % (k, r) for k, r in zip(spikes.keys(), rates)))
        print("Spike rate [Hz]: %0.3f" % (sum(counts) / sum(neurons)))

    if savefile is not None:
        np.savez(savefile,
                 images=images, labels=labels,
                 data_mean=data_mean, label_names=label_names,
                 dt=dt, pt=presentation_time, t=t, y=y, z=z, spikes=spikes)
        print("Saved '%s'" % savefile)

    # view(dt, images, labels, data_mean, label_names, t, y, z)
    errors, _, _ = error(dt, presentation_time, labels, t, y, z)
    print("Error: %f (%d samples)" % (errors.mean(), errors.size))


def error(dt, pt, labels, t, y, z):
    # filter outputs (better accuracy)
    # s = nengo.synapses.Alpha(0.002)  # 30ms_pt-0ms_alpha
    # s = nengo.synapses.Alpha(0.004)
    # s = nengo.synapses.Alpha(0.005)
    s = nengo.synapses.Alpha(0.01)
    # s = nengo.synapses.Alpha(0.02)
    y = nengo.synapses.filt(y, s, dt)
    # y = nengo.synapses.filtfilt(y, s, dt)

    # ct = 0.01  # classification time
    ct = 0.04  # classification time

    # take average class over last 10 ms of each presentation
    pn = int(pt / dt)
    cn = int(ct / dt)
    n = y.shape[0] / pn
    assert cn <= pn

    blocks = y.reshape(n, pn, y.shape[1])[:, -cn:, :]
    labels = labels[:n]
    assert blocks.shape[0] == labels.shape[0]
    errors = np.argmax(blocks.mean(1), axis=1) != labels
    return errors, y, z


def view(dt, pt, images, labels, data_mean, label_names, t, y, z, n_max=30):

    get_ind = lambda t: int(t / pt)

    # clip if more than n_max
    i_max = int(n_max * pt / dt)
    t = t[:i_max]
    y = y[:i_max]
    z = z[:i_max]

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
    if len(label_names) <= 10:
        plt.legend(label_names, fontsize=8, loc=2)

    plt.subplot(rows, cols, 3)
    plt.plot(t, z)
    plt.xlim([t[0], t[-1]])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('error (0 == correct)')

    plt.show()


if __name__ == '__main__':
    # test_softlifrate()
    # assert False

    import argparse
    parser = argparse.ArgumentParser(description="Run network in Nengo")
    # parser.add_argument('--multiview', action='store_const', const=1, default=None)
    parser.add_argument('loadfile', help="Checkpoint to load")
    parser.add_argument('savefile', nargs='?', default=None, help="Where to save output")
    parser.add_argument('--histload', help="Layer histograms created by run_numpy")
    parser.add_argument('--spikes', action='store_true', help="Count spikes")

    args = parser.parse_args()
    run(args.loadfile, args.savefile, histload=args.histload,
        count_spikes=args.spikes)
