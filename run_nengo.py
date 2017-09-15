from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import nengo
nengo.log(level='info')
import nengo_extras

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
        labels, probs = inputs
        return probs

        # pt = params['presentation_time']
        # nlabels, nprobs = layer['numInputs']
        # assert nlabels == 1

        # def label_error(t, x, labels=labels):
        #     return np.argmax(x) != labels[int(t / pt) % len(labels)]

        # u = nengo.Node(label_error, size_in=nprobs, label=name)
        # nengo.Connection(probs, u, synapse=None)
        # return u

    if layer['type'] == 'data':
        if layer['dataIdx'] == 0:
            assert pt is not None
            images = data[0]
            images = images.reshape(images.shape[0], -1)
            return nengo.Node(nengo.processes.PresentInput(images, pt),
                              label=name)
        else:
            return data[layer['dataIdx']]  # just output the raw data

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

        e = nengo.Ensemble(n, 1, label=name)
        nengo.Connection(input0, e.neurons)

        if ntype == 'ident':
            e.neuron_type = nengo.Direct()
            return e.neurons
        if ntype == 'relu':
            e.neuron_type = nengo.RectifiedLinear()
            e.gain = 1 * np.ones(n)
            e.bias = 0 * np.ones(n)
            return e.neurons
        if ntype.startswith('softlif'):  # includes softlifalpha and softlifalpharc
            params = neuron['params']
            if 't' not in params:
                print("Warning: Using default neuron params")
                tau_ref, tau_rc, alpha, amp = (0.001, 0.05, 0.825, 0.063)
                sigma = params.get('g', params.get('a', None))
            else:
                tau_ref, tau_rc, alpha, amp, sigma = [
                    params[k] for k in ['t', 'r', 'a', 'm', 'g']]

            # e.neuron_type = SoftLIFRate(sigma=sigma, tau_rc=tau_rc, tau_ref=tau_ref)
            # e.neuron_type = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)
            e.neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
            e.gain = alpha * np.ones(n)
            e.bias = 1. * np.ones(n)
            u = nengo.Node(size_in=n, label='%s_out' % name)
            nengo.Connection(e.neurons, u, transform=amp, synapse=None)
            return u
        raise NotImplementedError(ntype)
    if layer['type'] == 'softmax':
        # do nothing for now
        return input0
    if layer['type'] in ['dropout', 'dropout2']:
        u = nengo.Node(size_in=layer['outputs'], label=name)
        nengo.Connection(input0, u, transform=layer['keep'])
        return u
    if layer['type'] == 'fc':
        weights = layer['weights'][0]
        biases = layer['biases'].ravel()
        u = nengo.Node(size_in=layer['outputs'], label=name)
        b = nengo.Node(output=biases, label='%s_biases' % name)
        nengo.Connection(input0, u, transform=weights.T)
        nengo.Connection(b, u, synapse=None)
        return u
    if layer['type'] == 'conv':
        assert layer['sharedBiases']

        nc = layer['channels'][0]
        nx = layer['imgSize'][0]
        ny = layer['modulesX']
        nf = layer['filters']
        s = layer['filterSize'][0]
        st = layer['stride'][0]
        p = -layer['padding'][0]  # Alex makes -ve in layer.py (why?)

        filters = layer['weights'][0].reshape(nc, s, s, nf)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases']
        u = nengo.Node(nengo_extras.Conv2d(
            (nc, nx, nx), filters, biases, strides=st, padding=p), label=name)
        nengo.Connection(input0, u)
        return u
    if layer['type'] == 'local':
        nc = layer['channels'][0]
        nx = layer['imgSize'][0]
        ny = layer['modulesX']
        nf = layer['filters']
        s = layer['filterSize'][0]
        st = layer['stride'][0]
        p = -layer['padding'][0]  # Alex makes -ve in layer.py (why?)

        filters = layer['weights'][0].reshape(ny, ny, nc, s, s, nf)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases'][0].reshape(1, 1, 1)
        u = nengo.Node(nengo_extras.Conv2d(
            (nc, nx, nx), filters, biases, strides=st, padding=p), label=name)
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

        u = nengo.Node(nengo_extras.Pool2d(
            (c, nx, nx), s, strides=st), label=name)
        nengo.Connection(input0, u, synapse=None)
        return u

    raise NotImplementedError(layer['type'])


def build_target_layer(target_key, layers, data, network, outputs=None, hists=None, pt=None):
    hists = {} if hists is None else hists
    outputs = {} if outputs is None else outputs
    if target_key in outputs:
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


def run(loadfile, savefile=None, histload=None, count_spikes=False,
        n_max=None, backend='nengo', ocl_profile=False,
        presentation_time=None, synapse_type=None, synapse_tau=None):

    if backend == 'nengo':
        Simulator = nengo.Simulator
    elif backend == 'nengo_ocl':
        import nengo_ocl
        Simulator = nengo_ocl.Simulator
    else:
        raise ValueError("Unsupported backend %r" % backend)

    layers, data, dp = load_network(loadfile, sort_layers=True)
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
    # presentation_time = 0.06
    # presentation_time = 0.08
    # presentation_time = 0.1
    # presentation_time = 0.13
    # presentation_time = 0.15
    # presentation_time = 0.2

    synapse = None
    if synapse_type == 'lowpass':
        synapse = nengo.synapses.Lowpass(synapse_tau)
    elif synapse_type == 'alpha':
        synapse = nengo.synapses.Alpha(synapse_tau)
    else:
        raise ValueError("synapse type: %r" % synapse_type)

    network.config[nengo.Connection].synapse = synapse

    # network.config[nengo.Connection].synapse = nengo.synapses.Lowpass(0.0)
    # network.config[nengo.Connection].synapse = nengo.synapses.Lowpass(0.001)
    # network.config[nengo.Connection].synapse = nengo.synapses.Lowpass(0.005)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.001)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.002)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.003)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.004)
    # network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.005)

    outputs = build_target_layer(
        'logprob', layers, data, network, hists=hists, pt=presentation_time)

    # test whole network
    with network:
        # xp = nengo.Probe(outputs['data'], synapse=None)
        yp = nengo.Probe(outputs['probs'], synapse=None)
        # zp = nengo.Probe(outputs['logprob'], synapse=None)

        if count_spikes:
            spikes_p = OrderedDict(
                (name, nengo.Probe(outputs[name]))
                for name in layers if layers[name]['type'] == 'neuron')

    if ocl_profile:
        # profile
        import nengo_ocl
        sim = nengo_ocl.Simulator(network, profiling=True)
        sim.run(presentation_time)
        sim.print_profiling(sort=1)
    else:
        n = len(data[0])  # test on all examples
        if n_max is not None:
            n = min(n, n_max)

        print("Running %d examples for %0.3f s each" % (n, presentation_time))
        sim = Simulator(network)
        sim.run(n * presentation_time)

    dt = sim.dt
    t = sim.trange()
    y = sim.data[yp]
    # z = sim.data[zp]

    get_ind = lambda t: int(t / presentation_time)
    inds = slice(0, get_ind(t[-2]) + 1)
    images = data[0][inds]
    labels = data[1][inds]
    data_mean = dp.data_mean
    label_names = dp.batch_meta['label_names']

    spikes = None
    if count_spikes:
        trange = float(t[-1] - t[0])
        spikes = OrderedDict((k, sim.data[v]) for k, v in spikes_p.items())
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
                 dt=dt, pt=presentation_time, t=t, y=y, spikes=spikes)
        print("Saved '%s'" % savefile)

    errors, top5errors, _, _ = error(dt, presentation_time, labels, t, y)
    print("Error: %0.4f, %0.4f (%d samples)"
          % (errors.mean(), top5errors.mean(), errors.size))


def error_lims(dt, pt, labels, t, y, method='mean', ct_start=0., ct_end=1.):
    assert ct_start < ct_end
    cn0 = int(np.floor(ct_start * pt / dt))
    cn1 = int(np.ceil(ct_end * pt / dt))
    pn = int(pt / dt)
    n = y.shape[0] / pn
    assert cn0 >= 0
    assert cn1 <= pn
    assert cn0 < cn1

    # blocks to be used for classification
    blocks = y.reshape(n, pn, y.shape[1])[:, cn0:cn1, :]

    if method == 'mean':
        probs = blocks.mean(1)
    elif method == 'peak':
        probs = blocks.max(1)
    else:
        raise ValueError("Unrecognized method %r" % method)

    labels = labels[:n]
    assert probs.shape[0] == labels.shape[0]
    inds = np.argsort(probs, axis=1)
    top1errors = inds[:, -1] != labels
    top5errors = np.all(inds[:, -5:] != labels[:, None], axis=1)

    z_labels = labels[(t / pt).astype(int) % len(labels)]
    z = np.argmax(y, axis=1) != z_labels

    return top1errors, top5errors, z


def error(dt, pt, labels, t, y, method='mean'):
    # filter outputs (better accuracy)
    # s = nengo.synapses.Alpha(0.002)  # 30ms_pt-0ms_alpha
    # s = nengo.synapses.Alpha(0.004)
    # s = nengo.synapses.Alpha(0.005)
    # s = nengo.synapses.Alpha(0.01)
    # s = nengo.synapses.Alpha(0.02)
    # y = nengo.synapses.filt(y, s, dt)
    # y = nengo.synapses.filtfilt(y, s, dt)

    # ct = 0.01  # classification time
    # ct = 0.015  # classification time
    # ct = 0.02  # classification time
    # ct = 0.03  # classification time
    # ct = 0.04  # classification time
    # ct = 0.05  # classification time
    # ct = 0.06  # classification time
    # ct = 0.07  # classification time
    # ct = 0.075  # classification time
    # ct = 0.08  # classification time
    ct = 0.09  # classification time
    # ct = 0.10  # classification time
    # ct = 0.12  # classification time
    # ct = 0.15  # classification time

    top1errors, top5errors, z = error_lims(
        dt, pt, labels, t, y, method=method, ct_start=((pt - ct) / pt))
        # dt, pt, labels, t, y, method=method, ct_start=(ct / pt))
        # dt, pt, labels, t, y, method=method, ct_start=(ct / pt), ct_end=(0.06 / pt))
    return top1errors, top5errors, y, z


def view(dt, pt, images, labels, data_mean, label_names, t, y, z, spikes=None, n_max=30):
    import nengo.utils.matplotlib

    get_ind = lambda t: int(t / pt)

    # clip if more than n_max
    i_max = int(n_max * pt / dt)
    t = t[:i_max]
    y = y[:i_max]
    z = z[:i_max]

    # compute transition points
    transitions = np.diff((t / pt).astype(int))
    transitions = np.concatenate(([0], transitions)).astype(bool)
    transitions[-1] = 0
    t_trans = np.tile(t[transitions], (2, 1))
    def plot_transitions():
        y_trans = np.tile(np.array(plt.ylim())[:, None], (1, t_trans.shape[1]))
        plt.plot(t_trans, y_trans, 'k--')
        # print(t_trans)
        # print(y_trans)

    spikes = [] if spikes is None else spikes
    n_spikes = len(spikes)

    rows, cols = 3 + n_spikes, 1
    plt.figure()

    # plot images
    c, m, n = images.shape[1:]
    inds = slice(0, get_ind(t[-2]) + 1)

    imgs = images[inds]
    allimage = np.zeros((c, m, n * len(imgs)))
    for i, img in enumerate(imgs):
        img = (img + data_mean.reshape(1, c, m, n)) / 255.
        allimage[:, :, i * n:(i + 1) * n] = img.clip(0, 1)

    allimage = np.transpose(allimage, (1, 2, 0))

    plt.subplot(rows, cols, 1)
    plt.imshow(allimage, vmin=0, vmax=1, aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('input')

    # plot spikes
    for i, key in enumerate(sorted(spikes)):
        max_neurons = 20
        s = spikes[key]
        if s.shape[1] > max_neurons:
            s = s[:, np.random.permutation(s.shape[1])[:max_neurons]]

        plt.subplot(rows, cols, i+2)
        nengo.utils.matplotlib.rasterplot(t, s)
        plot_transitions()
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(key.rstrip('_neurons'))

    # plot classifier output
    plt.subplot(rows, cols, rows-1)
    plt.plot(t, y)
    plt.xlim([t[0], t[-1]])
    # if len(label_names) <= 10:
    #     plt.legend(label_names, fontsize=8, loc=2)
    plot_transitions()
    plt.xticks([])
    plt.yticks([0])
    plt.ylabel('output')

    # plot error
    plt.subplot(rows, cols, rows)
    plt.plot(t, z)
    plot_transitions()
    plt.xlim([t[0], t[-1]])
    plt.ylim([-0.1, 1.1])
    plt.yticks([0, 1])
    plt.xlabel('time [s]')
    plt.ylabel('error')

    # plt.tight_layout()

    # plt.show()


def classification_start_time_plot(dt, pt, labels, t, y):

    # s = nengo.synapses.Alpha(0.002)  # 30ms_pt-0ms_alpha
    # s = nengo.synapses.Alpha(0.004)
    # s = nengo.synapses.Alpha(0.005)
    # s = nengo.synapses.Alpha(0.01)
    # s = nengo.synapses.Alpha(0.02)
    # y = nengo.synapses.filt(y, s, dt)
    # y = nengo.synapses.filtfilt(y, s, dt)

    # dstart = 0.005
    dstart = 0.01
    dend = 0.001
    # ct_starts = dstart * np.arange(int(pt / dstart))
    # ct_starts = dstart * np.arange(int(0.8 * pt / dstart))
    ct_starts = dstart * np.arange(int(pt / dstart))[:7]

    results = []
    for ct_start in ct_starts:
        ct_ends = dend * np.arange(ct_start / dend + 1, pt / dend + 1)
        errors = []
        for ct_end in ct_ends:
            e, _, _ = error_lims(
                dt, pt, labels, t, y, ct_start=ct_start/pt, ct_end=ct_end/pt)
            errors.append(e.mean())
        results.append((ct_start, ct_ends, errors))

    plt.figure()
    for ct_start, ct_ends, errors in results:
        plt.semilogy(ct_ends, errors, label='%0.3f' % ct_start)

    error_min = min(*[np.min(errors) for _, _, errors in results])
    plt.xlim([0., pt])
    plt.ylim([0.8*error_min, 1.0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run network in Nengo")
    parser.add_argument('loadfile', help="Checkpoint to load")
    parser.add_argument('savefile', nargs='?', default=None, help="Where to save output")
    parser.add_argument('--histload', help="Layer histograms created by run_numpy")
    parser.add_argument('--spikes', action='store_true', help="Count spikes")
    parser.add_argument('--ocl', action='store_true', help="Run using Nengo OCL")
    parser.add_argument('--ocl-profile', action='store_true', help="Profile Nengo OCL")
    parser.add_argument('--n', default=None, type=int, help="Number of examples to run")
    args = parser.parse_args()

    backend = 'nengo_ocl' if args.ocl else 'nengo'
    run(args.loadfile, savefile=args.savefile, histload=args.histload,
        count_spikes=args.spikes, n_max=args.n, backend=backend,
        ocl_profile=args.ocl_profile)
