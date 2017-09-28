import datetime
import multiprocessing
import os
import sys
import time
import traceback

import numpy as np

from convnet import ConvNet
import run_numpy
import run_nengo


save_dir = os.path.join('checkpoints', 'auto-cifar10')
assert os.path.exists(save_dir)
real_stdout = sys.stdout
real_stderr = sys.stderr


class Logger(object):
    def __init__(self, logpath, terminal=real_stdout):
        self.terminal = terminal
        self.log = open(logpath, 'a')

    def close(self):
        # self.terminal.close()
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def reset_std(kind, default=None):
    def flushclose(stream):
        stream.write("\n")  # end any pending lines to ensure flush
        stream.flush()
        stream.close()

    kind = kind.lower()
    if 'out' in kind:
        flushclose(sys.stdout)
        sys.stdout = real_stdout if default is None else default
    if 'err' in kind:
        flushclose(sys.stderr)
        sys.stderr = real_stderr if default is None else default


class NetworkType(object):
    def __init__(self, name, layer_file, layer_params_file=None):
        self.name = name
        self.layer_file = layer_file
        self.layer_params_file = layer_params_file
        self.save_prefix = name

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.name)

    def filter_checkpoints(self):
        checkpoint_files = os.listdir(save_dir)
        files = [s for s in checkpoint_files
                 if s.split('_')[0] == self.save_prefix
                 and os.path.isdir(os.path.join(save_dir, s))]
        networks = [Network(self, s.split('_')[1]) for s in files]
        return networks

    def new_network(self, **kwargs):
        return Network(self, **kwargs)


class Network(object):
    def __init__(self, network_type, timestamp=None, seed=None):
        self.network_type = network_type
        self.timestamp = (
            timestamp or datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        self.seed = np.random.randint(2**30) if seed is None else seed

        self.numpy = None
        self.nengo = {}

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.network_type.name,
                               self.timestamp)

    def checkpoint_name(self):
        return '%s_%s' % (self.network_type.name, self.timestamp)

    def numpy_name(self):
        return '%s_numpy.npz' % self.checkpoint_name()

    def nengo_name(self, pres_time, synapse_type, synapse_tau):
        return '%s_%dms-pt_%dms-%s.npz' % (
            self.checkpoint_name(),
            1000*pres_time, 1000*synapse_tau, synapse_type)

    def get_op(self, n_epochs=None, params_file=None,
               train_range='1-5', test_range='6'):
        op = ConvNet.get_options_parser()
        load_dic = None

        for option in op.get_options_list():
            option.set_default()

        op.set_value('data_path',
                     os.path.expanduser('~/data/cifar-10-py-colmajor/'))
        op.set_value('dp_type', 'cifar')
        op.set_value('inner_size', '24')

        op.set_value('gpu', '0')
        op.set_value('testing_freq', '25')

        op.set_value('layer_path', 'layers/')
        op.set_value('layer_def', self.network_type.layer_file)
        op.set_value('layer_params',
                     params_file or self.network_type.layer_params_file)

        op.set_value('train_batch_range', train_range)
        op.set_value('test_batch_range', test_range)
        if n_epochs is not None:
            op.set_value('num_epochs', n_epochs, parse=False)

        checkpoint_path = os.path.join(save_dir, self.checkpoint_name())
        if os.path.exists(checkpoint_path):
            op.set_value('load_file', checkpoint_path)
            load_dic = ConvNet.load_checkpoint(checkpoint_path)
            old_op = load_dic['op']
            old_options = dict(old_op.options)
            old_op.merge_from(op)
            op.options, old_op.options = old_op.options, old_options
        else:
            op.set_value('save_file_override', checkpoint_path)

        return op, load_dic

    def load_numpy(self):
        numpy_path = os.path.join(save_dir, self.numpy_name())
        if not os.path.exists(numpy_path):
            test_numpy(self)

        if os.path.exists(numpy_path):
            data = np.load(numpy_path)
            self.numpy = tuple(data[k] for k in ('logprob', 'top1', 'top5'))

    def load_nengo(self, pres_time, synapse_type, synapse_tau, ct_start):
        key = (pres_time, synapse_type, synapse_tau, ct_start)
        fkey = key[:3]
        nengo_path = os.path.join(save_dir, self.nengo_name(*fkey))
        if not os.path.exists(nengo_path):
            test_nengo(self, *fkey)

        if os.path.exists(nengo_path):
            objs = np.load(nengo_path)
            kwargs = {k: objs[k] for k in ('dt', 'pt', 'labels', 't', 'y')}
            top1errors, top5errors, _ = run_nengo.error_lims(
                ct_start=ct_start, **kwargs)
            self.nengo[key] = (top1errors.mean(), top5errors.mean())


def run_process(function, args=(), kwargs={}, max_time=None):
    def wrapper(q, *args, **kwargs):
        q.put(function(*args, **kwargs))

    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=wrapper, args=(q,) + args, kwargs=kwargs)

    p.start()
    t0 = time.time()
    while p.is_alive():
        if max_time and (time.time() - t0) > max_time:
            print("KILLING")
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
            break

    if not q.empty():
        return q.get()
    else:
        return None


def train_proc(network, **kwargs):
    log_path = os.path.join(save_dir, network.checkpoint_name() + '.log')
    real_stdout = sys.stdout
    sys.stdout = open(log_path, 'a')
    convnet = None
    try:
        np.random.seed(network.seed)
        op, load_dic = network.get_op(**kwargs)
        convnet = ConvNet(op, load_dic)
        convnet.train()
        return True
    except RuntimeError:
        print(traceback.format_exc())
        if convnet:
            print("\nerrored at epoch %d" % (convnet.epoch))
    except:
        print(traceback.format_exc())
    finally:
        if convnet:
            convnet.destroy_model_lib()

        reset_std('out', real_stdout)


def test_numpy_proc(network):
    save_path = os.path.join(save_dir, network.numpy_name())
    log_path = save_path + '.log'
    real_stdout = sys.stdout
    sys.stdout = open(log_path, 'a')
    try:
        checkpoint_path = os.path.join(save_dir, network.checkpoint_name())
        layers, data, dp = run_numpy.load_network(checkpoint_path)
        outputs = run_numpy.compute_target_layer('logprob', layers, data)
        logprob, top1, top5 = outputs['logprob']
        np.savez(save_path, logprob=logprob, top1=top1, top5=top5)
    except:
        print(traceback.format_exc())
        print("Skipping run_numpy(%s)" % network)
    finally:
        reset_std('out', real_stdout)


def test_nengo_proc(network, pres_time, synapse_type, synapse_tau):
    checkpoint_path = os.path.join(save_dir, network.checkpoint_name())
    save_path = os.path.join(save_dir, network.nengo_name(
        pres_time, synapse_type, synapse_tau))
    log_path = save_path + '.log'
    real_stdout = sys.stdout
    sys.stdout = open(log_path, 'a')
    try:
        run_nengo.run(checkpoint_path, savefile=save_path, backend='nengo_ocl',
                      presentation_time=pres_time,
                      synapse_type=synapse_type, synapse_tau=synapse_tau)
    except:
        print(traceback.format_exc())
        print("Skipping run_nengo(%s, %s, %s, %s)" % (
            network, pres_time, synapse_type, synapse_tau))
    finally:
        reset_std('out', real_stdout)


def train(network):
    blocks = (
        (350, 'layer-params-cifar10-11pct.cfg', '1-4', '6'),
        (500, 'layer-params-cifar10-11pct.cfg', '1-5', '6'),
        (510, 'layer-params-cifar10-11pct-eps10.cfg', '1-5', '6'),
        (520, 'layer-params-cifar10-11pct-eps100.cfg', '1-5', '6'),
    )

    for n_epochs, params_file, train_range, test_range in blocks:
        s = run_process(train_proc, args=(network,), kwargs=dict(
            n_epochs=n_epochs, params_file=params_file,
            train_range=train_range, test_range=test_range),
                        max_time=5*n_epochs + 60)
        if s is None:
            # remove checkpoint file
            path = os.path.join(save_dir, network.checkpoint_name())
            if os.path.exists(path):
                print("Error during training, removing %r" % path)
                os.remove(path)

            break


def test_numpy(network):
    run_process(test_numpy_proc, args=(network,), max_time=200)


def test_nengo(network, pres_time, synapse_type, synapse_tau):
    run_process(test_nengo_proc,
                args=(network, pres_time, synapse_type, synapse_tau),
                max_time=200)


logpath = os.path.join(save_dir, "auto_cifar10_%s.log" % (
    datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))
sys.stdout = Logger(logpath)
try:
    n_trials = 5

    network_types = [
        NetworkType('lif', 'layers-cifar10-lif.cfg'),
        NetworkType('lifalpha', 'layers-cifar10-lifalpha-3ms.cfg'),
        NetworkType('lifalpha5ms', 'layers-cifar10-lifalpha-5ms.cfg'),
        NetworkType('lifalpharc', 'layers-cifar10-lifalpharc.cfg'),
        NetworkType('lifalpharc5ms', 'layers-cifar10-lifalpharc-5ms.cfg'),
        NetworkType('lifnoise10', 'layers-cifar10-lif-noise10.cfg'),
        NetworkType('lifnoise20', 'layers-cifar10-lif-noise20.cfg'),
    ]

    nengo_types = [
        (0.15, 'alpha', 0.000, 6./15),
        (0.15, 'alpha', 0.000, 1./15),
        (0.15, 'alpha', 0.001, 6./15),
        (0.15, 'alpha', 0.003, 6./15),
        (0.20, 'alpha', 0.005, 10./20),
    ]

    for network_type in network_types:
        networks = network_type.filter_checkpoints()
        for _ in range(len(networks), n_trials):
            train(network_type.new_network())

        networks = network_type.filter_checkpoints()
        for network in networks:
            network.load_numpy()
            # print('  %s: %s' % (network, network.numpy))

        for network in networks:
            for pt, synapse_type, synapse_tau, ct_start in nengo_types:
                network.load_nengo(pt, synapse_type, synapse_tau, ct_start)
            # for vals in nengo_types:
            #     print('  %s: %s' % (network, network.nengo[vals]))

        # --- print result summaries
        top1mean = 100*np.mean([n.numpy[1] for n in networks])
        # top5mean = 100*np.mean([n.numpy[2] for n in networks])
        top1min = 100*np.min([n.numpy[1] for n in networks])
        top1mini = np.argmin([n.numpy[1] for n in networks])
        print('%s: %0.2f (min %0.2f [%d])' % (
            network_type, top1mean, top1min, top1mini))
        for key in nengo_types:
            top1mean = 100*np.mean([n.nengo[key][0] for n in networks])
            # top5mean = 100*np.mean([n.nengo[key][1] for n in networks])
            top1min = 100*np.min([n.nengo[key][0] for n in networks])
            top1mini = np.argmin([n.nengo[key][0] for n in networks])
            strf = (network_type,) + key + (top1mean, top1min, top1mini)
            print('%s %0.3f %s(%0.3f) %0.3f: %0.2f (min %0.2f [%d])' % strf)
except:
    print(traceback.format_exc())
finally:
    reset_std('out')
