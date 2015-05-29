import datetime
import os
import sys
import time
import multiprocessing

import hyperopt
from hyperopt import hp
import numpy as np

from layer import LayerParser
# LayerParser.verbose = False
from convnet import ConvNet


def write_config_files(args, timestamp):
    args = dict(args)  # copy, since we may modify it

    layer_file_name = 'cifar10_%s_layers.cfg' % timestamp
    param_file_name = 'cifar10_%s_params.cfg' % timestamp

    dirname = './hyperopt_output'
    layer_file_path = os.path.join(dirname, layer_file_name)
    param_file_path = os.path.join(dirname, param_file_name)

    # format args
    neuron = ("softlif[%(amp)s,%(tau_ref)s,%(tau_rc)s,%(alpha)s,"
              "%(sigma)s,%(noise)s]" % args)

    layer_args = dict(args)
    layer_args.update(dict(neuron=neuron))
    layer_text = """
[data]
type=data
dataIdx=0

[labels]
type=data
dataIdx=1

[conv1]
type=conv
inputs=data
channels=3
filters=64
padding=2
stride=1
filterSize=5
neuron=%(neuron)s
initW=%(initW1)s
sumWidth=4
sharedBiases=1
gpu=0

[pool1]
type=pool
pool=avg
inputs=conv1
start=0
sizeX=3
stride=2
outputsX=0
channels=64

[conv2]
type=conv
inputs=pool1
filters=64
padding=2
stride=1
filterSize=5
channels=64
neuron=%(neuron)s
initW=%(initW2)s
sumWidth=2
sharedBiases=1

[pool2]
type=pool
pool=avg
inputs=conv2
start=0
sizeX=3
stride=2
outputsX=0
channels=64

[local3]
type=local
inputs=pool2
filters=64
padding=1
stride=1
filterSize=3
channels=64
neuron=%(neuron)s
initW=%(initW3)s

[local4]
type=local
inputs=local3
filters=32
padding=1
stride=1
filterSize=3
channels=64
neuron=%(neuron)s
initW=%(initW4)s

[fc10]
type=fc
outputs=10
inputs=local4
initW=%(initW5)s

[probs]
type=softmax
inputs=fc10

[logprob]
type=cost.logreg
inputs=labels,probs
gpu=0
    """ % layer_args
    with open(layer_file_path, 'w') as f:
        f.write(layer_text)

    epsW = "%(epsW_schedule)s[base=%(epsW_base)s;tgtFactor=%(epsW_tgtFactor)d]" % args
    epsB = "%(epsB_schedule)s[base=%(epsB_base)s;tgtFactor=%(epsB_tgtFactor)d]" % args
    args.update(dict(epsW=epsW, epsB=epsB))

    param_text = """
[conv1]
epsW=%(epsW)s
epsB=%(epsB)s
momW=%(momW)s
momB=%(momB)s
wc=%(wc1)s

[conv2]
epsW=%(epsW)s
epsB=%(epsB)s
momW=%(momW)s
momB=%(momB)s
wc=%(wc2)s

[local3]
epsW=%(epsW)s
epsB=%(epsB)s
momW=%(momW)s
momB=%(momB)s
wc=%(wc3)s

[local4]
epsW=%(epsW)s
epsB=%(epsB)s
momW=%(momW)s
momB=%(momB)s
wc=%(wc4)s

[fc10]
epsW=%(epsW)s
epsB=%(epsB)s
momW=%(momW)s
momB=%(momB)s
wc=%(wc5)s

[logprob]
coeff=1
    """ % args
    with open(param_file_path, 'w') as f:
        f.write(param_text)

    return layer_file_path, param_file_path


def check_costs(self, cost_output):
    costs, num_cases = cost_output
    for errname in costs:
        cost = costs[errname][0]
        if np.isnan(cost) or np.isinf(cost) or (cost / num_cases) > 1e3:
            return True
    return False


n_epochs = 200


def objective(layer_file_name, param_file_name, save_file_name):
    def logprob_errors(error_output):
        error_types, n = error_output
        logprob = error_types['logprob'][0] / n
        classifier = error_types['logprob'][1] / n
        logprob = np.inf if np.isnan(logprob) else logprob
        classifier = np.inf if np.isnan(classifier) else classifier
        return logprob, classifier

    real_stdout = sys.stdout
    sys.stdout = open(save_file_name + '.log', 'w')
    convnet = None
    try:
        # set up options
        op = ConvNet.get_options_parser()

        for option in op.get_options_list():
            option.set_default()

        op.set_value('data_path', os.path.expanduser('~/data/cifar-10-py-colmajor/'))
        op.set_value('dp_type', 'cifar')
        op.set_value('inner_size', '24')

        op.set_value('gpu', '0')
        op.set_value('testing_freq', '25')

        op.set_value('train_batch_range', '1-5')
        op.set_value('test_batch_range', '6')
        op.set_value('num_epochs', n_epochs, parse=False)

        op.set_value('layer_def', layer_file_name)
        op.set_value('layer_params', param_file_name)
        op.set_value('save_file_override', save_file_name)

        convnet = ConvNet(op, None)

        # train for three epochs and make sure error is okay
        convnet.num_epochs = 3
        convnet.train()

        logprob, error = logprob_errors(convnet.train_outputs[-1])
        if not (error > 0 and error < 0.85):
            # should get at most 85% error after three epochs
            print "\naborted (%s, %s)" % (logprob, error)
            return logprob, error

        # train for full epochs
        convnet.num_epochs = n_epochs
        convnet.train()

        logprob, error = logprob_errors(convnet.get_test_error())
        print "\nfinished (%s, %s)" % (logprob, error)

        return logprob, error
    except RuntimeError:
        print "\nerrored at epoch %d" % (convnet.epoch)
        return np.inf, 1.0
    finally:
        if convnet is not None:
            convnet.destroy_model_lib()

        print "\n"  # end any pending lines to ensure flush
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = real_stdout


def objective_wrapper(args):
    max_time = 7 * n_epochs + 10

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    save_file_name = os.path.join('./hyperopt_output',
                                  'cifar10_%s' % timestamp)

    # write config files
    layer_file_name, param_file_name = write_config_files(args, timestamp)

    def wrapper(q, *args):
        q.put(objective(*args))

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(
        q, layer_file_name, param_file_name, save_file_name))
    p.start()
    t0 = time.time()
    while p.is_alive():
        if (time.time() - t0) > max_time:
            print "KILLING"
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
            break

    if not q.empty():
        if q.qsize() > 1:
            print "WARNING: multiple values returned"
        logprob, error = q.get()
        print "Trained %s: %s, %s" % (save_file_name, logprob, error)
        return error
    else:
        print "WARNING: no return value"
        return np.inf


space = {
    # neuron params
    'amp': 0.063,
    # 'tau_ref': 0.001,
    # 'tau_rc': 0.05,
    # 'alpha': 0.825,
    # 'tau_ref': hp.uniform('tau_ref', 0.001, 0.005),
    'tau_ref': 0.002,
    'tau_rc': hp.uniform('tau_rc', 0.01, 0.06),
    # 'alpha': hp.uniform('alpha', 0.1, 10.0),
    'alpha': 1.0,
    'sigma': 0.02,
    'noise': 10.,

    # learning rate params
    'epsW_schedule': hp.choice('epsW_schedule', ['linear', 'exp']),
    'epsW_base': hp.lognormal('epsW_base', np.log(1e-3), np.log(1e1)),
    'epsW_tgtFactor': hp.qloguniform('epsW_tgtFactor', np.log(1), np.log(1e4), 1),
    'epsB_schedule': hp.choice('epsB_schedule', ['linear', 'exp']),
    'epsB_base': hp.lognormal('epsB_base', np.log(1e-3), np.log(1e1)),
    'epsB_tgtFactor': hp.qloguniform('epsB_tgtFactor', np.log(1), np.log(1e4), 1),
    'momW': hp.uniform('momW', 0.001, 0.999),
    'momB': hp.uniform('momB', 0.001, 0.999),

    # initial weight params
    'initW1': hp.lognormal('initW1', np.log(1e-4), np.log(1e2)),
    'initW2': hp.lognormal('initW2', np.log(1e-2), np.log(1e1)),
    'initW3': hp.lognormal('initW3', np.log(1e-2), np.log(1e1)),
    'initW4': hp.lognormal('initW4', np.log(1e-2), np.log(1e1)),
    'initW5': hp.lognormal('initW5', np.log(1e-2), np.log(1e1)),

    # weight costs
    'wc1': hp.lognormal('wc1', np.log(1e-8), np.log(1e3)),
    'wc2': hp.lognormal('wc2', np.log(1e-8), np.log(1e3)),
    'wc3': hp.lognormal('wc3', np.log(1e-3), np.log(3e1)),
    'wc4': hp.lognormal('wc4', np.log(1e-3), np.log(3e1)),
    'wc5': hp.lognormal('wc5', np.log(1e-2), np.log(1e1)),
}

best = hyperopt.fmin(objective_wrapper, space, algo=hyperopt.tpe.suggest, max_evals=400)
print best
