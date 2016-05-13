import collections

import numpy as np
import nengo
from nengo_extras import SoftLIFRate

from convdata import DataProvider, CIFARDataProvider
from python_util.gpumodel import IGPUModel


def get_depths(layers):
    depths = {}

    def get_depth(key):
        if key not in depths:
            inputs = [get_depth(k) for k in layers[key].get('inputs', [])]
            depths[key] = max(inputs) + 1 if len(inputs) > 0 else 0
        return depths[key]

    for key in layers:
        get_depth(key)

    return depths


def load_network(loadfile, multiview=None, sort_layers=False):
    load_dic = IGPUModel.load_checkpoint(loadfile)
    layers = load_dic['model_state']['layers']
    op = load_dic['op']

    if sort_layers:
        depths = get_depths(layers)
        layers = collections.OrderedDict(
            sorted(layers.items(), key=lambda item: depths[item[0]]))

    options = {}
    for o in load_dic['op'].get_options_list():
        options[o.name] = o.value

    dp_params = {}
    for v in ('color_noise', 'multiview_test', 'inner_size', 'scalar_mean', 'minibatch_size'):
        dp_params[v] = options[v]

    lib_name = "cudaconvnet._ConvNet"
    print("Importing %s C++ module" % lib_name)
    libmodel = __import__(lib_name,fromlist=['_ConvNet'])
    dp_params['libmodel'] = libmodel

    if multiview is not None:
        dp_params['multiview_test'] = multiview

    dp = DataProvider.get_instance(
        options['data_path'],
        batch_range=options['test_batch_range'],
        type=options['dp_type'],
        dp_params=dp_params, test=True)

    epoch, batchnum, data = dp.get_next_batch()
    images, labels = data[:2]
    images = images.T
    images.shape = (images.shape[0], dp.num_colors, dp.inner_size, dp.inner_size)
    labels.shape = (-1,)
    labels = labels.astype('int')
    assert images.shape[0] == labels.shape[0]

    if 1:
        rng = np.random.RandomState(8)
        i = rng.permutation(images.shape[0])
        images = images[i]
        labels = labels[i]

    data = [images, labels] + list(data[2:])
    # data['data_mean'] = dp.data_mean
    # data['label_names'] = dp.batch_meta['label_names']

    return layers, data, dp


def round_array(x, n_values, x_min, x_max):
    if x_min == x_max:
        return

    assert x_min < x_max
    np.clip(x, x_min, x_max, out=x)
    scale = float(n_values - 1) / (x_max - x_min)
    x[:] = np.round(x * scale) / scale


def round_layer(layer, n_values, clip_percent=0):
    if 'weights' in layer:
        for weights in layer['weights']:
            w_min = np.percentile(weights.ravel(), clip_percent)
            w_max = np.percentile(weights.ravel(), 100 - clip_percent)
            round_array(weights, n_values, w_min, w_max)

    if 'biases' in layer:
        for biases in layer['biases']:
            b_min = biases.min()
            b_max = biases.max()
            round_array(biases, n_values, b_min, b_max)
