import numpy as np

from convdata import DataProvider, CIFARDataProvider
from python_util.gpumodel import IGPUModel


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
