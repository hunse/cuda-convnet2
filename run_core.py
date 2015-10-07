import numpy as np
import nengo

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
    images.shape = (images.shape[0], dp.num_colors, dp.inner_size, dp.inner_size)
    labels.shape = (-1,)
    labels = labels.astype('int')
    assert images.shape[0] == labels.shape[0]

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
        j = softrelu(J - 1, sigma=self.sigma)
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = 1. / (
            self.tau_ref + self.tau_rc * np.log1p(1. / j[j > 0]))


def test_softlifrate():
    import matplotlib.pyplot as plt

    neurons = SoftLIFRate(sigma=0.002, tau_rc=0.02, tau_ref=0.002)
    x = np.linspace(-1, 1, 101)
    r = neurons.rates(x, 1., 1.)
    plt.plot(x, r)
    plt.show()


if __name__ == '__main__':
    test_softlifrate()
