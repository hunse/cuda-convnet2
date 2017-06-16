import argparse

import matplotlib.pyplot as plt
import numpy as np

import layer as lay


class DummyOpParser(object):
    def get_value(self, key):
        if key == 'gpu':
            return [0]


class DummyTrainDataProvider(object):
    def __init__(self, dims):
        self.dims = dims

    def get_data_dims(self, idx=0):
        return self.dims if idx == 0 else 1

    def get_num_classes(self):
        return self.dims


class Model(object):
    def __init__(self, layer_def, layer_params):
        dims = 100
        self.dims = dims

        self.libmodel = None

        self.op = DummyOpParser()

        # init model state
        self.train_data_provider = DummyTrainDataProvider(dims)
        self.layers = lay.LayerParser.parse_layers(layer_def, layer_params, self, layers={})

        self.device_ids = [0]
        self.minibatch_size = 256
        self.conserve_mem = True

    def start_batch(self):
        target = 'neuron1'
        n = 200
        # n = 1
        # n = 3

        # -- initialize just like this to avoid random bugs when copying data
        X = 0 * np.ones((self.dims, n), dtype=np.single)
        Y = 0 * np.ones((1, n), dtype=np.single)

        # x = np.linspace(-3, 10, self.dims)
        x = np.linspace(-5, 20, self.dims)
        X[:] = x[:, None]

        all_data = [X, Y]
        odims = self.layers[target]['outputs']
        self.x = x
        self.features = np.zeros((n, odims), dtype=np.single)

        self.libmodel.startFeatureWriter(all_data, [self.features], [target])

    def finish_batch(self):
        return self.libmodel.finishBatch()

    def run(self):
        self.start_batch()
        self.finish_batch()

        # --- plot
        plt.figure()

        x = self.x
        y = self.features
        y_mean = y.mean(axis=0)
        # print(y_mean)
        y_50 = np.percentile(y, 50, axis=0)
        y_25_75 = np.array([y_50 - np.percentile(y, 25, axis=0),
                            np.percentile(y, 75, axis=0) - y_50])
        y_min_max = np.array([y_mean - y.min(axis=0), y.max(axis=0) - y_mean])
        plt.plot(x, y_mean, 'k-')
        plt.plot(x, y_50, 'kx')
        eb = plt.errorbar(x, y_mean, y_min_max, fmt=None, ecolor='k')
        eb[-1][0].set_linestyle(':')
        plt.errorbar(x, y_50, y_25_75, fmt=None, ecolor='k')

        # plt.savefig('check_nonlinearity.pdf')
        plt.show()

    def __enter__(self):
        lib_name = "cudaconvnet._ConvNet"
        print "========================="
        print "Importing %s C++ module" % lib_name
        self.libmodel = __import__(lib_name, fromlist=['_ConvNet'])

        self.libmodel.initModel(self.layers,
                                self.device_ids,
                                self.minibatch_size,
                                self.conserve_mem)


    def __exit__(self, e_type, e_val, e_tb):
        self.libmodel.destroyModel()


if __name__ == '__main__':

    # layer_def = 'layers/check-softlif.cfg'
    # layer_def = 'layers/check-softlifalpha.cfg'
    layer_def = 'layers/check-softlifalpharc.cfg'
    layer_params = 'layers/layer-params-cifar10-11pct.cfg'


    model = Model(layer_def, layer_params)
    with model:
        model.run()
