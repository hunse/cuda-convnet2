"""
See how translation-invariant features in a network are
"""
import numpy as np

from run_core import load_network
from run_numpy import compute_target_layer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run network in Numpy")
    parser.add_argument('loadfile', help="Checkpoint to load")
    parser.add_argument('--n', type=int, help="Number of images to test")
    args = parser.parse_args()

    layers, data, dp = load_network(args.loadfile)

    # --- get data
    bidx = 0
    assert dp.batches_generated > bidx

    n_images = 1000
    r = 2
    nc, nx, ny = dp.num_colors, dp.img_size, dp.inner_size
    assert (nx - ny) % 2 == 0
    p = (nx - ny) / 2
    assert r <= p

    raw = dp.data_dic[bidx]['data'].astype('float32').T.reshape(-1, nc, nx, nx)
    labels = dp.data_dic[bidx]['labels'].astype('int32')
    raw, labels = raw[:n_images], labels[:n_images]

    irange, jrange = np.arange(-r, r+1), np.arange(-r, r+1)
    ni, nj = len(irange), len(jrange)
    nij = ni * nj

    images = np.zeros((raw.shape[0], ni, nj, nc, ny, ny), dtype=raw.dtype)
    for ii, i in enumerate(irange):
        for jj, j in enumerate(jrange):
            images[:, ii, jj, :, :, :] = raw[:, :, p+i:p+i+ny, p+j:p+j+ny]
    del raw

    images = images.reshape(-1, nc, ny, ny)
    images -= dp.data_mean.T.reshape(1, nc, ny, ny)
    labels = labels.repeat(nij)

    # import matplotlib.pyplot as plt
    # from hunse_tools.plotting import tile
    # tile(np.transpose(images, (0, 2, 3, 1)) / 255., rows=10, cols=nij)
    # plt.show()

    data = [images, labels]

    # print(data[0].shape, data[1].shape)

    # --- compute outputs
    outputs = compute_target_layer('logprob', layers, data)

    display_layers = [name for name, layer in layers.items()
                      if layer['type'] in ('fc', 'conv', 'local', 'pool', 'neuron')]

    for name in sorted(display_layers):
        output = outputs[name]
        output = output.reshape(n_images, ni, nj, -1)

        # subtract out centre
        output = output - output[:, r:r+1, r:r+1, :]

        # vertical invariance for each feature
        inv_v = output.std(axis=1)
        # inv_v[(output == 0).all(axis=1)] = np.nan  # some component must be on
        inv_v[(output == 0).any(axis=1)] = np.nan  # all components must be on

        # horizontal invariance for each feature
        inv_h = output.std(axis=2)
        # inv_h[(output == 0).all(axis=2)] = np.nan  # some component must be on
        inv_h[(output == 0).any(axis=2)] = np.nan  # all components must be on

        # inv = np.sqrt(inv_v**2 + inv_h**2)

        inv_v = np.nanmean(inv_v, axis=(0, 1))
        inv_h = np.nanmean(inv_h, axis=(0, 1))

        print("%s: %0.3f (%0.3f), %0.3f (%0.3f)" % (
            name, inv_v.mean(), inv_v.std(), inv_h.mean(), inv_h.std()))
