
import numpy as np

from run_numpy import compute_target_layer

import mnist

import nengo
import nengo_ocl
from nengo_extras.cuda_convnet import load_model_pickle, CudaConvnetNetwork

# --- data

# train, _, test = mnist.load(spaun=False)
train, _, test = mnist.load(spaun=True)
train_image_mean = train[0].mean(0)

def preprocess(images_labels):
    images, labels = images_labels
    images = images.astype(np.float32).reshape(-1, 1, 28, 28)
    images -= train_image_mean.reshape(1, 1, 28, 28)
    images += 1  # TODO: why is this necessary?? why is dp shifted like this?
    return images, labels

train, test = preprocess(train), preprocess(test)

# n = 1000
n = 2000
# n = 10000
train = train[0][:n], train[1][:n]

n = 100
# n = 2000
test = test[0][:n], test[1][:n]

classes = np.unique(train[1])

# --- cuda-convnet network
print("Computing class means...")

cc_model = load_model_pickle('mnist-lif-0097.pkl')
layers = cc_model['model_state']['layers']
# import pdb; pdb.set_trace()

# okey = 'fc1'
okey = 'dropout1'
train_outputs = compute_target_layer(okey, layers, train)

cmeans = []
for c in classes:
    cmean = train_outputs[okey][train[1] == c].mean(0)
    cmeans.append(cmean)

cmeans = np.array(cmeans)
# cmeans[:10] = layers['fc10']['weights'][0].T
cmeans /= np.sqrt((cmeans**2).sum(axis=1, keepdims=True))

cmeans[:] = 0
cmeans[:10] = layers['fc10']['weights'][0].T


if 0:
    cmean_dots = np.dot(cmeans, cmeans.T)

    cmean_mean = cmeans.mean(0)
    cmeans2 = cmeans - cmean_mean
    cmeans2 /= np.sqrt((cmeans2**2).sum(axis=1, keepdims=True))
    cmean2_dots = np.dot(cmeans2, cmeans2.T)

    print(cmean2_dots)


if 1:  # test statically
    test_outputs = compute_target_layer(okey, layers, test)
    z = np.argmax(np.dot(test_outputs[okey], cmeans.T), axis=1)

    nclasses = classes[-1] + 1
    confusion = np.zeros((nclasses, nclasses), dtype=np.int32)
    for i in range(nclasses):
        confusion[i] = np.bincount(z[test[1] == i], minlength=nclasses)

    print(confusion)
    print((z != test[1]).mean())

    if 'cmean_mean' in locals():
        z2 = np.argmax(np.dot(test_outputs[okey] - cmean_mean, cmeans2.T), axis=1)
        print((z2 != test[1]).mean())

    assert 0

# --- Spiking network
presentation_time = 0.1

with nengo.Network() as model:
    u = nengo.Node(nengo.processes.PresentInput(test[0], presentation_time))
    ccnet = CudaConvnetNetwork(cc_model, synapse=nengo.synapses.Alpha(0.005))
    nengo.Connection(u, ccnet.inputs['data'], synapse=None)

    output_p = nengo.Probe(ccnet.layer_outputs[okey], synapse=0.01)

with nengo_ocl.Simulator(model) as sim:
    n_pres = 1000
    sim.run(n_pres * presentation_time)

nt = int(presentation_time / sim.dt)
similarities = np.dot(sim.data[output_p], cmeans.T)
blocks = similarities.reshape(n_pres, nt, cmeans.shape[0])
choices = np.argmax(blocks[:, -20:, :].mean(axis=1), axis=1)
accuracy = (choices == test[1][:n_pres]).mean()
print('Spiking accuracy (%d examples): %0.3f' % (n_pres, accuracy))
