
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
classes = np.unique(train[1])

# --- cuda-convnet network
cc_model = load_model_pickle('spaun-lif-0084.pkl')
layers = cc_model['model_state']['layers']

okey = 'fc10'

if 0:  # test statically
    test_outputs = compute_target_layer(okey, layers, test)
    z = np.argmax(test_outputs[okey], axis=1)

    nclasses = classes[-1] + 1
    confusion = np.zeros((nclasses, nclasses), dtype=np.int32)
    for i in range(nclasses):
        confusion[i] = np.bincount(z[test[1] == i], minlength=nclasses)

    print(confusion)
    print((z != test[1]).mean())

    assert 0

# --- Spiking network
presentation_time = 0.1

with nengo.Network() as model:
    u = nengo.Node(nengo.processes.PresentInput(test[0], presentation_time))
    ccnet = CudaConvnetNetwork(cc_model, synapse=nengo.synapses.Alpha(0.005))
    nengo.Connection(u, ccnet.inputs['data'], synapse=None)

    output_p = nengo.Probe(ccnet.layer_outputs[okey], synapse=0.01)

with nengo_ocl.Simulator(model) as sim:
    # n_pres = 10
    n_pres = 1000
    sim.run(n_pres * presentation_time)

nt = int(presentation_time / sim.dt)
outputs = sim.data[output_p]
blocks = outputs.reshape(n_pres, nt, outputs.shape[-1])
choices = np.argmax(blocks[:, -20:, :].mean(axis=1), axis=1)
accuracy = (choices == test[1][:n_pres]).mean()
print('Spiking accuracy (%d examples): %0.3f' % (n_pres, accuracy))
