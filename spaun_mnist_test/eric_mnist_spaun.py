
import numpy as np

from run_numpy import compute_target_layer

import mnist

import nengo
import nengo_ocl
from nengo_extras.cuda_convnet import load_model_pickle, CudaConvnetNetwork


cc_model = load_model_pickle('checkpoints/mnist-lif-0097.pkl')
# print(cc_model['op'].keys())
# assert 0


def get_dp(load_dic, multiview=False):
    from convdata import DataProvider

    options = load_dic['op']
    # for o in load_dic['op'].get_options_list():
        # options[o.name] = o.value

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

    return dp


def data_from_dp(load_dic, multiview=False):
    dp = get_dp(load_dic, multiview=multiview)
    epoch, batchnum, data = dp.get_next_batch()

    data = data[0].T.reshape(-1, 1, 28, 28), data[1].ravel().astype(np.int64)
    return data


if 1:
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

else:
    data = data_from_dp(cc_model)
    train, test = data, data
    print train[0].shape

# n = 1000
n = 2000
# n = 10000
train = train[0][:n], train[1][:n]

# n = 100
n = 2000
test = test[0][:n], test[1][:n]

classes = np.unique(train[1])

print(train[0].mean(), train[0].min(), train[0].max())

layers = cc_model['model_state']['layers']
# outputs = compute_target_layer('logprob', layers, test)

if 1:
    okey = 'fc1'
    train_outputs = compute_target_layer(okey, layers, train)
    test_outputs = compute_target_layer(okey, layers, test)

    pointers = []
    for c in classes:
        pointer = train_outputs[okey][train[1] == c].mean(0)
        pointers.append(pointer)

    pointers = np.array(pointers)
    pointers /= np.sqrt((pointers**2).sum(axis=1, keepdims=1))

    z = np.argmax(np.dot(test_outputs[okey], pointers.T), axis=1)
else:
    # outputs = compute_target_layer('logprob', layers, train)
    outputs = compute_target_layer('logprob', layers, test)
    z = np.argmax(outputs['probs'], axis=1)

nclasses = classes[-1] + 1
confusion = np.zeros((nclasses, nclasses), dtype=np.int32)
for i in range(nclasses):
    confusion[i] = np.bincount(z[test[1] == i], minlength=nclasses)

print(confusion)

print((z != test[1]).mean())


# --- Spiking network
presentation_time = 0.1

with nengo.Network() as model:
    u = nengo.Node(nengo.processes.PresentInput(test[0], presentation_time))
    ccnet = CudaConvnetNetwork(cc_model, synapse=nengo.synapses.Alpha(0.005))
    nengo.Connection(u, ccnet.inputs['data'], synapse=None)

    output_p = nengo.Probe(ccnet.layer_outputs['fc1'], synapse=0.01)


with nengo_ocl.Simulator(model) as sim:
    n_pres = 100
    sim.run(n_pres * presentation_time)


nt = int(presentation_time / sim.dt)
similarities = np.dot(sim.data[output_p], pointers.T)
blocks = similarities.reshape(n_pres, nt, pointers.shape[0])
choices = np.argmax(blocks[:, -20:, :].mean(axis=1), axis=1)
accuracy = (choices == test[1][:n_pres]).mean()
print('Spiking accuracy (%d examples): %0.3f' % (n_pres, accuracy))
