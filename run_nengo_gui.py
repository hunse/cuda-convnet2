import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_gui
nengo.log(level='info')

from run_nengo import build_target_layer, load_network


import base64
import PIL
import cStringIO


backend = 'nengo_ocl'

loadfile = 'checkpoints/ilsvrc2012-lif-48'
layers, data, dp = load_network(loadfile)

# image = data[0][0]
# image = np.transpose(image, (1, 2, 0))
# print(image.min(), image.max())

# plt.imshow((image / 255. + 0.5).clip(0, 1))
# plt.show()



# --- build network in Nengo
model = nengo.Network()

presentation_time = 0.2

model.config[nengo.Connection].synapse = nengo.synapses.Lowpass(0.0)
# model.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.003)

outputs = build_target_layer(
    'logprob', layers, data, model, pt=presentation_time)

label_names = dp.batch_meta['label_names']
vocab_names = [
    (name.split(',')[0] if ',' in name else name).upper().replace(' ', '_')
    for i, name in enumerate(label_names)]
vocab_vectors = np.eye(len(vocab_names))

# number duplicates
unique = set()
duplicates = []
for name in vocab_names:
    if name in unique:
        duplicates.append(name)
    else:
        unique.add(name)

duplicates = {name: 0 for name in duplicates}
for i, name in enumerate(vocab_names):
    if name in duplicates:
        vocab_names[i] = '%s%d' % (name, duplicates[name])
        duplicates[name] += 1


with model:
    vocab = nengo.spa.Vocabulary(len(vocab_names))
    for name, vector in zip(vocab_names, vocab_vectors):
        vocab.add(name, vector)

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        output = nengo.spa.State(len(vocab_names), subdimensions=10, vocab=vocab)
    nengo.Connection(outputs['probs'], output.input, transform=1.0, synapse=nengo.synapses.Alpha(0.01))

    # --- image display
    input_image = outputs['data']
    input_shape = data[0].shape[1:]

    def display_func(t, x):
        values = x.reshape(input_shape)
        values = values.transpose((1,2,0))
        # values = values * 255.
        values = (values / 255. + 0.5).clip(0, 1) * 255.

        values = values.astype('uint8')

        png = PIL.Image.fromarray(values)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())

        display_func._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

    display_node = nengo.Node(display_func, size_in=input_image.size_out)
    nengo.Connection(input_image, display_node, synapse=None)

for obj in model.all_nodes + model.all_ensembles:
    obj.label = ''

# from nengo.utils.builder import remove_passthrough_nodes
# model = remove_passthrough_nodes(model)



# test whole network
# with model:
#     # xp = nengo.Probe(outputs['data'])
#     yp = nengo.Probe(outputs['probs'])
#     zp = nengo.Probe(outputs['logprob'])

#     spikes_p = {name: nengo.Probe(outputs[name])
#                 for name in layers if layers[name]['type'] == 'neuron'}

nengo_gui.GUI(__file__, backend=backend).start()
