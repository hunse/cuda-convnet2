import numpy as np

dt = 0.001

# -- MNIST
# neurons = [12544, 12544, 2000]
# synapses = [313600, 5017600, 6272000, 20000]
# rates = [4.6, 15.6, 4.2]
# presentation_time = 0.2

# -- CIFAR-10
# neurons = [36864, 9216, 2304, 1152]
# synapses = [2764800, 14745600, 1327104, 663552, 11520]
# rates = [173.3, 99.0, 9.7, 7.2]
# presentation_time = 0.2

# -- Imagenet
neurons = [193600, 139968, 64896, 43264, 43264, 4096, 4096]
synapses = [70276800, 223948800, 112140288, 149520384, 99680256, 37748736, 16777216, 4096000]
rates = [178.1, 48.8, 26.6, 30.6, 35.6, 19.1, 10.7]
# rates = [1000.0, 178.1, 48.8, 26.6, 30.6, 35.6, 19.1, 10.7]
# presentation_time = 0.08
presentation_time = 0.2

neurons = np.array(neurons, dtype=float)
synapses = np.array(synapses, dtype=float)
rates = np.array(rates, dtype=float)
average_rate = (rates * neurons).sum() / neurons.sum()

# --- compute flops on standard hardware
flops_update = 1
flops_synop = 2
# flops = flops_synop * synapses.sum() + flops_update * neurons.sum()
flops0 = flops_synop * synapses[0]
flops = flops0 + flops_synop * synapses[1:].sum() + flops_update * neurons.sum()

# --- compute energy on neuromorphic hardware
flopjoules_update = 0.25
flopjoules_synop = 0.08
# synops = (synapses * rates).sum()
synops = (synapses[1:] * rates).sum()
updates = neurons.sum() / dt
energy = flops0 + (flopjoules_synop * synops + flopjoules_update * updates) * presentation_time

print("Average rate: %s" % (average_rate,))
print("Synops/s = %s, updates/s = %s" % (synops, updates))
print("flops = %0.2e, energy = %0.2e, efficiency = %0.2f" % (flops, energy, (flops / energy)))
