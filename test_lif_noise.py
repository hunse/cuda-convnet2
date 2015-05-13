import matplotlib.pyplot as plt
import numpy as np
import nengo
plt.ion()

tend = 100.0

# synapse = nengo.synapses.Lowpass(0.005)
synapse = nengo.synapses.Alpha(0.005)
# synapse = nengo.synapses.Alpha(0.01)


with nengo.Network() as model:
    u = nengo.Node(nengo.processes.WhiteNoise(tend, high=0.1).f())
    # u = nengo.Node(1.0)
    # a = nengo.Ensemble(1, 1, encoders=[[1]])
    a = nengo.Ensemble(1000, 1)
    nengo.Connection(u, a, synapse=None)
    up = nengo.Probe(u)
    jp = nengo.Probe(a.neurons, 'input')
    rp = nengo.Probe(a.neurons, synapse=synapse)

    # v = nengo.Node(0)
    b = nengo.Ensemble(1000, 1)
    b.gain = np.zeros(1000)
    b.bias = np.linspace(-5, 35, 1000)
    kp = nengo.Probe(b.neurons, 'input')
    sp = nengo.Probe(b.neurons, synapse=synapse)

sim = nengo.Simulator(model)
sim.run(tend)

t = sim.trange()
tmin = 1.0
# u = sim.data[up]
# x = np.dot(u, sim.data[a].encoders.T)
x1 = sim.data[jp][t > tmin].ravel()
y1 = sim.data[rp][t > tmin].ravel()
x2 = sim.data[kp][t > tmin].ravel()
y2 = sim.data[sp][t > tmin].ravel()

# x = x.ravel()
# y = y.ravel()
m = (x1 >= -5)
x1, y1 = x1[m], y1[m]
m = (x2 >= -5)
x2, y2 = x2[m], y2[m]

# H, xedges, yedges = np.histogram2d(x.ravel(), y.ravel(), bins=(31, 30))

# plt.figure()
# plt.contourf(xedges[:-1], yedges[:-1], H.T)


bins = np.linspace(-5, 35, 101)
binc = 0.5 * (bins[:-1] + bins[1:])
means1 = np.zeros(len(binc))
stds1 = np.zeros(len(binc))
means2 = np.zeros(len(binc))
stds2 = np.zeros(len(binc))

def mean_std(x, y, lower, upper):
    yi = y[(x >= lower) & (x < upper)]
    mean = yi.mean() if yi.size > 0 else 0
    std = yi.std() if yi.size > 1 else 0
    return mean, std


for i, [lower, upper] in enumerate(zip(bins[:-1], bins[1:])):
    means1[i], stds1[i] = mean_std(x1, y1, lower, upper)
    means2[i], stds2[i] = mean_std(x2, y2, lower, upper)


# analytical curve
tau_rc = 0.02
tau_ref = 0.002
yref = np.zeros(len(bins))
yref[bins > 1] = 1. / (tau_ref + tau_rc * np.log1p(1./bins[bins > 1]))

# dtir = sim.dt / 10.
# tir = dtir * np.

# means0 = np.zeros(len(bins))
# stds0 = np.zeros(len(bins))

plt.figure()
plt.plot(bins, yref, 'k')
plt.errorbar(binc, means1, stds1)
plt.errorbar(binc, means2, stds2)

plt.figure()
plt.plot(binc, stds1)
plt.plot(binc, stds2)

# plt.figure()
# plt.plot(binc, stds / means)

# plt.scatter(x.ravel(), y.ravel())
# plt.xlim([-4, 40])

plt.show()
