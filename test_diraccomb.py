import matplotlib.pyplot as plt
import numpy as np
import nengo
plt.ion()

tau = 0.005
# s = nengo.synapses.Lowpass(tau)
s = nengo.synapses.Alpha(tau)

if 0:
    r = 100.
    p = 1. / r

    dt = 0.0001
    t = dt * np.arange(int(round(tau / dt)))
    x = np.zeros_like(t)
    tmp = t % p
    x[(tmp <= 0.5*dt) | (tmp > p - 0.5*dt)] = 1.
    y = nengo.synapses.filt(x, s, dt)

    # # f = np.fft.fftfreq(len(t), d=dt)
    # n = len(t)
    # f = np.arange(n) / (n * dt)
    # fs = np.fft.fftfreq(len(t), d=dt)
    # X = np.fft.fft(x)
    # X[abs(X) < 1e-8] = 0
    # X.imag[abs(X.imag) < 1e-8] = 0
    # Y = X * (1. / (tau * 2 * np.pi * 1.j * fs + 1))
    # # Y = X * (1. / (tau * 1.j * fs + 1))
    # y2 = np.fft.ifft(Y)
    # # print X.real
    # # print X.imag

    # f = np.fft.rfftfreq(len(t), d=dt)
    X0 = np.fft.rfft(x)
    # Y = X * (1. / (tau * 2 * np.pi * 1.j * f + 1))
    # y2 = np.fft.irfft(Y)


    X0 = np.fft.rfft(x)

    f = np.fft.rfftfreq(len(t), d=dt)
    df = f[1]
    # X = np.zeros(len(f), dtype='complex128')
    X = np.zeros(len(f))
    fmr = f % r
    X[(fmr <= 0.5*df) | (fmr > (r - 0.5*df))] = 1.

    Y = X * (1. / (tau * 2 * np.pi * 1.j * f + 1))
    y2 = np.fft.irfft(Y)

    Y3 = np.zeros_like(Y)
    for i in range(-3, 4):
        fi = f + i * (1. / dt)

        Xi = np.zeros(len(fi))
        fmr = fi % r
        Xi[(fmr <= 0.5*df) | (fmr > (r - 0.5*df))] = 1.

        Y3 += Xi * (1. / (tau * 2 * np.pi * 1.j * fi + 1))

    y3 = np.fft.irfft(Y3)

    # plt.figure()
    # plt.plot(t, x)

    plt.figure()
    plt.plot(t, y)
    plt.plot(t, y2)
    plt.plot(t, y3)

    plt.figure()
    plt.plot(f, X0.real)
    plt.plot(f, X.real)



def filtered_mean_std1(rates):
    dt = 0.001

    tmax = 1.0
    tclip = 0.03  # at least 5 * tau?

    t = dt * np.arange(int(round(tmax / dt)))
    x = np.zeros((len(t), len(rates)))
    for i, r in enumerate(rates):
        p = 1. / r
        tmp = t % p
        x[(tmp <= 0.5*dt) | (tmp > p - 0.5*dt), i] = 1. / dt

    y = nengo.synapses.filt(x, s, dt, axis=0)

    # TODO: don't clip, take sections with full periods between spikes
    y = y[t > tclip]
    return y.mean(axis=0), y.std(axis=0)


def filtered_mean_std2(rates):
    dt = 0.001

    tmax = 1.0
    tclip = 0.03

    t = dt * np.arange(int(round(tmax / dt)))
    f = np.fft.rfftfreq(len(t), d=dt)
    df = f[1]

    # Y = X * (1. / (tau * 2 * np.pi * 1.j * f[:, None] + 1))
    # y = np.fft.irfft(Y)

    Y = np.zeros((len(f), len(rates)), dtype='complex128')
    for i in range(-3, 4):
        fi = f + i * (1. / dt)

        X = np.zeros(Y.shape)
        for j, r in enumerate(rates):
            fmr = fi % r
            X[(fmr <= 0.5*df) | (fmr > (r - 0.5*df)), j] = 1.

        # Xi = np.zeros(len(fi))
        # fmr = fi % r
        # Xi[(fmr <= 0.5*df) | (fmr > (r - 0.5*df))] = 1.

        Y += X * (1. / (tau * 2 * np.pi * 1.j * fi[:, None] + 1))

    y = np.fft.irfft(Y, axis=0)
    return y.mean(axis=0), y.std(axis=0)


tau_rc = 0.02
tau_ref = 0.002
j = np.linspace(-5, 35, 51)
rates = np.zeros(len(j))
rates[j > 1] = 1. / (tau_ref + tau_rc * np.log1p(1./j[j > 1]))

means1, stds1 = filtered_mean_std1(rates)
means2, stds2 = filtered_mean_std2(rates)

plt.figure()
plt.plot(j, rates, 'k')
plt.errorbar(j, means1, stds1)
plt.errorbar(j, means2, stds2)

# plt.

# Y += X * (1. /



# for i, ff in enumerate(f):
    # if
# X[abs(f
