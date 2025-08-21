import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy as np
import scipy.signal as sign
from scipy import fft

f = [.038580777748, .126969125396, -.077161555496, -.607491641386,
     .745687558934, -.226584265197]
N, fs = 512, 1024
t = np.arange(N) / fs

# f_mirr = np.zeros(len(f))
# for i in range(len(f)):
#     f_mirr[i] = (-1) ** (i) * f[i]

f_mirr = -((-1) ** np.arange(1, len(f) + 1)) * f

fig0 = plt.figure("wavelets")
plt.plot(f, 'g', label='high-pass')
plt.plot(f_mirr, 'r', label='low-pass')
plt.ylabel('filter')
plt.legend()

harmonic = np.sin(2 * np.pi * t)
chirp = sign.chirp(t, f0=10, f1=200, t1=N / fs, )
rab_fs, rabarbar = scipy.io.wavfile.read('rabarbar8k.wav')

fig1, ax1 = plt.subplots(nrows=1, ncols=3, layout='constrained', num="TEST")
ax1[0].plot(t, harmonic, 'r')
ax1[1].plot(t, chirp, 'g')
ax1[2].plot(rabarbar)

fig2, ax2 = plt.subplots(nrows=2, ncols=3, layout='constrained', num="low_high")

h_harmonic = np.convolve(harmonic, f, mode='full')
h_chirp = np.convolve(chirp, f, mode='full')
h_rabarbar = np.convolve(rabarbar, f, mode='full')

ax2[0][0].plot(h_harmonic, 'g',label='high-pass')
ax2[0][1].plot(h_chirp, 'g')
ax2[0][2].plot(h_rabarbar, 'g')
ax2[0][0].legend()

l_harmonic = np.convolve(harmonic, f_mirr, mode='full')
l_chirp = np.convolve(chirp, f_mirr, mode='full')
l_rabarbar = np.convolve(rabarbar, f_mirr, mode='full')

ax2[1][0].plot(l_harmonic, 'r', label='low-pass')
ax2[1][1].plot(l_chirp, 'r')
ax2[1][2].plot(l_rabarbar, 'r')
ax2[1][0].legend()

convo_low = np.convolve(chirp, f_mirr)
# print(len(convo_low))
signal = rabarbar

convo_high = np.convolve(chirp, f, mode='full')
convo_high = sign.decimate(convo_high, 2)

scalogram = np.array([convo_high])

for i in range(15):
    convo_high = np.convolve(signal, f, mode='full')

    convo_low = np.convolve(signal, f_mirr, mode='full')
    convo_high = sign.resample(convo_high, int(len(t) / (2 ** i)))
    signal = sign.resample(convo_low, int(len(t) / (2 ** i)))

    if len(signal)<20:
        break
    print(signal)

fig3 = plt.figure("Final")
plt.plot(signal, label="lowpass")
plt.plot(convo_high, label="highpass")
plt.legend()

plt.show()

plt.show()
