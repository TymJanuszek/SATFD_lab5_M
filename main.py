import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy as np
import scipy.signal as sign
from scipy import fft

f = [ .038580777748 , .126969125396 , -.077161555496 , -.607491641386 ,
.745687558934 , -.226584265197 ]
N, fs = 512, 1024
t = np.arange(N) / fs

f_mirr = np.zeros(len(f))
for i in range(len(f)):
    f_mirr[i] = (-1) ** (i) * f[i]

fig0 = plt.figure("wavelets")
plt.plot(f, 'g', label='high-pass')
plt.plot(f_mirr, 'r', label='low-pass')
plt.ylabel('filter')
plt.legend()



harmonic = np.sin(50*np.pi*t)
chirp = sign.chirp(t, f0=10, f1=200, t1=N / fs, )
rab_fs, rabarbar = scipy.io.wavfile.read('rabarbar8k.wav')

fig1, ax1 = plt.subplots(nrows=1, ncols=3, layout='constrained', num="TEST")
ax1[0].plot(t, harmonic, 'r')
ax1[1].plot(t, chirp, 'g')
ax1[2].plot(rabarbar)


plt.show()