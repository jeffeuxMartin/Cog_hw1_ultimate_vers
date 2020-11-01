import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from scipy import signal as dsp
import thorns.waves as wv
import cochlea
import thorns as th
import contextlib

fs = 100e3
cf = 1000

fname = 'newSong__T__1604008780.511391.wav'
fname = 'newSong__T__1604008788.747618.wav'
fname = 'newSong__AA0__1604008730.3519979.wav'

with contextlib.closing(wave.open(fname,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

spf = wave.open(fname, "r")

# Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.frombuffer(signal, dtype=np.int16)
number_lv = np.ceil(np.log10(np.max(signal)))

spf.close()

signal = signal / 10. ** number_lv

signal = dsp.resample(signal, int(fs * duration))

plt.figure(1)
plt.title("Signal Wave...")
wv.plot_signal(signal, int(fs))
plt.show()

anf_trains = cochlea.run_zilany2014(
    signal,
    int(fs),
    anf_num=(0,0,50),
    cf=(125, 10e3, 100),
    seed=0,
    species='human'
)

th.plot_raster(th.accumulate(anf_trains))
plt.show()
