import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import contextlib
import scipy.signal as dsp
import thorns.waves as wv

fname = 'newSong__T__1604008780.511391.wav'
fname = 'newSong__T__1604008788.747618.wav'
fname = 'newSong__AA0__1604008730.3519979.wav'

with contextlib.closing(wave.open(fname,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

fs = 100e3
cf = 1000

spf = wave.open(fname, "r")

# Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.frombuffer(signal, dtype=np.int16)
number_lv = np.ceil(np.log10(np.max(signal)))
spf.close()

signal = signal / 10. ** number_lv

res_ = dsp.resample(signal, int(fs * duration))

plt.figure(1)
wv.plot_signal(res_, int(fs))
plt.title("Signal Wave...")
plt.show()

sound = lambda t: res_[np.round(t % duration * fs).astype(int) - 1]

# [+] import packages
import numpy as np, nengo, matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot

model = nengo.Network(
    label="Decoding with Two Neurons")
with model:
    # stim = nengo.Node(lambda t: np.sin(10 * t))
    stim = nengo.Node(sound)
    
    ens = nengo.Ensemble(2, dimensions=1,
                            encoders=[[1], [-1]],
                            intercepts=[-.5, -.5],
                            max_rates=[100, 100])
    nengo.Connection(stim, ens)
    stim_p = nengo.Probe(stim)
    spikes_p = nengo.Probe(ens.neurons, "spikes")

sim = nengo.Simulator(model)
sim.run(.6)

# [+] draw figures
nengo.utils.matplotlib.plot_tuning_curves(ens, sim)
t = sim.trange()
plt.figure(figsize=(12, 6))
plt.plot(t, sim.data[stim_p], "g")
ax = plt.gca()
plt.ylabel("Output")
plt.xlabel("Time");
rasterplot(t, sim.data[spikes_p], 
           ax=ax.twinx(), use_eventplot=True)
plt.ylabel("Neuron")
plt.show()
