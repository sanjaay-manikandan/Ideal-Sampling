# IDEAL, NATURAL, & FLAT-TOP -SAMPLING
# AIM
To write a simple Python program for the construction and reconstruction of ideal, natural, and flat-top sampling.
# TOOLS REQUIRED
Python IDE with Numpy and Scipy
# PROGRAM
### IDEAL SAMPLING
```import numpy as np
import matplotlib.pyplot as plt

fs = 100
f = 5
t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*f*t)

# Continuous signal
plt.figure(figsize=(10, 3))
plt.plot(t, x)
plt.title("Continuous Signal")
plt.grid()
plt.show()

# Ideal sampling
plt.figure(figsize=(10, 3))
plt.stem(t, x)
plt.title("Ideal Sampling")
plt.grid()
plt.show()

# Reconstruction (Interpolation)
t_rec = np.linspace(0, 1, 1000)
x_rec = np.interp(t_rec, t, x)

plt.figure(figsize=(10, 3))
plt.plot(t_rec, x_rec)
plt.title("Reconstructed Signal")
plt.grid()
plt.show()
```
### NATURAL SAMPLING
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

t = np.linspace(0,1,1000)
x = np.cos(2*np.pi*5*t)

p = ((t*50)%1 < 0.5)        # pulse train
xn = x * p                 # natural sampling

b,a = butter(4, 10/(0.5*1000))
xr = lfilter(b, a, xn)

plt.figure(figsize=(10,2))
plt.plot(t,x)
plt.title("Message Signal")
plt.grid()
plt.show()

plt.figure(figsize=(10,2))
plt.plot(t,p)
plt.title("Pulse Train")
plt.grid()
plt.show()

plt.figure(figsize=(10,2))
plt.plot(t,xn)
plt.title("Natural Sampling")
plt.grid()
plt.show()

plt.figure(figsize=(10,2))
plt.plot(t,xr)
plt.title("Reconstructed Signal")
plt.grid()
plt.show()

```
### FLAT-TOP SAMPLING
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

fs = 1000
t = np.linspace(0, 1, fs)
x = np.sin(2*np.pi*5*t)

Ts = fs//50                          # sampling interval
n = np.arange(0, fs, Ts)             # ideal sampling instants

xh = np.repeat(x[n], Ts)[:fs]        # flat-top (sample & hold)

b, a = butter(4, 10/(0.5*fs))
xr = filtfilt(b, a, xh)

plt.figure(figsize=(10,3))
plt.plot(t, x)
plt.title("Message Signal")
plt.grid()
plt.show()

plt.figure(figsize=(10,3))
plt.stem(t[n], x[n], basefmt=" ")
plt.title("Ideal Sampling Instances")
plt.grid()
plt.show()

plt.figure(figsize=(10,3))
plt.plot(t, xh)
plt.title("Flat-Top Sampling")
plt.grid()
plt.show()

plt.figure(figsize=(10,3))
plt.plot(t, xr)
plt.title("Reconstructed Signal")
plt.grid()
plt.show()
```
# OUTPUT WAVEFORM
### IDEAL SAMPLING
<img width="450" height="451" alt="image" src="https://github.com/user-attachments/assets/d0868cda-83f2-4780-a528-a7beafadf7e3" />

### NATURAL SAMPLING
<img width="491" height="715" alt="image" src="https://github.com/user-attachments/assets/998668fb-1f78-49db-aa94-1d69f27972da" />


### FLAT-TOP SAMPLING
<img width="527" height="748" alt="image" src="https://github.com/user-attachments/assets/ed97df3b-ad15-457a-a886-d9b6f25639a7" />


# RESULT
Thus, the python program for ideal sampling, natural sampling and flat-top sampling has been executed and verified successfully.
