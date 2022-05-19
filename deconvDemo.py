import numpy as np
import matplotlib.pyplot as plt
from scipy import pi
from numpy.random import normal
from numpy.random import poisson
from scipy.signal import convolve
import time

import deconvLib as dcv
import FRC_lib as FRC

plt.close('all')

#%% generate PSF

N = 256

L = 10

x = np.linspace(-L/2, L/2, num = N)

X, Y = np.meshgrid(x,x)

mu = 0
sigma = 0.1
h = dcv.gauss2d(X, Y, mu, sigma)
h /= np.sum(h)
# T = 1
# h = 1*disk2d(X, Y, T)

plt.figure()
plt.imshow(h)

#%% Generate object

o = np.zeros([N,N])

o[N//3,N//3] = 1
o[N//2,N//4] = 1
o[N//2 + N//8,N//2 + N//9] = 1

plt.figure()
plt.imshow(o)

#%% Generate images

noise = 1
signal = 5e2

i_0 = signal*convolve(o,h,mode='same')
i_0[i_0<0] = 0

i_g = [i_0 + normal(0, noise, o.shape), i_0 + normal(0, noise, o.shape)]

i_p = [poisson(lam = i_0), poisson(lam = i_0)]

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(i_g[0])
ax[0,1].imshow(i_g[1])
ax[1,0].imshow(i_p[0])
ax[1,1].imshow(i_p[1])

#%%

pxsize = L/N

res, k, th, frc_smooth, frc = FRC.FRC_resolution(i_p[0], i_p[1], px = pxsize, method = 'fixed')

plt.figure()
plt.plot(k, frc, '.')
plt.plot(k, frc_smooth)
plt.plot(k, th)
plt.ylim( ( 0, 1 ) )

sigma_meas = res / (2*np.sqrt(2*np.log(2)))
h_frc = dcv.gauss2d(X, Y, 0, sigma_meas)
h_frc /= np.sum(h_frc)

plt.figure()
plt.imshow(h)

#%%

regW = 1e-4

obj_g = dcv.deconv_Wiener_FFT(h, i_p[0], reg = regW)
obj_g_frc = dcv.deconv_Wiener_FFT(h_frc, i_p[0], reg = regW)

eps = 1e-4

obj_p = dcv.deconv_RL_FFT(h, i_p[0], max_iter = 50, epsilon = eps, reg = 0)
obj_p_frc = dcv.deconv_RL_FFT(h_frc, i_p[0], max_iter = 50, epsilon = eps, reg = 0)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(obj_g)
ax[0,1].imshow(obj_g_frc)
ax[1,0].imshow(obj_p)
ax[1,1].imshow(obj_p_frc)