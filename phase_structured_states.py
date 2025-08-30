from numpy import linspace, array, pi, sin, cos, sqrt, dot, exp, pi, meshgrid, arctan2, real, imag, abs, ones, argmin, argmax, arctan, max, zeros, conj 
from scipy.integrate import simps 
from matplotlib import pyplot as plt 
from matplotlib import rc, cm, figure 
from scipy import special 
import numpy as np 
from scipy.special import factorial
import matplotlib.cm as cm 
from sys import exit

## Cylindrical coordinate transformation
def rho(x,y): 
    return sqrt((x)**2 + (y)**2)
def phi(x,y): 
    return arctan2(y,x)
def rho_vec(x,y): 
    return array( [cos(phi(x,y)), sin(phi(x,y))] )
def phi_vec(x,y): 
    return array( [-sin(phi(x,y)), cos(phi(x,y))] )

## Hermite - Gauss wavefunctions, Cartesian
def wfxn_HG(x,y,n,m,w0): 
    hrmite_n = special.hermite(n) 
    hrmite_m = special.hermite(m) 
    return (1/w0)*sqrt(2/( pi*factorial(n)*factorial(m) ))*(2**(-(n+m)/2))*hrmite_n(x*sqrt(2)/w0)*hrmite_m(y*sqrt(2)/w0)*exp(-(x**2 + y**2)/(w0**2))

def phs_HG(x,y,n,m,w0):
    return arctan2(imag(wfxn_HG(x,y,n,m,w0)), real(wfxn_HG(x,y,n,m,w0)))

## Laguerre - Gauss wavefunctions, Cartesian
def wfxn_LG(x,y,l,p,w0): 
    laguerre_l = special.genlaguerre(p,abs(l)) 
    return (1/w0)*sqrt(2*factorial(p)/( pi*factorial(abs(l)+p) ) )*((rho(x,y)*sqrt(2))/w0)**(abs(l))*(laguerre_l((2*rho(x,y)**2)/w0**2))*exp(-(rho(x,y)**2)/(w0**2))*exp(complex(0,1)*l*phi(x,y))

def phs_LG(x,y,l,p,w0):
    return arctan2(imag(wfxn_LG(x,y,l,p,w0)), real(wfxn_LG(x,y,l,p,w0)))

## Bessel wavefunctions -- Cartesian
def wfxn_Bes(x,y,kp,l): 
    J_l = special.jv(abs(l),kp*rho(x,y)) 
    return J_l*exp(complex(0,1)*l*phi(x,y))

def phs_Bes(x,y,kp,l): 
    return arctan2(imag(wfxn_Bes(x,y,kp,l)), real(wfxn_Bes(x,y,kp,l)))

## Bessel wavefunctions -- Cartesian
def wfxn_pw(x,y,kx,ky): 
    scale = 1
    return scale*exp(complex(0,1)*(kx*x + ky*y))

def phs_pw(x,y,kx,ky): 
    return arctan2(imag(wfxn_pw(x,y,kx,ky)), real(wfxn_pw(x,y,kx,ky)))

## Set up cartesian space
x_min = -3.0*10**(-9)
x_max = 3.0*10**(-9)
x_num = 1000
y_min = -3.0*10**(-9)
y_max = 3.0*10**(-9)
y_num = 1000

## initialize 2D evaluation grid (xy):
x_vals = linspace(x_min, x_max, num=x_num)
y_vals = linspace(y_min, y_max, num=y_num)
X, Y = meshgrid(x_vals, y_vals)

## Quantum numbers and other
w0 = 1.5*10**(-9) 
lmbda = 2*x_max
aa = 1
kp = 2*pi/(lmbda)*2
kx = 0
ky = 2*pi/(lmbda)
l = 1
p = 1
n = 0
m = 0
clr = cm.gray
phs_clr = cm.hsv
# phs_clr = cm.twilight

# # Plot plane wave wavefunctions
# fig, ax = plt.subplots(figsize=(5, 5))
# psi = wfxn_pw(X, Y,kx,ky)
# alpha_arr = 1-abs(psi)/max(abs(psi))
# mask = zeros((x_num, y_num))
# imshow_kwargs = {'vmax': 3.14, 'vmin': -3.14, 'cmap': phs_clr,'aspect': 1,'interpolation': 'None'}
# ax.imshow(phs_pw(X,Y,kx,ky), **imshow_kwargs, extent=[X.min(), X.max(), Y.min(), Y.max()])
# imshow_kwargs1 = {'vmax': 1, 'vmin': 0,'cmap': clr,'aspect': 1,'interpolation': 'None'}
# ax.imshow(mask, **imshow_kwargs1, extent=[X.min(), X.max(), Y.min(), Y.max()], alpha=alpha_arr)
# plt.gca().set_aspect("equal")
# plt.xlim((X.min(),X.max()))
# plt.ylim((Y.min(),Y.max()))
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()
# plt.show()   

# Plot Bessel wavefunctions
# fig, ax = plt.subplots(figsize=(5, 5))
# psi = wfxn_Bes(X, Y, kp,l)
# alpha_arr = 1-abs(psi)/max(abs(psi))
# mask = zeros((x_num, y_num))
# imshow_kwargs = {'vmax': 3.14, 'vmin': -3.14, 'cmap': phs_clr,'aspect': 1,'interpolation': 'None'}
# ax.imshow(phs_Bes(X,Y,kp,l), **imshow_kwargs, extent=[X.min(), X.max(), Y.min(), Y.max()])
# imshow_kwargs1 = {'vmax': 1, 'vmin': 0,'cmap': clr,'aspect': 1,'interpolation': 'None'}
# ax.imshow(mask, **imshow_kwargs1, extent=[X.min(), X.max(), Y.min(), Y.max()], alpha=alpha_arr)
# plt.gca().set_aspect("equal")
# plt.xlim((X.min(),X.max()))
# plt.ylim((Y.min(),Y.max()))
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()
# plt.show()

# # Plot Laguerre - Gauss wavefunctions
fig, ax = plt.subplots(figsize=(5, 5))
psi = wfxn_LG(X, Y, l,p, w0)
alpha_arr = 1-abs(psi)/max(abs(psi))
mask = zeros((x_num, y_num))
imshow_kwargs = {'vmax': 3.14, 'vmin': -3.14, 'cmap': phs_clr,'aspect': 1,'interpolation': 'None'}
ax.imshow(phs_LG(X,Y,l,p,w0), **imshow_kwargs, extent=[X.min(), X.max(), Y.min(), Y.max()])
imshow_kwargs1 = {'vmax': 1, 'vmin': 0,'cmap': clr,'aspect': 1,'interpolation': 'None'}
ax.imshow(mask, **imshow_kwargs1, extent=[X.min(), X.max(), Y.min(), Y.max()], alpha=alpha_arr)
plt.gca().set_aspect("equal")
plt.xlim((X.min(),X.max()))
plt.ylim((Y.min(),Y.max()))
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()        

# # Plot Hermite - Gauss wavefunctions
fig, ax = plt.subplots(figsize=(5, 5))
psi = wfxn_HG(X, Y, n,m, w0)
alpha_arr = 1-abs(psi)/max(abs(psi))
mask = zeros((x_num, y_num))
imshow_kwargs = {'vmax': 3.14, 'vmin': -3.14, 'cmap': phs_clr,'aspect': 1,'interpolation': 'None'}
ax.imshow(phs_HG(X,Y,n,m,w0), **imshow_kwargs, extent=[X.min(), X.max(), Y.min(), Y.max()])
imshow_kwargs1 = {'vmax': 1, 'vmin': 0,'cmap': clr,'aspect': 1,'interpolation': 'None'}
ax.imshow(mask, **imshow_kwargs1, extent=[X.min(), X.max(), Y.min(), Y.max()], alpha=alpha_arr)
plt.gca().set_aspect("equal")
plt.xlim((X.min(),X.max()))
plt.ylim((Y.min(),Y.max()))
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()       
