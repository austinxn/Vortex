from numpy import linspace, array, pi, sin, cos, sqrt, dot, exp, pi, meshgrid, arctan2, real, imag, abs, ones, argmin, argmax, arctan, max, zeros, conj 
from scipy.integrate import simps 
from matplotlib import pyplot as plt 
from matplotlib import rc, cm, figure 
from scipy import special 
import numpy as np 
from scipy.special import factorial
import matplotlib.cm as cm 
from sys import exit


## Set up the basis vectors and functions
#########################################################
## Cylindrical coordinate transformation
def rho(x,y): 
    return sqrt((x)**2 + (y)**2)
def phi(x,y): 
    return arctan2(y,x)
def rho_vec(x,y): 
    return array( [cos(phi(x,y)), sin(phi(x,y))] )
def phi_vec(x,y): 
    return array( [-sin(phi(x,y)), cos(phi(x,y))] )
#########################################################


## Define the wavefunctions
#########################################################
## Hermite - Gauss wavefunctions
def wfxn_HG(x,y,n,m,w0): 
    n_a = abs(n)
    m_a = abs(m)
    hrmite_n = special.hermite(n_a) 
    hrmite_m = special.hermite(m_a) 
    return (1/w0)*sqrt(2/( pi*factorial(n_a)*factorial(m_a) ))*(2**(-(n_a+m_a)/2))*hrmite_n(x*sqrt(2)/w0)*hrmite_m(y*sqrt(2)/w0)*exp(-(x**2 + y**2)/(w0**2))

def phs_HG(x,y,n,m,w0):
    return arctan2(imag(wfxn_HG(x,y,n,m,w0)), real(wfxn_HG(x,y,n,m,w0)))
################################
## Laguerre - Gauss wavefunctions
def wfxn_LG(x,y,l,p,w0): 
    laguerre_l = special.genlaguerre(p,abs(l)) 
    return (1/w0)*sqrt(2*factorial(p)/( pi*factorial(abs(l)+p) ) )*((rho(x,y)*sqrt(2))/w0)**(abs(l))*(laguerre_l((2*rho(x,y)**2)/w0**2))*exp(-(rho(x,y)**2)/(w0**2))*exp(complex(0,1)*l*phi(x,y))

def phs_LG(x,y,l,p,w0):
    return arctan2(imag(wfxn_LG(x,y,l,p,w0)), real(wfxn_LG(x,y,l,p,w0)))
################################
## Bessel wavefunctions
def wfxn_Bes(x,y,kp,l): 
    J_l = special.jv(abs(l),kp*rho(x,y)) 
    return J_l*exp(complex(0,1)*l*phi(x,y))

def phs_Bes(x,y,kp,l): 
    return arctan2(imag(wfxn_Bes(x,y,kp,l)), real(wfxn_Bes(x,y,kp,l)))
################################
## Plane-waves -- Cartesian
def wfxn_pw(x,y,kx,ky): 
    return exp(complex(0,1)*(kx*x + ky*y))

def phs_pw(x,y,kx,ky): 
    return arctan2(imag(wfxn_pw(x,y,kx,ky)), real(wfxn_pw(x,y,kx,ky)))
#########################################################


## Define the effective transition current densities
#########################################
def J_Bes(x,y,kp_i,kp_f,l_i,l_f):
    l_ip1 = abs(l_i) + 1
    l_im1 = abs(l_i) - 1
    l_fp1 = abs(l_f) + 1
    l_fm1 = abs(l_f) - 1 
    J_i = special.jv(abs(l_i),kp_i*rho(x,y))
    J_f = special.jv(abs(l_f),kp_f*rho(x,y))
    J_ip1 = special.jv(abs(l_ip1),kp_i*rho(x,y))
    J_im1 = special.jv(abs(l_im1),kp_i*rho(x,y))
    J_fp1 = special.jv(abs(l_fp1),kp_f*rho(x,y))
    J_fm1 = special.jv(abs(l_fm1),kp_f*rho(x,y))
    rho_pf = complex(0,1)*((kp_i/2)*(J_f)*(J_im1-J_ip1) - (kp_f/2)*J_i*(J_fm1-J_fp1))
    phi_pf = -(1/rho(x,y))*(l_i + l_f)*J_i*J_f
    return -(rho_pf*rho_vec(x,y) + phi_pf*phi_vec(x,y))*exp(complex(0,1)*(l_i - l_f)*phi(x,y))

def phase_J_Bes(x,y,kp_i,kp_f,l_i,l_f):
    pf = phi_vec(x,y)[0]*J_Bes(x,y,kp_i,kp_f,l_i,l_f)[0] + phi_vec(x,y)[1]*J_Bes(x,y,kp_i,kp_f,l_i,l_f)[1]
    return arctan2( imag(pf), real(pf) )
#########################################
def J_LG(x,y,w0,p_i,p_f,l_i,l_f):
    lg_i = special.genlaguerre(p_i,abs(l_i))
    lg_ip1 = special.genlaguerre(p_i, abs(l_i+1))
    lg_f = special.genlaguerre(p_f,abs(l_f))
    lg_fp1 = special.genlaguerre(p_f, abs(l_f+1))
    pf_1 = (1/w0**2)*sqrt(2*factorial(p_i)/( pi*factorial(abs(l_i)+p_i) ) )*sqrt(2*factorial(p_f)/( pi*factorial(abs(l_f)+p_f) ) )
    pf_2 = (1/rho(x,y))*((rho(x,y)*sqrt(2))/w0)**(abs(l_i) + abs(l_f))*lg_i((2*rho(x,y)**2)/w0**2)*lg_f((2*rho(x,y)**2)/w0**2)
    rho_pf = complex(0,1)*(abs(l_i) - abs(l_f)) - (4*rho(x,y)**2/w0**2)*lg_i((2*rho(x,y)**2)/w0**2)*lg_f((2*rho(x,y)**2)/w0**2)**(-1)*(  lg_ip1((2*rho(x,y)**2)/w0**2)*lg_f((2*rho(x,y)**2)/w0**2) - lg_i((2*rho(x,y)**2)/w0**2)*lg_fp1((2*rho(x,y)**2)/w0**2) )
    phi_pf = -(l_i + l_f)
    return -pf_1*pf_2*(rho_pf*rho_vec(x,y) + phi_pf*phi_vec(x,y))*exp(-(2*rho(x,y)**2)/(w0**2))*exp(complex(0,1)*(l_i - l_f)*phi(x,y))

def phase_J_LG(x,y,w0,p_i,p_f,l_i,l_f):
    pf = phi_vec(x,y)[0]*J_LG(x,y,w0,p_i,p_f,l_i,l_f)[0] + phi_vec(x,y)[1]*J_LG(x,y,w0,p_i,p_f,l_i,l_f)[1]
    return arctan2( imag(pf), real(pf) )
#########################################
def J_HG(x,y,w0,n_i,m_i,n_f,m_f):
    x_pf = sqrt(n_i)*wfxn_HG(x,y,n_i-1,m_i,w0)*wfxn_HG(x,y,n_f,m_f,w0) - sqrt(n_i+1)*wfxn_HG(x,y,n_i+1,m_i,w0)*wfxn_HG(x,y,n_f,m_f,w0) - sqrt(n_f)*wfxn_HG(x,y,n_i,m_i,w0)*wfxn_HG(x,y,n_f-1,m_f,w0) + sqrt(n_f+1)*wfxn_HG(x,y,n_i,m_i,w0)*wfxn_HG(x,y,n_f+1,m_f,w0)
    y_pf = sqrt(m_i)*wfxn_HG(x,y,n_i,m_i-1,w0)*wfxn_HG(x,y,n_f,m_f,w0) - sqrt(m_i+1)*wfxn_HG(x,y,n_i,m_i+1,w0)*wfxn_HG(x,y,n_f,m_f,w0)- sqrt(m_f)*wfxn_HG(x,y,n_i,m_i,w0)*wfxn_HG(x,y,n_f,m_f-1,w0) + sqrt(m_f+1)*wfxn_HG(x,y,n_i,m_i,w0)*wfxn_HG(x,y,n_f,m_f+1,w0) 
    pf = -complex(0,1)/w0
    return -pf*array([x_pf, y_pf])

def phase_J_HG(x,y,w0,n_i,m_i,n_f,m_f):
    vec = array([1,0])
    pf = vec[0]*J_HG(x,y,w0,n_i,m_i,n_f,m_f)[0] + vec[1]*J_HG(x,y,w0,n_i,m_i,n_f,m_f)[1]
    return arctan2( imag(pf), real(pf) )
#########################################
def J_pw_single(x,y,kxi,kyi,kxf,kyf):
    qx = kxi - kxf
    qy = kyi - kyf
    qqx = kxi + kxf
    qqy = kyi + kyf 
    pf = exp(complex(0,1)*(qx*x + qy*y))
    return -(1/2)*array([pf*qqx, pf*qqy])

def phase_J_pw_single(x,y, kxi,kyi,kxf,kyf):
    vec = array([1,0])
    pf = vec[0]*J_pw_single(x,y,kxi,kyi,kxf,kyf)[0] + vec[1]*J_pw_single(x,y, kxi,kyi,kxf,kyf)[1]
    return arctan2( imag(pf), real(pf) ) 
#########################################
def J_pw_super(x,y,kxi,kyi,phi):
    pf_x = exp(complex(0,1)*kxi*x)
    pf_y = exp(complex(0,1)*kyi*y)
    return -(1/(2*sqrt(2)))*array([pf_x*kxi, pf_y*kyi*exp(complex(0,1)*phi)])

def phase_J_pw_super(x,y,kxi,kyi,phi):
    pf = phi_vec(x,y)[0]*J_pw_super(x,y,kxi,kyi,phi)[0] + phi_vec(x,y)[1]*J_pw_super(x,y,kxi,kyi,phi)[1]
    return arctan2( imag(pf), real(pf) )
#########################################
def J_HGpinhole(x,y,w0,n_i,m_i,n_f,m_f):
    x_pf = sqrt(n_i)*wfxn_HG(x,y,n_i-1,m_i,w0) - sqrt(n_i+1)*wfxn_HG(x,y,n_i+1,m_i,w0)
    y_pf = sqrt(m_i)*wfxn_HG(x,y,n_i,m_i-1,w0) - sqrt(m_i+1)*wfxn_HG(x,y,n_i,m_i+1,w0)
    pf = complex(0,1)/w0
    return pf*array([x_pf, y_pf])

def phase_J_HGpinhole(x,y,w0,n_i,m_i,n_f,m_f):
    vec = array([0,1])
    pf = vec[0]*J_HGpinhole(x,y,w0,n_i,m_i,n_f,m_f)[0] + vec[1]*J_HGpinhole(x,y,w0,n_i,m_i,n_f,m_f)[1]
    return arctan2( imag(pf), real(pf) )
#########################################
def J_LG10pinhole(x,y,w0):
    x_pf = wfxn_HG(x,y,0,0,w0) - sqrt(2)*wfxn_HG(x,y,2,0,w0) - complex(0,1)*wfxn_HG(x,y,1,1,w0)
    y_pf = -wfxn_HG(x,y,1,1,w0) + complex(0,1)*wfxn_HG(x,y,0,0,w0) - complex(0,1)*wfxn_HG(x,y,0,2,w0)
    pf = 1/(w0*sqrt(2))
    return pf*array([x_pf, y_pf])

def phase_J_LG10pinhole(x,y,w0):
    pf = phi_vec(x,y)[0]*J_LG10pinhole(x,y,w0)[0] + phi_vec(x,y)[1]*J_LG10pinhole(x,y,w0)[1]
    return arctan2( imag(pf), real(pf) )
#########################################
#########################################################

## Set up cartesian space
x_min = -1.0*10**(-9)
y_min = -1.0*10**(-9)
x_max = 1.0*10**(-9)
y_max = 1.0*10**(-9)
x_num = 1000
y_num = 1000

## initialize 2D evaluation grid (xy):
x_vals = linspace(x_min, x_max, num=x_num)
y_vals = linspace(y_min, y_max, num=y_num)
X, Y = meshgrid(x_vals, y_vals)

## Quantum numbers and other
w0 = 0.75*10**(-9) 
lmbda = 2*x_max
# aa = 1/0.05
aa = 1

## pw
kx_i = 2*pi/(lmbda*aa)
kx_f =0
ky_i = 0
ky_f = 0


## pw super
kx_isup = 2*pi/(lmbda*aa)
ky_isup = 2*pi/(lmbda*aa)
phii = 1*pi/2

## Laguerre-Gauss and Bessel
l_i = 2
l_f = 0
kp_i = 2*pi/(lmbda*aa)
kp_f = 2*pi/(lmbda*aa)
p_i = 0
p_f = 0

## Hermite-Gauss 
n_i = 1
m_i = 0
n_f = 1
m_f = 0

## Details for plots
## looks better for mainly transverse vectors
a_nx = 70
a_ny = 70
## looks better for mainly longitudinal vectors
# a_nx = 145
# a_ny = 140
## 
hwid = 4.0
hlen = 4.5
len_scl = 1
wid_scl = 0.010
unts = 'xy'
piv_loc = 'tail'
arw_clr = 'black'
clr = cm.gray
phs_clr = cm.hsv

## Define our functions to be plotted
Z_LG = sqrt(abs(J_LG(X,Y,w0,p_i,p_f,l_i,l_f)[0])**2 + abs(J_LG(X,Y,w0,p_i,p_f,l_i,l_f)[1])**2)
Z_LG_x = J_LG(X[::a_nx, ::a_ny], Y[::a_nx, ::a_ny],w0,p_i,p_f,l_i,l_f)[0]/abs(max(Z_LG)) 
Z_LG_y = J_LG(X[::a_nx, ::a_ny], Y[::a_nx, ::a_ny],w0,p_i,p_f,l_i,l_f)[1]/abs(max(Z_LG)) 

## Plot J_LG
fig, ax = plt.subplots(figsize=(7, 7)) 
alpha_arr = 1-Z_LG/Z_LG.max()  
mask = zeros((x_num, y_num)) 
imshow_kwargs = {'vmax': 3.14, 'vmin': -3.14, 'cmap': phs_clr,'aspect': 1,'interpolation': 'None'}
ax.imshow(phase_J_LG(X,Y,w0,p_i,p_f,l_i,l_f), **imshow_kwargs, extent=[X.min(), X.max(), Y.min(), Y.max()])
imshow_kwargs1 = {'vmax': Z_LG.max(), 'vmin': Z_LG.min(),'cmap': clr,'aspect': 1,'interpolation': 'none'} 
ax.imshow(mask, **imshow_kwargs1, extent=[X.min(), X.max(), Y.min(), Y.max()], alpha=alpha_arr) 
plt.quiver(X[::a_nx, ::a_ny], Y[::a_nx, ::a_ny], imag(Z_LG_x), imag(Z_LG_y),color= arw_clr, pivot = piv_loc, headwidth = hwid, headlength = hlen, scale_units = 'inches', scale = len_scl, width = wid_scl) 
plt.gca().set_aspect("equal") 
plt.xlim((X.min(),X.max())) 
plt.ylim((Y.min(),Y.max()))
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
