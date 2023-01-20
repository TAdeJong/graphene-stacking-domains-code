# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

# # Frenkel-Kontorova Phase Diagram
#
# Some plots using GSFE and the model as described by Lebedeva and Popov of the different phases.

# +
import numpy as np
import matplotlib.pyplot as plt

#import latticegen
import os
from matplotlib import ticker
# %matplotlib inline

from scipy.constants import value

stackingcolors = dict(AB='C3', BA='C3', SP='C4', AA='C5')

# +
a = 0.246*1e-9
l = a / np.sqrt(3) #0.1430*1e-9 # nm, bondlength graphene 
nu = 0.174
k = 331 # J/m^2, elastic constant under uniaxial stress

uc_area = a**2*0.5*np.sqrt(3)
Vmax = 1.61e-3*value('electron volt') / uc_area*2

def GSFE(ux, uy, k0=2*np.pi/(3*0.142), Vmax=1.61):
    """
    Approximate potential energy surface for interlayer interaction by first Fourier harmonics
    
    Given by equation 1 in https://doi.org/10.1103/PhysRevLett.124.116101
    
    Parameters
    ----------
    ux : np.array
        relative displacement in armchair dir
    uy : np.array
        relative displacement in zigzag dir
    k0 : float
        wavelength determined from bond length of graphene
    Vmax : float
        barrier to relative in-plane motion of the layers in meV/atom
    """
    res = np.cos(2*k0*ux - 2*np.pi/3)
    res -= 2*np.cos(k0*ux-np.pi/3)*np.cos(k0*uy*np.sqrt(3))
    return 2*Vmax*(3/2 + res)

def W0(Vmax=Vmax):
    return np.sqrt(k*l*l*Vmax/(1-nu*nu)) * (3*np.sqrt(3)/np.pi -1)

def epsilonc0(Vmax=Vmax):
    return (1-nu)*W0(Vmax) / (k*l)

def epsilonc0s(Vmax=Vmax):
    return epsilonc0(Vmax=Vmax)*np.sqrt((7-nu)/6)

def epsilonc1(Vmax=Vmax):
    ratio = 1.238 # (np.sqrt((7+4*nu)*(7-nu)/6)-2) / (np.sqrt(7+4*nu) - 2)
    return epsilonc0(Vmax)*ratio

def epsilonc2(Vmax=Vmax):
    return 1.5*np.sqrt(np.sqrt(3)*(7-nu)*Vmax/(2*np.pi*k))

def epsilon2triperiod(epsilon):
    """Naive period in triangles from epsilon
    c.f.  the inverse operation:
    eps = 0.246 / (NMPERPIXEL/wavelengths[i])
    eps = a * pks
    """
    return l/epsilon

def triperiod2epsilon(L):
    """Naive epsilon from triangles period
    c.f.  the  operation:
    eps = 0.246 / (NMPERPIXEL/wavelengths[i])
        = 0.246 * wavelengths[i] / NMPERPIXEL
    eps = a * pks / NMPERPIXEL
    """
    return l/L

def L0(epsilon, Vmax=Vmax):
    """network period for the triangular phase
    
    Length of the triangle sides, i.e.
    the linespacing is L0() / sqrt(3).
    As defined by Lebedeva and Popov in:
    https://doi.org/10.1103/PhysRevLett.124.116101
    """
    factor = np.sqrt(3) * (7 + 4*nu) / (4 * (1+nu))
    return factor * l / (epsilon - epsilonc0(Vmax=Vmax))

def L0_to_eps(L, Vmax=Vmax):
    factor = np.sqrt(3) * (7 + 4*nu) / (4 * (1+nu))
    return epsilonc0(Vmax=Vmax) + factor * l/L

def L0s(epsilon, Vmax=Vmax):
    """stripe period
    
    For the periodicity of the full lattice, i.e.
    linespacing is L0s() / 2
    As defined by Lebedeva and Popov in eq. 17 in:
    https://doi.org/10.1103/PhysRevLett.124.116101
    """
    factor = np.sqrt(3)/(2*(1+nu))
    epsilonc0s = epsilonc0(Vmax=Vmax) * np.sqrt((7-nu) / 6)
    return factor * l / (epsilon-epsilonc0s)

def L0s_to_eps(L, Vmax=Vmax):
    """Inverse of L0s"""
    factor = np.sqrt(3)/(2*(1+nu))
    epsilonc0s = epsilonc0(Vmax=Vmax) * np.sqrt((7-nu) / 6)
    return epsilonc0s + factor * l/L

def L0s2(epsilon, Vmax=Vmax):
    """stripe period large eps
    
    i.e. close to the transition to triangular incommensurate
    
    For the periodicity of the full lattice, i.e.
    linespacing is L0s() / 2
    As defined by Lebedeva and Popov in eq. 21 in:
    https://doi.org/10.1103/PhysRevLett.124.116101
    """
    factor = (7-nu) / (2*np.sqrt(3)*(1+nu))
    return factor * l/epsilon

def L0s2_to_eps(L, Vmax=Vmax):
    """Inverse of L0s2"""
    factor = (7-nu) / (2*np.sqrt(3)*(1+nu))
    return factor * l/L


# -

# pks = k0-k1 = 1/a - 1/(a*(1+e))
# a*pks = 1 - 1/(1+epsilon) = (epsilon)/(1+epsilon)

Vmax = 1.61e-3*value('electron volt') / uc_area*2
Vmax

for e in [epsilonc0(Vmax), epsilonc1(Vmax), epsilonc2(Vmax)]:
    print(e, epsilon2triperiod(e)*1e9*2/np.sqrt(3))

epsilonc0()

epsilon2triperiod(0.00299)*1e9

epsilon2triperiod(0.00368)*1e9 *2/2*np.sqrt(3)

eps = np.linspace(0,1.3/100, 5000)

eps[0]

# +
fig, ax = plt.subplots()

trieps = eps[np.logical_and(eps > epsilonc0(),
                            eps < epsilonc1()
                           )]
ax.semilogy(trieps*100,
             L0(trieps)*1e6, #/np.sqrt(3),
            label='triangular')
ax.semilogy(eps[eps>epsilonc1()]*100,
             L0s(eps[eps>epsilonc1()])*1e6,#/2,
             label='stripe')

#ax.semilogy(eps[eps>epsilonc1()]*100,
#             L0s2(eps[eps>epsilonc1()])*1e6, #/2,
#            label='stripe2')

ax.semilogy(eps*100, epsilon2triperiod(eps)*1e6, label='naive')
for X in [0.6, 1.4, 1.8]:
    trieps = eps[np.logical_and(eps > epsilonc0(Vmax*X), eps < epsilonc1(Vmax*X))]
    ax.semilogy(trieps*100,
                 L0(trieps, Vmax=Vmax*X)*1e6, #/np.sqrt(3),
                alpha=0.3, color='C0')
    ax.semilogy(eps[eps>epsilonc1(Vmax*X)]*100,
                 L0s(eps[eps>epsilonc1(Vmax*X)], Vmax=Vmax*X)*1e6, #/2,
                color='C1', alpha=0.3)
    #for e in [epsilonc0(Vmax*X), epsilonc1(Vmax*X), epsilonc2(Vmax*X)]:
    #    ax.axvline(e*100, alpha=0.1, color='black')

plt.margins(x=0.01)
for e in [epsilonc0(Vmax), epsilonc1(Vmax), epsilonc2(Vmax)]:
    ax.axvline(e*100, alpha=0.3, color='black')
ax.legend()
ax.set_ylim(1e-2,1e1)
ax.set_xlabel('ϵ (%)')
ax.set_ylabel('L (μm)')
#ax.set_xscale('log')
#ax.set_xlim(1e-1,None)

# +
fig, ax = plt.subplots(figsize=[6, 3.5], constrained_layout=True)

Vnew = Vmax 

eps = np.linspace(0, epsilonc2(Vmax=Vnew)+1e-3, 5000)

tri = np.logical_and(eps > epsilonc0(Vmax=Vnew),
                            eps < epsilonc1(Vmax=Vnew)
                           )
ax.semilogy(eps[tri]*100,
             L0(eps[tri],Vmax=Vnew)/np.sqrt(3) * 1e6,
            label='triangular', color='C0')


stripe = np.logical_and(eps > epsilonc1(Vmax=Vnew),
                        eps < epsilonc2(Vmax=Vnew))
stripeps = eps[stripe]
ax.semilogy(stripeps*100,
             L0(stripeps, Vmax=Vnew)/np.sqrt(3) * 1e6,
            linestyle=':', color='C0')

select = np.logical_and(eps > epsilonc0s(Vmax=Vnew),
                            eps < epsilonc1(Vmax=Vnew))
ax.semilogy(eps[select]*100,
             L0s(eps[select], Vmax=Vnew)/2 * 1e6,
            linestyle=':', color='C1')


scale = (stripeps - epsilonc1(Vmax=Vnew)) / (epsilonc2(Vmax=Vnew) - epsilonc1(Vmax=Vnew))
vals = (1-scale) * L0s(stripeps, Vmax=Vnew) + scale * L0s2(stripeps, Vmax=Vnew)
ax.semilogy(stripeps*100,
            vals/2 * 1e6, label='stripe', color='C1')
#ax.semilogy(eps[eps>epsilonc1()]*100,
#             L0s(eps[eps>epsilonc1()])/2 * 1e6,
#            label='stripe', color='C1')



ax.semilogy(eps*100, 
            np.where(stripe, epsilon2triperiod(eps)/2,
                     epsilon2triperiod(eps))* 1e6,
            label='non-interacting', color='C2')

ax.semilogy(eps[tri]*100,
             epsilon2triperiod(eps[tri])/2 * 1e6,
            linestyle=':', color='C2')
ax.semilogy(eps[stripe]*100,
             epsilon2triperiod(eps[stripe]) * 1e6,
            linestyle=':', color='C2')

#ax.semilogy(eps[stripe]*100, 
#            epsilon2triperiod(eps[stripe]) * 1e6/2,
#            label='naive stripe')


plt.margins(x=0.01)
for e in [epsilonc0(Vnew), epsilonc1(Vnew), epsilonc2(Vnew)]:
    ax.axvline(e*100, alpha=0.3, color='black')
ax.legend()
ax.set_ylim(None,1e1)
ax.set_xlabel('ϵ (%)')
ax.set_ylabel('1/k (μm)')
ax.set_title('a', fontweight='bold', loc='left')
ax.annotate(r'$\epsilon_{c0}$', (epsilonc0(Vnew)*100, 4e-3), xytext=(-5,5),
            textcoords='offset points', ha='right', size='large')
ax.annotate(r'$\epsilon_{c1}$', (epsilonc1(Vnew)*100, 4e-3), xytext=(5,5),
            textcoords='offset points', ha='left', size='large')
ax.annotate(r'$\epsilon_{c2}$', (epsilonc2(Vnew)*100, 4e-3), xytext=(5,5),
            textcoords='offset points', ha='left', size='large')


# +
fig, ax = plt.subplots(figsize=[5, 3], constrained_layout=True)

Vnew = Vmax 

eps = np.linspace(0, epsilonc2(Vmax=Vnew)+1e-3, 5000)

tri = np.logical_and(eps > epsilonc0(Vmax=Vnew),
                            eps < epsilonc1(Vmax=Vnew)
                           )
ax.semilogy(eps[tri]*100,
             L0(eps[tri], Vmax=Vnew)/np.sqrt(3) * 1e6,
            label='triangular', color='C0')


stripe = np.logical_and(eps > epsilonc1(Vmax=Vnew),
                        eps < epsilonc2(Vmax=Vnew))
stripeps = eps[stripe]
ax.semilogy(stripeps*100,
             L0(stripeps, Vmax=Vnew)/np.sqrt(3) * 1e6,
            linestyle=':', color='C0')

select = np.logical_and(eps > epsilonc0s(Vmax=Vnew),
                            eps < epsilonc1(Vmax=Vnew))
ax.semilogy(eps[select]*100,
             L0s(eps[select], Vmax=Vnew)/2 * 1e6,
            linestyle=':', color='C1')


scale = (stripeps - epsilonc1(Vmax=Vnew)) / (epsilonc2(Vmax=Vnew) - epsilonc1(Vmax=Vnew))
vals = (1-scale) * L0s(stripeps, Vmax=Vnew) + scale * L0s2(stripeps, Vmax=Vnew)
ax.semilogy(stripeps*100,
            vals/2 * 1e6, label='stripe', color='C1')


ax.semilogy(eps*100, 
            np.where(stripe, epsilon2triperiod(eps)/2,
                     epsilon2triperiod(eps))* 1e6,
            label='non-interacting', color='C2')

ax.semilogy(eps[tri]*100,
             epsilon2triperiod(eps[tri])/2 * 1e6,
            linestyle=':', color='C2')
ax.semilogy(eps[stripe]*100,
             epsilon2triperiod(eps[stripe]) * 1e6,
            linestyle=':', color='C2')



plt.margins(x=0.01)
for e in [epsilonc0(Vnew), epsilonc1(Vnew), epsilonc2(Vnew)]:
    ax.axvline(e*100, alpha=0.3, color='black')
ax.legend()
ax.set_ylim(None,1e1)
ax.set_xlabel('ϵ (%)')
ax.set_ylabel('1/k (μm)')
ax.set_title('(a)',
             #fontweight='bold',
             loc='left')
ax.annotate(r'$\epsilon_{c0}$', (epsilonc0(Vnew)*100, 4e-3), xytext=(-5,5),
            textcoords='offset points', ha='right', size='large')
ax.annotate(r'$\epsilon_{c1}$', (epsilonc1(Vnew)*100, 4e-3), xytext=(5,5),
            textcoords='offset points', ha='left', size='large')
ax.annotate(r'$\epsilon_{c2}$', (epsilonc2(Vnew)*100, 4e-3), xytext=(5,5),
            textcoords='offset points', ha='left', size='large')
plt.savefig(os.path.join('plots', 'lebedeva-phases.pdf'))

# +
fig, ax = plt.subplots()
trieps = eps[np.logical_and(eps > epsilonc0(), eps < epsilonc1())]
ax.plot(trieps*100,
             1e-9 / (L0(trieps)/np.sqrt(3)),
        label='triangular'
       )
stripeps = eps[stripe]
merge = True
if not merge:
    ax.plot(stripeps*100,
                 1e-9/(L0s(stripeps))*2, label='stripe')
    ax.plot(stripeps*100,
                 1e-9/(L0s2(stripeps))*2, label='stripe2')
else:
    scale = (stripeps-epsilonc1()) / (epsilonc2() - epsilonc1())
    vals = (1-scale)*L0s(stripeps) + scale*L0s2(stripeps)
    ax.plot(stripeps*100, 1e-9/vals*2, label='stripe')
    #vals = (1-scale)*2e-9/L0s(stripeps) + scale*2e-9/L0s2(stripeps)
    #ax.plot(stripeps*100, vals, label='stripecombi2')
plt.margins(x=0.001)

#ax.plot(eps*100, 1e-9 / epsilon2triperiod(eps))
ax.plot(eps*100, 
        1e-9/np.where(stripe, epsilon2triperiod(eps)/2,
                     epsilon2triperiod(eps)),
        label='non-interact')
for e in [epsilonc0(Vmax), epsilonc1(Vmax), epsilonc2(Vmax)]:
    ax.axvline(e*100, alpha=0.3, color='black')
ax.set_xlabel('ϵ (%)')
ax.set_ylabel('k (1/nm)')
ax.legend()
ax.set_ylim(0,None)
# -

stripeps, epsilonc1(), epsilonc2()

scale = (stripeps-epsilonc1()) / (epsilonc2() - epsilonc1())
scale
