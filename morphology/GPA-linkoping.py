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
#     display_name: Python [conda env:pyL5cupy]
#     language: python
#     name: conda-env-pyL5cupy-py
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pyGPA.geometric_phase_analysis as GPA
from pyGPA.imagetools import gauss_homogenize2, fftplot
import os
from skimage.io import imread
from moisan2011 import per

from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, disk
from pyGPA.imagetools import to_KovesiRGB, indicate_k
import colorcet  # noqa: F401
import pyGPA.cuGPA as cuGPA
from matplotlib_scalebar.scalebar import ScaleBar


# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# +
folder = '/mnt/storage-linux/domainsdata/2019-08-12-G1193/20190812_214011_2.3um_529.8_sweep-STAGE_X-STAGE_YBF_fullCP'
name = 'stitch_v10_2021-12-01_1400_sobel_2_bw_200.tif'

image = imread(os.path.join(folder, name)).T.astype(float)
NMPERPIXEL = 1.36
#image = image / image.max()
# -

1152816 * 16.0 * 1352, (image > 0).sum(), 1152816 * 16.0 * 0.25 * 1352/(image > 0).sum()

16.0*0.25 * 8

plt.hist(image[image > 0].ravel(), bins=500)
plt.ylim(0, 3e5)

f, ax = plt.subplots(ncols=2)
ax[0].imshow((image > 0).T)
ax[1].imshow(image.T)

# Small image to extract pks
crop1 = image[5700:6500, 1600:2400]
plt.imshow(crop1.T)
plt.colorbar()

# Larger crop to perform GPA and statistics on
crop2 = image[5120:7620, 1800:4300]
plt.imshow(crop2.T)

smooth1 = gauss_homogenize2(crop1, mask=np.ones_like(crop1), sigma=80)
plt.imshow(smooth1 > 0.8)
smooth1 = np.clip(smooth1, 0.8, None)  # Clip out adsorbates

p, s = per(smooth1 - smooth1.mean(), inverse_dft=False)
fftim = np.fft.fftshift(np.abs(p))
pks, _ = GPA.extract_primary_ks(smooth1 - smooth1.mean(), plot=True, sigma=2, pix_norm_range=(10, 70), DoG=False)
pks2, _ = GPA.extract_primary_ks(smooth1 - smooth1.mean(), plot=True, sigma=2, pix_norm_range=(30, 70), DoG=False)

pks2[2] = pks2[0] - pks2[1]

f, axs = plt.subplots(ncols=2, figsize=[9, 4.5])
im = fftplot(fftim.T,  # ndi.filters.gaussian_filter(fftim, sigma=2),
             d=NMPERPIXEL,
             pcolormesh=False, ax=axs[1], origin='upper', vmax=2000, cmap='cet_fire_r')
fftr = 0.09
axs[1].set_xlim(-fftr, fftr)
axs[1].set_ylim(fftr, -fftr)
axs[1].set_xlabel('$k_x$ (periods / nm)')
axs[1].set_ylabel('$k_y$ (periods / nm)')
#plt.colorbar(im, ax=axs[0])
for i, ks in enumerate([pks, pks2[:2]]):
    axs[1].scatter(*np.concatenate([-ks, ks]).T[::-1]/NMPERPIXEL, marker='o',
                   s=400, facecolors='none', linewidths=2, edgecolors=f'C{i}')
axs[1].scatter(*np.stack([-pks2[2], pks2[2]]).T[::-1]/NMPERPIXEL, marker='o',
               s=400, facecolors='none', linewidths=2, edgecolors=f'C{i}', linestyle='--')
im = axs[0].imshow(smooth1 - smooth1.mean(), cmap='gray', origin='upper')
im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
axs[0].set_xlabel('x (nm)')
for i in range(2):
    axs[i].set_title('ab'[i], fontweight='bold', loc='left')
plt.tight_layout()
#plt.savefig(os.path.join('plots', 'GPAstripe-extract.pdf'))

from matplotlib import ticker

f, axs = plt.subplots(ncols=2, figsize=[5, 2.65], constrained_layout=True)
im = fftplot(fftim.T,  # ndi.filters.gaussian_filter(fftim, sigma=2),
             d=NMPERPIXEL,
             pcolormesh=False, ax=axs[1], origin='upper', vmax=2000, cmap='cet_fire_r')
fftr = 0.09
axs[1].set_xlim(-fftr, fftr)
axs[1].set_ylim(fftr, -fftr)
axs[1].set_xlabel('$k_x$ (periods / nm)')
axs[1].set_ylabel('$k_y$ (periods / nm)')
#plt.colorbar(im, ax=axs[0])
for i, ks in enumerate([pks, pks2[:2]]):
    axs[1].scatter(*np.concatenate([-ks, ks]).T[::-1]/NMPERPIXEL, marker='o',
                   s=150, facecolors='none', linewidths=1, edgecolors=f'C{i}')
axs[1].scatter(*np.stack([-pks2[2], pks2[2]]).T[::-1]/NMPERPIXEL, marker='o',
               s=150, facecolors='none', linewidths=1, edgecolors=f'C{i}', linestyle='--')
im = axs[0].imshow(smooth1 - smooth1.mean(), cmap='gray', origin='upper')
im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
axs[0].set_xlabel('x (nm)')
for i in range(2):
    axs[i].set_title("({})".format('bc'[i]),
                     #fontweight='bold',
                     loc='left')
#plt.tight_layout()
axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
axs[1].yaxis.set_ticklabels([])
plt.savefig(os.path.join('plots', 'GPAstripe-extract2.pdf'))

smooth = gauss_homogenize2(crop2, mask=np.ones_like(crop2), sigma=70)

cropped = smooth
f, axs = plt.subplots(ncols=2, figsize=[12, 5])
axs[0].imshow(cropped.T, vmin=0.8, vmax=1.18)
axs[1].hist(cropped.ravel(), bins=100);

clipped = np.clip(cropped, 0.8, None)

1/np.linalg.norm(pks, axis=1).mean(), 1/np.linalg.norm(pks2, axis=1).mean()

pks2scale = 1.1
u2, gs2 = GPA.extract_displacement_field(clipped, pks2*pks2scale,
                                         ksteps=6,  # 4,
                                         kwscale=4, sigma=25,
                                         return_gs=True, wfr_func=cuGPA.wfr2_grad_opt)

# +


magnitudes2 = np.stack([np.abs(g['lockin']) for g in gs2])
angles2 = np.stack([np.angle(g['lockin']) for g in gs2])
grads2 = np.stack([g['grad'] for g in gs2])
gradmags2 = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs2])

f, axs = plt.subplots(ncols=3, nrows=2, figsize=[18, 12])
for i, ax in enumerate(axs.flat[:3]):
    ax.imshow(angles2[i].T, cmap='twilight', interpolation='nearest')
    ax.imshow(clipped.T, cmap='gray', alpha=0.5, vmax=1.18)
    indicate_k(pks2, i, ax=ax)
axs[1, 0].imshow(to_KovesiRGB((magnitudes2/np.quantile(magnitudes2[0], 0.99)  # , axis=(1,2), keepdims=True)
                               ).T))
axs[1, 1].imshow(clipped.T, cmap='gray')
axs[1, 2].imshow(to_KovesiRGB((gradmags2 < 0.06).astype(float).T))
# -

plt.figure(figsize=[6, 6])
plt.imshow(to_KovesiRGB((magnitudes2 == np.max(magnitudes2, axis=0, keepdims=True)).astype(float).T
                        ))
plt.imshow(clipped.T, cmap='gray', vmax=np.quantile(clipped, 0.98), alpha=0.5)

u, gs = GPA.extract_displacement_field(clipped, pks, ksteps=3, kwscale=4, sigma=25,
                                       return_gs=True, wfr_func=cuGPA.wfr2_grad_opt)
magnitudes = np.stack([np.abs(g['lockin']) for g in gs])
angles = np.stack([np.angle(g['lockin']) for g in gs])
grads = np.stack([g['grad'] for g in gs])
gradmags = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs])

fftr2 = 0.07

# +
f, axs = plt.subplots(ncols=4, nrows=2, figsize=[9, 4.5], sharex=True, sharey=True, constrained_layout=True)

#magnitudes = np.stack([np.abs(g['lockin']) for g in gs])
#angles = np.stack([np.angle(g['lockin']) for g in gs])
#grads = np.stack([g['grad'] for g in gs])
#gradmags = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs])
for i, ax in enumerate(axs[0][:3]):
    ax.imshow(angles[i], cmap='twilight', interpolation='nearest', vmax=np.pi, vmin=-np.pi)
    ax.imshow(clipped, cmap='gray', alpha=0.3)
    for axis in ['left', 'right', 'top', 'bottom']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color(to_KovesiRGB(np.eye(3))[i])
    inax = indicate_k(pks[:, ::-1], i, ax=ax, s=3, colors=[to_KovesiRGB(np.eye(3))[i], 'black'])
    indicate_k(pks2[:, ::-1], i=[], ax=inax, inset=False, s=3, colors=[to_KovesiRGB(np.eye(3))[i], 'black'])
    inax.set_xlim(-fftr2, fftr2)
    inax.set_ylim(-fftr2, fftr2)
    inax.set_facecolor('white')
    inax.patch.set_alpha(0.5)
for i, ax in enumerate(axs[1][:3]):
    angleim = ax.imshow(angles2[i], cmap='twilight', interpolation='nearest', vmax=np.pi, vmin=-np.pi)
    ax.imshow(clipped, cmap='gray', alpha=0.3)
    for axis in ['left', 'right', 'top', 'bottom']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color(to_KovesiRGB(np.eye(3))[i])
    inax = indicate_k(pks2[:, ::-1], i, ax=ax, s=3, colors=[to_KovesiRGB(np.eye(3))[i], 'black'])
    indicate_k(pks[:, ::-1], i=[], ax=inax, inset=False, s=3, colors=[to_KovesiRGB(np.eye(3))[i], 'black'])
    inax.set_xlim(-fftr2, fftr2)
    inax.set_ylim(-fftr2, fftr2)
    inax.set_facecolor('white')
    inax.patch.set_alpha(0.5)

axs[0, 3].imshow(np.swapaxes(to_KovesiRGB((magnitudes / np.quantile(magnitudes,
                                                                    0.99,
                                                                    # axis=(1,2),
                                                                    keepdims=True)
                                           ).T), 0, 1))
axs[1, 3].imshow(np.swapaxes(to_KovesiRGB((magnitudes2/np.quantile(magnitudes2,
                                                                   0.99,
                                                                   # axis=(1,2),
                                                                   keepdims=True)).T), 0, 1))

scalebar = ScaleBar(1e-9*NMPERPIXEL, "m", length_fraction=0.25,
                    location="lower right", box_alpha=0.3,
                    width_fraction=0.025
                    )
axs[1, 3].add_artist(scalebar)
cbar = plt.colorbar(angleim, ax=axs[:, :3], aspect=30, ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels(['-π', '0', 'π'])

for i, ax in enumerate(axs.flat):
    ax.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
    ax.set_title("({})".format('abcdefghijk'[i]),
                 #fontweight='bold',
                 loc='left')
# plt.tight_layout()
plt.savefig(os.path.join('plots', 'GPAphasessampleB.pdf'), dpi=600)

# +
loc = [1100, 350]
rplot = 200
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=[5, 5], constrained_layout=True)
ax[1, 0].imshow(angles[0, loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
                cmap='twilight',
                interpolation='none',
               vmin=-np.pi, vmax=np.pi
               )
ax[0, 0].imshow(clipped[loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
                cmap='gray', interpolation='none')
loc = [1350, 1600]
aim = ax[1, 1].imshow(angles2[0, loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
                      cmap='twilight', 
                      interpolation='none',
                      vmin=-np.pi, vmax=np.pi)
ax[0, 1].imshow(clipped[loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
                cmap='gray', interpolation='none')

inax = indicate_k(pks[:, ::-1], 0, ax=ax[1, 0], s=10, colors=[to_KovesiRGB(np.eye(3))[0], 'white'])
indicate_k(pks2[:, ::-1], i=[], ax=inax, inset=False, s=10, colors=[to_KovesiRGB(np.eye(3))[0], 'white'])
inax.set_xlim(-fftr2, fftr2)
inax.set_ylim(-fftr2, fftr2)

inax = indicate_k(pks[:, ::-1], [], ax=ax[1, 1], s=10, colors=[to_KovesiRGB(np.eye(3))[0], 'white'])
indicate_k(pks2[:, ::-1], i=0, ax=inax, inset=False, s=10, colors=[to_KovesiRGB(np.eye(3))[0], 'white'])
inax.set_xlim(-fftr2, fftr2)
inax.set_ylim(-fftr2, fftr2)

circ = plt.Circle([180, 220], radius=50, fc='none', ec='C0',
                  linewidth=3, alpha=0.7)
ax[0, 0].add_patch(circ)
circ = plt.Circle([150, 135], radius=50, fc='none', ec='C1',
                  linewidth=3, alpha=0.7)
ax[0, 1].add_patch(circ)
circ = plt.Circle([280, 300], radius=50, fc='none', ec='C2',
                  linewidth=3, alpha=0.7)
ax[0, 1].add_patch(circ)

circ = plt.Circle([180, 220], radius=50, fc='none', ec='C0',
                  linewidth=3, alpha=0.7)
ax[1, 0].add_patch(circ)
circ = plt.Circle([150, 135], radius=50, fc='none', ec='C1',
                  linewidth=3, alpha=0.7)
ax[1, 1].add_patch(circ)
circ = plt.Circle([280, 300], radius=50, fc='none', ec='C2',
                  linewidth=3, alpha=0.7)
ax[1, 1].add_patch(circ)

for i, a in enumerate(ax.flat):
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
    a.set_title("({})".format('abcd'[i]), loc='left',
                #fontweight='bold'
               )


scalebar = ScaleBar(1e-9*NMPERPIXEL, "m", length_fraction=0.25,
                    location="lower left", box_alpha=0.3,
                    width_fraction=0.025
                    )
ax[1, 0].add_artist(scalebar)


cbar = plt.colorbar(aim, ax=ax[1, :], aspect=20, ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels(['-π', '0', 'π'])

# plt.tight_layout()
plt.savefig(os.path.join('plots', 'GPAGSiCdisl.pdf'))
# -

threshes = np.array([threshold_otsu(m) for m in magnitudes])
thresholded = (magnitudes > np.mean(threshes[:, None, None]))
mask = np.any(thresholded[[0, 2]], axis=0)
eroded_mask = binary_erosion(mask, disk(15))

(~mask).sum(), (mask).sum()

# Percentages of different orientation stripe domains
for i in range(3):
    print((np.where(mask,
                    np.nan,
                    np.argmax(magnitudes2, axis=0)) == i).sum() / (~mask).sum())

plt.figure(figsize=[12, 9])
plt.imshow(np.where(mask.T[..., None],
                    np.nan,
                    to_KovesiRGB((magnitudes2 == np.max(magnitudes2, axis=0, keepdims=True)).astype(float).T
                                 )))
plt.imshow(clipped.T, cmap='gray', vmax=np.quantile(clipped, 0.98), alpha=0.5)
plt.colorbar()

threshes2 = np.array([threshold_otsu(m) for m in magnitudes2])
rgb = ['red', 'green', 'blue']
fig, axs = plt.subplots(ncols=2, figsize=[12, 6])
axs[0].hist([m.ravel() for m in magnitudes2], bins=200, histtype='barstacked', alpha=0.7, color=rgb)
for i, t in enumerate(threshes2):
    axs[0].axvline(t, color=rgb[i])
axs[1].imshow(to_KovesiRGB((magnitudes2 > threshes2[:, None, None]).astype(float).T))

thresholded = (magnitudes2 > threshes2[:, None, None])


wadvs = np.stack([np.moveaxis(gs2[i]['grad'], -1, 0) / 2 / np.pi + pks2[i, :, None, None]*pks2scale for i in range(3)])

allwavelengths = np.linalg.norm(wadvs, axis=1)
wavelengths = np.where(magnitudes2 == np.max(magnitudes2, axis=0, keepdims=True), allwavelengths, 0.)
wavelengths = np.nanmax(wavelengths, axis=0)

C0toC1 = LinearSegmentedColormap.from_list("C0toC1", ['C0', 'C1'])

# +
from scipy.constants import value
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

vals = [
    np.where(
        magnitudes2[i] == np.max(
            magnitudes2,
            axis=0,
            keepdims=True),
        allwavelengths[i],
        np.nan) for i in range(3)]
vals = [(0.246 / (2*NMPERPIXEL/val) * 100)[~np.isnan(val)] for val in vals] 
#vals = [ (L0s_to_eps(2*1e-9 / (val/NMPERPIXEL)) * 100)[~np.isnan(val)] for val in vals]
#vals = [ (2*1e-9 / (val/NMPERPIXEL))[~np.isnan(val)] for val in vals]

wadvs1 = np.stack([np.moveaxis(gs[i]['grad'], -1, 0) / (2*np.pi) + pks[i, :, None, None] for i in range(3)])
allwavelengths1 = np.linalg.norm(wadvs1, axis=1)
vals1 = [allwavelengths1[i][mask] for i in range(3)]
vals1 = [(0.246 / (NMPERPIXEL/val) * 100) for val in vals1] # use better function here for triangles
plt.hist(vals1, bins=1000, histtype='barstacked', color=list('rgb'))#, range=(0,3e-7))
#plt.xlabel('triangular domain period (m)')

weighted_vals = np.average(np.stack(vals1), weights=np.stack([m[mask] for m in magnitudes]), axis=0)

wl_im = np.where(mask, np.nan, 0.246 / (2*NMPERPIXEL/wavelengths) * 100)

Xim = np.full_like(mask, np.nan, dtype=float)
Xim[mask] = weighted_vals


# +
f, axs = plt.subplots(ncols=3, figsize=[9, 3.2],
                      gridspec_kw={'width_ratios': [1, 1, 0.6]},
                      constrained_layout=True)

axs[0].imshow(~mask, cmap=C0toC1)
axs[0].imshow(clipped, cmap='gray', alpha=0.5, vmax=np.quantile(clipped, 0.999))

axs[2].hist(np.concatenate(vals).ravel(), bins=200,
            alpha=0.5, range=(0.2, 0.7), color='C1',
            label='stripe',
            orientation='horizontal')
axs[2].hist(weighted_vals.ravel(), bins=200,
            alpha=0.5, range=(0.2, 0.7),
            color='C0', label='triangular',
            orientation='horizontal')
axs[2].legend()
axs[2].set_ylim(0.2, 0.7)
axs[2].set_ylabel(r'extracted $\tilde{\epsilon}$ (%)')
axs[2].axhline(0.3, label=r'$\epsilon_{c0}$', color='black', alpha=0.5)
leftx = axs[2].get_xlim()[1]
print(axs[2].get_xlim())
axs[2].annotate(r'$\epsilon_{c0}$', (leftx, 0.3),
                ha='right', va='bottom',
                xytext=(-1.5, 1.5), textcoords='offset points')
axs[2].axhline(0.37, color='black', alpha=0.5)
axs[2].annotate(r'$\epsilon_{c1}$', (leftx, 0.37),
                ha='right', va='bottom',
                xytext=(-1.5, 1.5), textcoords='offset points')
#axs[2].set_xlabel('number of pixels')
axs[0].set_title('(a)', loc='left')
axs[1].set_title('(b)', loc='left')
axs[2].set_title('(c)', loc='left')
im = axs[1].imshow(np.where(mask, Xim, wl_im), cmap='inferno', vmin=0.2, vmax=0.7)
plt.colorbar(im, ax=axs[:2], fraction=0.05, aspect=50)
axs[1].contour(mask, levels=[0.5], colors='white')
for i in range(2):
    axs[i].tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
axs[2].tick_params(
    bottom=False,      # ticks along the bottom edge are off
    left=True,
    labelbottom=False,
    labelleft=True)
scalebar = ScaleBar(1e-9*NMPERPIXEL, "m", length_fraction=0.15,
                    location="lower left", box_alpha=0.4,
                    width_fraction=0.02
                    )
axs[0].add_artist(scalebar)
plt.savefig(os.path.join('plots', 'Linkopingstrainhist.pdf'), dpi=600)

# +
plt.figure(figsize=[6, 4.5])
indication = (~mask).astype(float)
indication[Xim < 0.37] = -1
indication[Xim < 0.3] = -2
indication[Xim > 0.45] = 1
indication[wl_im > 0.45] = 4
indication[wl_im < 0.45] = 3
indication[wl_im < 0.37] = 2

#plt.imshow(clipped, cmap='gray', vmax=np.quantile(clipped,0.999))
plt.contourf(indication, origin='upper', cmap='tab20c',
             levels=np.arange(9)-2.5, vmax=18,
             # alpha=0.7
             )

plt.gca().set_aspect('equal')
plt.gca().tick_params(
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    labelbottom=False,
    labelleft=False)
scalebar = ScaleBar(1e-9*NMPERPIXEL, "m", length_fraction=0.15,
                    location="lower left", box_alpha=0.4,
                    width_fraction=0.02
                    )
plt.gca().add_artist(scalebar)
cbar = plt.colorbar(ticks=[-2, -1, 0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(['ϵ < 0.3', 'ϵ < 0.37', 'ϵ > 0.37', 'ϵ > 0.45', 'ϵ < 0.37', 'ϵ < 0.45', 'ϵ > 0.45'])
plt.tight_layout()
plt.savefig(os.path.join('plots', 'Linkopingstrainvaluelocs6x6.pdf'), dpi=600)
