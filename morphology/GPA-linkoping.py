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

# +
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import pyGPA.geometric_phase_analysis as GPA
from pyGPA.imagetools import gauss_homogenize2, fftplot
import os
from skimage.io import imread
from moisan2011 import per
import scipy.ndimage as ndi

from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, disk
from latticegen.transformations import rotate
from pyGPA.imagetools import to_KovesiRGB, indicate_k
import colorcet
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

# r = 1500
# image = image[r:-r, r:-r].astype(float)
plt.imshow(image.T)

1152816 * 16.0 * 1352, (image > 0).sum(), 1152816 * 16.0 * 0.25 * 1352/(image > 0).sum()

16.0*0.25 * 8

plt.hist(image[image > 0].ravel(), bins=500)
plt.ylim(0, 3e5)

f, ax = plt.subplots(ncols=2)
ax[0].imshow((image > 0).T)
ax[1].imshow((image > .50).T)

crop1 = image[5700:6500, 1600:2400]
plt.imshow(crop1.T)
plt.colorbar()

crop2 = image[5120:7620, 1800:4300]
plt.imshow(crop2.T)

plt.hist(crop1.ravel(), bins=100)

smooth1 = gauss_homogenize2(crop1, mask=np.ones_like(crop1), sigma=80)
plt.imshow(smooth1 > 0.8)
smooth1 = np.clip(smooth1, 0.8, None)

plt.imshow(smooth1.T)
plt.colorbar()

p, s = per(smooth1 - smooth1.mean(), inverse_dft=False)

fftim = np.fft.fftshift(np.abs(p))

f, ax = plt.subplots(figsize=[12, 8])
im = fftplot(ndi.filters.gaussian_filter(fftim, sigma=2),
             pcolormesh=False, ax=ax, interpolation='none')
fftr = 0.2
plt.xlim(-fftr, fftr)
plt.ylim(-fftr, fftr)
plt.colorbar(im, ax=ax)

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
plt.savefig(os.path.join('plots', 'GPAstripe-extract.pdf'))

smooth = gauss_homogenize2(crop2, mask=np.ones_like(crop2), sigma=70)

X = ndi.gaussian_filter(crop2, sigma=50)
cropped = smooth
f, axs = plt.subplots(ncols=3, figsize=[16, 5])
axs[0].imshow(X.T)
axs[1].imshow(cropped.T,  vmin=0.8, vmax=1.18)
axs[2].hist(cropped.ravel(), bins=100)

clipped = np.clip(cropped, 0.8, None)

p, s = per(clipped - clipped.mean(), inverse_dft=False)

fftim = np.fft.fftshift(np.abs(p))

1/np.linalg.norm(pks, axis=1).mean(), 1/np.linalg.norm(pks2, axis=1).mean()

pks2scale = 1.1
u2, gs2 = GPA.extract_displacement_field(clipped, pks2*pks2scale,
                                         ksteps=6,  # 4,
                                         kwscale=4, sigma=25,
                                         return_gs=True, wfr_func=cuGPA.wfr2_grad_opt)

# +
f, axs = plt.subplots(ncols=3, nrows=2, figsize=[18, 12])

magnitudes2 = np.stack([np.abs(g['lockin']) for g in gs2])
angles2 = np.stack([np.angle(g['lockin']) for g in gs2])
grads2 = np.stack([g['grad'] for g in gs2])
gradmags2 = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs2])
for i, ax in enumerate(axs.flat[:3]):
    ax.imshow(angles2[i].T, cmap='twilight', interpolation='nearest')
    ax.imshow(clipped.T, cmap='gray', alpha=0.5, vmax=1.18)
    indicate_k(pks2, i, ax=ax)
axs[1, 0].imshow(to_KovesiRGB((magnitudes2/np.quantile(magnitudes2[0], 0.99)  # , axis=(1,2), keepdims=True)
                               ).T))
axs[1, 1].imshow(clipped.T, cmap='gray')
axs[1, 2].imshow(to_KovesiRGB((gradmags2 < 0.06).astype(float).T))
# -

plt.hist([mag.ravel() for mag in gradmags2[::-1]], bins=100, histtype='barstacked')

plt.figure(figsize=[12, 12])
plt.imshow(to_KovesiRGB((magnitudes2 == np.max(magnitudes2, axis=0, keepdims=True)).astype(float).T
                        ))
plt.imshow(clipped.T, cmap='gray', vmax=np.quantile(clipped, 0.98), alpha=0.5)

u, gs = GPA.extract_displacement_field(clipped, pks, ksteps=3, kwscale=4, sigma=25,
                                       return_gs=True, wfr_func=cuGPA.wfr2_grad_opt)
magnitudes = np.stack([np.abs(g['lockin']) for g in gs])
angles = np.stack([np.angle(g['lockin']) for g in gs])
grads = np.stack([g['grad'] for g in gs])
gradmags = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs])

# +
f, axs = plt.subplots(ncols=3, nrows=2, figsize=[18, 12])


for i, ax in enumerate(axs.flat[:3]):
    ax.imshow(angles[i].T, cmap='twilight', interpolation='nearest')
    ax.imshow(clipped.T, cmap='gray', alpha=0.5)
    indicate_k(pks, i, ax=ax)
axs[1, 0].imshow(to_KovesiRGB((magnitudes/np.quantile(magnitudes, 0.99, axis=(1, 2), keepdims=True)).T))
axs[1, 1].imshow(clipped.T)
axs[1, 2].imshow(to_KovesiRGB((gradmags < 0.05).astype(float).T))
# -

fftr2 = 0.07

NMPERPIXEL, fftr2

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
    ax.set_title('abcdefghijk'[i], fontweight='bold', loc='left')
# plt.tight_layout()
plt.savefig(os.path.join('plots', 'GPAphasessampleB.pdf'), dpi=600)
# -

loc = [1100, 350]
rplot = 200
fig, ax = plt.subplots(ncols=2, figsize=[9, 4.5])
ax[0].imshow(angles[0, loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
             cmap='twilight', interpolation='nearest')
ax[0].imshow(clipped[loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
             cmap='gray', alpha=0.7)
loc = [1350, 1600]
ax[1].imshow(angles2[0, loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
             cmap='twilight', interpolation='nearest')
ax[1].imshow(clipped[loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
             cmap='gray', alpha=0.7)

5/9

# +
loc = [1100, 350]
rplot = 200
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=[5, 5], constrained_layout=True)
ax[1, 0].imshow(angles[0, loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
                cmap='twilight', interpolation='none')
ax[0, 0].imshow(clipped[loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
                cmap='gray', interpolation='none')
loc = [1350, 1600]
ax[1, 1].imshow(angles2[0, loc[0]-rplot:loc[0]+rplot, loc[1]-rplot:loc[1]+rplot],
                cmap='twilight', interpolation='none')
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

for i, a in enumerate(ax.flat):
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
    a.set_title('abcd'[i], loc='left', fontweight='bold')


scalebar = ScaleBar(1e-9*NMPERPIXEL, "m", length_fraction=0.25,
                    location="lower left", box_alpha=0.3,
                    width_fraction=0.025
                    )
ax[1, 0].add_artist(scalebar)
# plt.tight_layout()
plt.savefig(os.path.join('plots', 'GPAGSiCdisl.pdf'))
# -

threshes = np.array([threshold_otsu(m) for m in magnitudes])
thresholded = (magnitudes > np.mean(threshes[:, None, None]))

mask = np.any(thresholded[[0, 2]], axis=0)
eroded_mask = binary_erosion(mask, disk(15))

plt.imshow(np.where(mask, np.nan, np.argmax(magnitudes2, axis=0)).T)

mask.shape

(~mask).sum(), (mask).sum()

for i in range(3):
    print((np.where(mask, np.nan, np.argmax(magnitudes2, axis=0)) == i).sum()/(~mask).sum())

plt.figure(figsize=[12, 9])
plt.imshow(np.where(mask.T[..., None], np.nan, to_KovesiRGB((magnitudes2 == np.max(magnitudes2, axis=0, keepdims=True)).astype(float).T
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


wadvs = np.stack([np.moveaxis(gs2[i]['grad'], -1, 0) / 2/np.pi + pks2[i, :, None, None]*pks2scale for i in range(3)])

# +
# wadvs = []
# for i in range(3):
#     gphase = np.moveaxis(gs2[i]['grad'],-1,0) / 2/np.pi #still needed?
#     w = gphase + pks2[i,:, None, None]*pks2scale
#     wadvs.append(w)
# wadvs = np.stack(wadvs)

wxs = np.concatenate([wadvs[:, 0], -wadvs[:, 0]])
wys = np.concatenate([wadvs[:, 1], -wadvs[:, 1]])
#wxs = np.clip(wxs, -0.15,0.15)
#wys = np.clip(wys, -0.15,0.15)
p, _ = per(clipped-clipped.mean(), inverse_dft=False)
fftim = np.abs(np.fft.fftshift(p))
r = 0.08
fig, axs = plt.subplots(ncols=3, figsize=[26, 8], sharex=True, sharey=True)
axs[0].hist2d(wxs.ravel(),  # [np.stack([masks]*2).ravel()],
              wys.ravel(),  # [np.stack([masks]*2).ravel()],
              bins=500, cmap='cet_fire_r', vmax=2000,
              range=[(-r, r), (-r, r)])  # , ax=axs[0]);
axs[0].set_aspect('equal')
#plt.title(f'sigma={sigma}, kstep={kstep}')
pks2weights = magnitudes2 / np.quantile(magnitudes2[0], 0.99)
axs[1].hist2d(wxs.ravel()[np.stack([~eroded_mask]*6).ravel()],
              wys.ravel()[np.stack([~eroded_mask]*6).ravel()],
              weights=np.stack([pks2weights[:, ~eroded_mask]]*2).ravel(),
              bins=500, cmap='cet_fire_r', vmax=2000,
              range=[(-r, r), (-r, r)])  # , ax=axs[0]);
axs[1].set_aspect('equal')
fftplot(fftim, ax=axs[2], pcolormesh=False, vmax=np.quantile(fftim, 0.999),
        vmin=np.quantile(fftim, 0.01), cmap='cet_fire_r', interpolation='none', origin='lower')
#plt.title(f'sigma={sigma}, kstep={kstep}')
axs[0].set_ylim(-r, r)
axs[0].set_xlim(-r, r)
axs[1].scatter(*(pks2*1.1).T)
axs[1].scatter(*-(pks2*1.1).T, color='C0')
axs[1].scatter(*pks.T)
axs[1].scatter(*-pks.T, color='C1')
plt.tight_layout()
# -

(magnitudes2 == np.max(magnitudes2, axis=0, keepdims=True)).shape

allwavelengths = np.linalg.norm(wadvs, axis=1)
wavelengths = np.where(magnitudes2 == np.max(magnitudes2, axis=0, keepdims=True), allwavelengths, 0.)
wavelengths = np.nanmax(wavelengths, axis=0)

f, ax = plt.subplots(ncols=2, figsize=[16, 6.5], constrained_layout=True)
ax[0].imshow(np.where(mask, np.nan, NMPERPIXEL/oldwavelengths))
ax[1].imshow(np.where(mask, np.nan, NMPERPIXEL/wavelengths))

# +
NMPERPIXEL = 1.36
oldwavelengths = np.linalg.norm(wadvs[0], axis=0)
plangles = np.arctan2(*wadvs[0][::-1])

f, axs = plt.subplots(ncols=2, figsize=[16, 6.5], constrained_layout=True)
wl_im = np.where(mask, np.nan, NMPERPIXEL/wavelengths)
im = axs[0].imshow(wl_im.T,
                   vmax=np.nanquantile(wl_im, 0.999),
                   vmin=np.nanquantile(wl_im, 0.001), cmap='inferno')
plt.colorbar(im, ax=axs[0], label='wavelength (nm)')
extent = np.array(im.get_extent())*NMPERPIXEL/1e3
im.set_extent(extent)
im = axs[0].imshow(clipped.T, cmap='gray', alpha=0.5)
im.set_extent(extent)

angleplot = -np.where(mask, np.nan, np.rad2deg(plangles))
im = axs[1].imshow(angleplot.T,
                   vmax=np.nanquantile(angleplot, 0.99),
                   vmin=np.nanquantile(angleplot, 0.01))
im.set_extent(extent)
plt.colorbar(im, ax=axs[1], label='angle (degree)')
im = axs[1].imshow(clipped.T, cmap='gray', alpha=0.5)
im.set_extent(extent)

# +
f, axs = plt.subplots(ncols=2, figsize=[9, 3.8], constrained_layout=True, sharey=True)
wl_im = np.where(mask, np.nan, 0.246 / (2*NMPERPIXEL/wavelengths) * 100)
im = axs[0].imshow(wl_im,
                   vmax=np.nanquantile(wl_im, 0.9999),
                   vmin=np.nanquantile(wl_im, 0.01), cmap='inferno')
plt.colorbar(im, ax=axs[0], label='relative strain (%)', shrink=0.98, extend='min')
extent = np.array(im.get_extent())*NMPERPIXEL/1e3
im.set_extent(extent)
im = axs[0].imshow(clipped, cmap='gray', alpha=0.5)
im.set_extent(extent)

#angleplot = -np.where(mask, np.nan, np.rad2deg(angles))
im = axs[1].imshow(angleplot,
                   vmax=np.nanquantile(angleplot, 0.99),
                   vmin=np.nanquantile(angleplot, 0.01))
im.set_extent(extent)
plt.colorbar(im, ax=axs[1], label='angle (degree)', shrink=0.98, extend='both')
im = axs[1].imshow(clipped, cmap='gray', alpha=0.5)
im.set_extent(extent)
axs[0].set_title('a', fontweight='bold', loc='left')
axs[1].set_title('b', fontweight='bold', loc='left')
for i in range(2):
    axs[i].set_xlabel('x (μm)')
plt.savefig(os.path.join('plots', 'Linkopingextractedvalues.pdf'), dpi=600)
# -


# +
C0toC1 = LinearSegmentedColormap.from_list("C0toC1", ['C0', 'C1'])
plt.figure(figsize=[10, 10])

plt.imshow(mask, cmap=C0toC1)
plt.imshow(clipped, cmap='gray', alpha=0.5, vmax=np.quantile(clipped, 0.999))
# -

0.246 / (2*NMPERPIXEL/wavelengths) * 100

mask.sum() / np.ones_like(mask).sum()

vals = [
    np.where(
        magnitudes2[i] == np.max(
            magnitudes2,
            axis=0,
            keepdims=True),
        allwavelengths[i],
        np.nan) for i in range(3)]
vals = [(0.246 / (2*NMPERPIXEL/val) * 100)[~np.isnan(val)] for val in vals]

[v.shape for v in vals]

plt.hist(vals, bins=1000, histtype='barstacked', color=list('rgb'))

plt.hist(vals, bins=1000, histtype='barstacked', color=list('rgb'))

wadvs1 = np.stack([np.moveaxis(gs[i]['grad'], -1, 0) / 2/np.pi + pks[i, :, None, None] for i in range(3)])
allwavelengths1 = np.linalg.norm(wadvs1, axis=1)
vals1 = [allwavelengths1[i][mask] for i in range(3)]
vals1 = [(0.246 / (NMPERPIXEL/val) * 100) for val in vals1]
plt.hist(vals1, bins=1000, histtype='barstacked', color=list('rgb'))

np.concatenate(vals).ravel().shape

plt.hist([np.concatenate(vals).ravel(), np.concatenate(vals1).ravel()], bins=1000, histtype='barstacked')

weighted_vals = np.average(np.stack(vals1), weights=np.stack([m[mask] for m in magnitudes]), axis=0)

Xim = np.full_like(mask, np.nan, dtype=float)
Xim[mask] = weighted_vals
# Xim[~mask] =
plt.imshow(np.where(mask, Xim, wl_im), cmap='inferno')
plt.colorbar()
plt.contour(mask, levels=[0.5], colors='white')


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
axs[2].set_ylabel('extracted strain (%)')
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
axs[0].set_title('a', fontweight='bold', loc='left')
axs[1].set_title('b', fontweight='bold', loc='left')
axs[2].set_title('c', fontweight='bold', loc='left')
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
# -

plt.figure(figsize=[18, 18])
#plt.imshow(np.where(mask & (Xim < 0.3), Xim, np.nan), cmap='inferno', vmin=0.2, vmax=0.7)
#plt.imshow(np.where(mask & (Xim < 0.37), Xim, np.nan), cmap='inferno', vmin=0.2, vmax=0.7)
plt.imshow(np.where(~mask & (wl_im < 0.37), wl_im, np.nan), cmap='inferno', vmin=0.2, vmax=0.7)
plt.imshow(clipped, cmap='gray', alpha=0.5, vmax=np.quantile(clipped, 0.999))

plt.hist(np.concatenate(vals).ravel(), bins=1000, alpha=0.5, range=(0.1, 0.7))
plt.hist(weighted_vals.ravel(), bins=1000, alpha=0.5, range=(0.1, 0.7))


plt.hist(np.concatenate(vals).ravel(), bins=1000, alpha=0.5, range=(0.1, 0.7))
plt.hist(np.concatenate(vals1).ravel(), bins=1000, alpha=0.5, range=(0.1, 0.7))

fig, axs = plt.subplots(ncols=3, figsize=[26, 8], sharex=True, sharey=True)
for i in range(3):
    im = axs[i].imshow(np.where(mask, (0.246 / (NMPERPIXEL/allwavelengths1[i]) * 100), np.nan), vmin=0.1, vmax=0.6)
    plt.colorbar(im, ax=axs[i])

fig, axs = plt.subplots(ncols=3, figsize=[26, 8], sharex=True, sharey=True)
for i in range(3):
    axs[i].imshow(np.where(mask, angles[i], np.nan), cmap='twilight')

threshes = np.array([threshold_otsu(m) for m in magnitudes])
#threshes = np.array([threshold_otsu(magnitudes)]*3)
fig, axs = plt.subplots(ncols=2, figsize=[12, 6])
axs[0].hist([m.ravel() for m in magnitudes], bins=200, histtype='barstacked', alpha=0.7, color=rgb)
for i, t in enumerate(threshes):
    axs[0].axvline(t, color=rgb[i])
axs[1].imshow(to_KovesiRGB((magnitudes > threshes[:, None, None]).astype(float).T))

plt.figure(figsize=[5, 5])
plt.imshow(to_KovesiRGB((magnitudes > 0.011).astype(float).T))

angle = np.deg2rad((np.rad2deg(np.arctan2(pks[:, 1], pks[:, 0])) % 60).mean())

kmean = np.linalg.norm(pks, axis=1).mean()

pksnew = np.array([rotate(np.array([0, kmean]), -angle-np.pi/3*i) for i in range(3)])

# +
wadvs = []
for i in range(3):
    phase = np.angle(gs[i]['lockin'])
    #gphase = np.stack(_wrapToPi(np.array(np.gradient(-unwrap_phase(phase))))/2/np.pi)
    gphase = np.moveaxis(gs[i]['grad'], -1, 0)/2/np.pi
    print(gphase.shape, gphase.mean(axis=(1, 2)))
    w = gphase + pks[i, :, None, None]
    wadvs.append(w)
wadvs = np.stack(wadvs)

wxs = np.concatenate([wadvs[:, 0], -wadvs[:, 0]])
wys = np.concatenate([wadvs[:, 1], -wadvs[:, 1]])
# -

wadvs.mean(axis=(-1, -2))

pksnew = np.nanmean(np.where(eroded_mask, wadvs, np.nan), axis=(-1, -2))
pksnew

for ks in [np.nanmean(np.where(eroded_mask, wadvs, np.nan), axis=(-1, -2))*1.1,
           pksnew,
           pks, pks2]:
    plt.scatter(*np.stack([ks, -ks], axis=0).T)
plt.gca().set_aspect('equal')

u, gs = GPA.extract_displacement_field(clipped, pksnew, ksteps=3, kwscale=4, sigma=25,
                                       return_gs=True, wfr_func=cuGPA.wfr2_grad_opt)

# +
f, axs = plt.subplots(ncols=3, nrows=2, figsize=[18, 12])

magnitudes = np.stack([np.abs(g['lockin']) for g in gs])
angles = np.stack([np.angle(g['lockin']) for g in gs])
grads = np.stack([g['grad'] for g in gs])
gradmags = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs])
for i, ax in enumerate(axs.flat[:3]):
    ax.imshow(angles[i].T, cmap='twilight', interpolation='nearest')
    ax.imshow(clipped.T, cmap='gray', alpha=0.5)
    indicate_k(pks, i, ax=ax)
axs[1, 0].imshow(to_KovesiRGB((magnitudes/np.quantile(magnitudes, 0.99, axis=(1, 2), keepdims=True)).T))
axs[1, 1].imshow(clipped.T)
axs[1, 2].imshow(to_KovesiRGB((gradmags < 0.05).astype(float).T))
# -

wadvs.mean(axis=(-1, -2)), pks

#wxs = np.clip(wxs, -0.15,0.15)
#wys = np.clip(wys, -0.15,0.15)
p, _ = per(clipped-clipped.mean(), inverse_dft=False)
fftim = np.abs(np.fft.fftshift(p))
r = 0.04
fig, axs = plt.subplots(ncols=3, figsize=[26, 8], sharex=True, sharey=True)
axs[0].hist2d(wxs.ravel()[np.stack([mask]*6).ravel()],
              wys.ravel()[np.stack([mask]*6).ravel()],
              bins=500, cmap='cet_fire_r', vmax=2000,
              range=[(-r, r), (-r, r)])
axs[0].set_aspect('equal')
#plt.title(f'sigma={sigma}, kstep={kstep}')
pksweights = magnitudes/np.quantile(magnitudes[0], 0.99)
axs[1].hist2d(wxs.ravel()[np.stack([eroded_mask]*6).ravel()],
              wys.ravel()[np.stack([eroded_mask]*6).ravel()],
              # weights=np.stack([pksweights[:,eroded_mask]]*2).ravel(),
              bins=500, cmap='cet_fire_r', vmax=2000,
              range=[(-r, r), (-r, r)])
axs[1].set_aspect('equal')
fftplot(fftim, ax=axs[2], pcolormesh=False, vmax=np.quantile(fftim, 0.9999),
        vmin=np.quantile(fftim, 0.01), cmap='cet_fire_r',
        interpolation='none', origin='lower')
#plt.title(f'sigma={sigma}, kstep={kstep}')
axs[0].set_ylim(-r, r)
axs[0].set_xlim(-r, r)
axs[1].scatter(*(pks2*1.2).T)
axs[1].scatter(*-(pks2*1.2).T, color='C0')
axs[1].scatter(*pksnew.T, color=rgb, s=100)
axs[1].scatter(*-pksnew.T, color=rgb, s=220)
plt.tight_layout()

for ks in [np.nanmean(np.where(eroded_mask, wadvs, np.nan), axis=(-1, -2)), pks]:
    plt.scatter(*np.stack([ks, -ks], axis=0).T)
plt.gca().set_aspect('equal')
