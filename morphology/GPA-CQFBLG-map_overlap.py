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
#     display_name: Python [conda env:pyGPA-cupy]
#     language: python
#     name: conda-env-pyGPA-cupy-py
# ---

# +
import colorcet  # noqa: F401
import functools
from importlib import reload
import pyGPA.cuGPA as cuGPA
from pyGPA.imagetools import to_KovesiRGB, indicate_k
from latticegen.transformations import rotate
from dask.distributed import Client
import numpy as np
import matplotlib.pyplot as plt
import pyGPA.geometric_phase_analysis as GPA
from pyGPA.imagetools import gauss_homogenize2, fftplot
import os
from dask_image.imread import imread

from moisan2011 import per
import scipy.ndimage as ndi
import dask.array as da
from dask_cuda import LocalCUDACluster
from skimage.filters import threshold_otsu
from matplotlib_scalebar.scalebar import ScaleBar

# %matplotlib inline
# -

cluster = LocalCUDACluster(threads_per_worker=4, device_memory_limit=0.8)
client = Client(cluster)
client

# +
folder = '/mnt/storage-linux/domainsdata/2019-04-09-CQFBLG_01_litho'
name = os.path.join('20190409_115642_3.5um_636.8_sweep-STAGE_X-STAGE_Y',
                    'stitch_v10_2021-11-15_1406_sobel_4_bw_200')

image = imread(os.path.join(folder, name+'.tif')).astype(float).squeeze().T.rechunk((1000, 1000))
NMPERPIXEL = 2.23
# -

plt.imshow(image > 0)

2592/8/4

1152816 * 16.0 * 0.25 * 2592/(image > 0).sum().compute()

sigma = 50
smooth = da.map_overlap(gauss_homogenize2, image,
                        da.ones_like(image),
                        depth=3*sigma, sigma=sigma)

plt.imshow(smooth[1500:3500, 1500:3500].compute())

f, ax = plt.subplots(figsize=[12, 12])
cropped = smooth[4600:4600+1024, 4400:4400+1024].compute()
ax.imshow(cropped.T, vmin=0.75)

plt.hist(cropped.ravel(), bins=100);

clipped = np.clip(cropped, 0.75, None)

p, s = per(clipped - clipped.mean(), inverse_dft=False)

fftim = np.fft.fftshift(np.abs(p))

f, ax = plt.subplots(figsize=[12, 12])
im = fftplot(ndi.filters.gaussian_filter(fftim, sigma=1), pcolormesh=False, ax=ax, vmax=4e3)
fftr = 0.2
plt.xlim(-fftr, fftr)
plt.ylim(-fftr, fftr)
plt.colorbar(im, ax=ax)

pk, _ = GPA.extract_primary_ks(clipped - clipped.mean(),
                               plot=True, sigma=15, pix_norm_range=(70, 200), DoG=False)


reload(cuGPA)

pks = np.array([rotate(pk, i*np.pi/3) for i in range(3)]).squeeze()

# +

kw = np.linalg.norm(pks, axis=1).mean() / 8
kstep = kw / 6
sigma = 15

gfunc = functools.partial(cuGPA.wfr2_only_lockin, sigma=sigma, kw=kw, kstep=kstep)

gs = da.stack([da.map_overlap(gfunc, smooth, depth=4*sigma, dtype=smooth.dtype,
                              kvec=pk) for pk in pks])
# -

xslice = slice(5600, 5600+2048)
yslice = slice(2500, 2500+2048)

# + tags=[]
plt.figure(figsize=[12, 12])
plt.imshow(image[xslice, yslice].T, cmap='gray')
# -

magnitudes = np.abs(gs)
angles = np.angle(gs)
X = magnitudes[:, xslice, yslice].compute()

gfunc2 = functools.partial(cuGPA.wfr2_only_grad, sigma=sigma, kw=kw, kstep=kstep)
grads = da.stack([da.map_overlap(gfunc2, smooth, depth=[0, 4*sigma, 4*sigma],
                                 dtype=smooth.dtype, new_axis=2, chunks=(1000, 1000, 2),
                                 kvec=pk) for pk in pks])

ws = grads[:, xslice, yslice].compute()
ws = np.moveaxis(ws, -1, 1)
ws.shape

ws = ws / (2*np.pi)
ws = ws + pks[..., None, None]
ws.shape

normmag = (X/X.mean(axis=(1, 2), keepdims=True)).max(axis=0)
wavelengths = np.where(X/X.mean(axis=(1, 2), keepdims=True) == normmag,
                       np.linalg.norm(ws, axis=1),
                       np.nan)
relangles = np.arctan2(ws[:, 1], ws[:, 0])
relangles = np.where(X/X.mean(axis=(1, 2), keepdims=True) == normmag,
                     relangles,
                     np.nan)

plt.figure(figsize=[18, 18])
for norm in 1/wavelengths:
    plt.imshow(norm.T, cmap='inferno',
               vmax=np.nanquantile(1/wavelengths, 0.999),
               vmin=np.nanquantile(1/wavelengths, 0.001))
plt.colorbar()
plt.imshow(smooth[xslice, yslice].T, cmap='gray',
           vmax=np.quantile(cropped, 0.98), alpha=0.3, vmin=0.75)


plt.imshow(np.where(normmag > 1.5, relangles, np.nan).T)

plt.imshow(normmag.T)

(normmag > 1.5).sum() / (2048*2048)

plt.imshow((normmag > 1.5).T)
plt.imshow(smooth[xslice, yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)

fig, ax = plt.subplots(ncols=2, figsize=[18, 9])
ax[0].imshow(np.where(normmag > 1.5, rgbim.T, np.nan).T)
ax[1].imshow(np.stack(rgbmasks).T.astype(float))

(np.stack(rgbmasks).T.astype(float) > 0).sum(), (~np.isnan(np.where(normmag > 1.5, rgbim.T, np.nan))).sum()/3

np.isnan(wavelengths).sum()/2/2048

np.any(np.array(rgbmasks), axis=0).sum()/2048

plt.imshow(~np.any(np.array(rgbmasks), axis=0).T)

plt.imshow(np.stack(rgbmasks).T.astype(float))

np.array(rgbmasks).sum()/2048

[3, 4, 5][[0, 2, 1]]

allnormmag = (X/X.mean(axis=(1, 2), keepdims=True))
rgbmasks = [(allnormmag[i] == np.max(allnormmag, axis=0)) & (normmag > 1.5) for i in range(3)]
plt.imshow(np.stack(rgbmasks).T.astype(float))

(normmag > 1.5).sum()/2048/2048

plt.imshow(np.stack(rgbmasks).max(axis=0))

[r.sum()/2048/2048 for r in rgbmasks]

# +
fig, ax = plt.subplots(ncols=3, figsize=[9, 3.2],
                       gridspec_kw={'width_ratios': [1, 1, 0.6]}, constrained_layout=True)
rgbim = to_KovesiRGB((X/np.quantile(X, 0.99, axis=(1, 2), keepdims=True)).T)
# ax[0].imshow(rgbim)
ax[0].imshow(np.where(np.stack(rgbmasks).max(axis=0).T[..., None],
                      to_KovesiRGB(np.stack(rgbmasks).T).astype(float), np.nan))
ax[0].imshow(smooth[xslice, yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98),
             alpha=0.65, vmin=0.75
             )


ax[1].imshow(np.where(np.any(np.array(rgbmasks), axis=0),
                      1, 0).T, cmap='gray', interpolation='none')
im = ax[1].imshow(np.where(np.any(np.array(rgbmasks), axis=0),
                           np.nanmean(0.246 / (2*NMPERPIXEL/wavelengths) * 100, axis=0), np.nan).T,
                  cmap='inferno', interpolation='none', vmax=0.7, vmin=0.2)
plt.colorbar(im, ax=ax[:2], fraction=0.05, aspect=50)
ax[1].imshow(smooth[xslice, yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98),
             alpha=0.65, vmin=0.75)

ax[2].hist([np.where(normmag > 1.5, 0.246 / (2*NMPERPIXEL/wavelengths[i]) * 100, np.nan).ravel() for i in [0, 2, 1]],
           histtype='barstacked', bins=1000, color=list('rbg'),
           orientation='horizontal', range=(0.2, 0.7))
ax[2].set_ylabel('relative strain (%)')
ax[2].set_ylim(0.2, 0.7)

#ax[0].imshow(np.where(normmag > 1.5, rgbim.T, np.nan).T)


#ax[0].imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
for i in range(2):
    ax[i].tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
ax[2].tick_params(
    bottom=False,      # ticks along the bottom edge are off
    left=True,
    labelbottom=False,
    labelleft=True)
scalebar = ScaleBar(1e-9*NMPERPIXEL, "m", length_fraction=0.15,
                    location="lower left", box_alpha=0.4,
                    width_fraction=0.02
                    )
ax[0].add_artist(scalebar)
for i, l in enumerate('abc'):
    ax[i].set_title(l, fontweight='bold', loc='left')

ax[2].axhline(0.3, label=r'$\epsilon_{c0}$', color='black', alpha=0.5)
leftx = ax[2].get_xlim()[1]
ax[2].annotate(r'$\epsilon_{c0}$', (leftx, 0.3),
               ha='right', va='bottom',
               xytext=(-1.5, 1.5), textcoords='offset points')
ax[2].axhline(0.37, color='black', alpha=0.5)
ax[2].annotate(r'$\epsilon_{c1}$', (leftx, 0.37),
               ha='right', va='bottom',
               xytext=(-1.5, 1.5), textcoords='offset points')
# plt.tight_layout()
plt.savefig(os.path.join('plots', 'CQFBLGstrainhist.pdf'), dpi=600)
# -

[x.shape for x in rgbmasks]

NMPERPIXEL

vals = [np.where((allnormmag[i] == np.max(allnormmag, axis=0)),  # & (normmag > 1.5),
                 wavelengths[i], np.nan) for i in range(3)]
print([(~np.isnan(val)).sum() for val in vals])
vals = [(0.246 / (2*NMPERPIXEL/val) * 100)[~np.isnan(val)] for val in vals]
plt.hist(vals,
         histtype='barstacked', bins=1000, color=list('rgb'), range=(0.3, 0.65))

[np.nanquantile(np.where((allnormmag[i] == np.max(allnormmag, axis=0)), allnormmag[i], np.nan), 0.1) for i in range(3)]

(3 - np.isnan(wavelengths).sum(axis=0) > 1).sum()

np.sum([(allnormmag[i] == np.max(allnormmag, axis=0)).sum() for i in range(3)])

(~np.isnan(np.stack([np.where(normmag > 1.5,
                              0.246 / (2*NMPERPIXEL/wavelengths[i]) * 100,
                              np.nan).ravel() for i in range(3)]))).sum(axis=1)

plt.hist([np.where(normmag > 1.5, 0.246 / (2*NMPERPIXEL/wavelengths[i]) * 100, np.nan).ravel() for i in range(3)],
         histtype='barstacked', bins=1000, color=list('rgb'))
plt.xlabel('strain in percent')

plt.hist(np.where(normmag > 1.5, 0.246 / (2*NMPERPIXEL/wavelengths) * 100, np.nan).ravel(), bins=1000)
plt.xlabel('strain in percent')

plt.hist(np.where(normmag > 1.5, relangles, np.nan).ravel(), bins=1000)

plt.figure(figsize=[18, 18])
for i, angle in enumerate(relangles):
    plt.imshow(angle.T - i*np.pi/3, cmap='seismic',
               vmax=np.pi/3,
               vmin=0)
plt.colorbar()
plt.imshow(smooth[xslice, yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.3, vmin=0.75)

pks

newpks

normmag = (X/X.mean(axis=(1, 2), keepdims=True)).max(axis=0)

weights.shape

ws[:, 0].shape

# +
weights = np.where(np.logical_and(X/X.mean(axis=(1, 2), keepdims=True) == normmag, normmag > 1.5),
                   normmag, 0)

#weights = (normmag > threshold_otsu(normmag)-0.2)

newpks = np.array([np.average(ws[:, i], axis=(-1, -2),
                              weights=weights) for i in range(2)]).T
# -

(weights > 0).sum() / (weights > -1).sum()


f, ax = plt.subplots(figsize=[12, 12])
wxs = np.concatenate([ws[:, 0][weights > 0], -ws[:, 0][weights > 0]])
wys = np.concatenate([ws[:, 1][weights > 0], -ws[:, 1][weights > 0]])
plt.hist2d(wxs.ravel(),
           wys.ravel(),
           bins=500, cmap='cet_fire_r', vmax=3e3, range=[(-fftr, fftr), (-fftr, fftr)])

f, ax = plt.subplots(figsize=[12, 12])
wxs = np.concatenate([ws[:, 0], -ws[:, 0]])
wys = np.concatenate([ws[:, 1], -ws[:, 1]])
plt.hist2d(wxs.ravel(),
           wys.ravel(),
           bins=500, cmap='cet_fire_r', vmax=1e4, range=[(-fftr, fftr), (-fftr, fftr)])

f, ax = plt.subplots(figsize=[12, 12])
im = fftplot(ndi.filters.gaussian_filter(fftim, sigma=1), pcolormesh=False, ax=ax, vmax=1e3, origin='lower')
fftr = 0.15
plt.xlim(-fftr, fftr)
plt.ylim(-fftr, fftr)
plt.scatter(*np.concatenate([pks, -pks]).T)
plt.scatter(*np.concatenate([newpks, -newpks]).T)
plt.gca().set_aspect('equal')

plt.figure(figsize=[18, 18])
plt.imshow(to_KovesiRGB((X/np.quantile(X, 0.99, axis=(1, 2), keepdims=True)).T))
plt.imshow(smooth[xslice, yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
plt.colorbar()

fig, ax = plt.subplots(ncols=2, figsize=[18, 9])
rgbim = to_KovesiRGB((X/np.quantile(X, 0.99, axis=(1, 2), keepdims=True)).T)
ax[0].imshow(rgbim)
#ax[0].imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
ax[1].imshow(np.where(normmag > 1.5, rgbim.T, np.nan).T)
ax[1].imshow(smooth[xslice, yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)

Y = angles[:, xslice, yslice]  # .compute()
Y.shape

Y[0].min()

# +
f, axs = plt.subplots(ncols=3,  nrows=2, figsize=[18, 12])

for i in range(3):
    axs[0, i].imshow(Y[i].T, cmap='twilight', interpolation='nearest')
    axs[1, i].imshow(X[i].T,)
# -

X.mean(axis=(1, 2))

threshold_otsu(normmag)

plt.hist((X/X.mean(axis=(1, 2), keepdims=True)).ravel(), bins=1000)
plt.hist(normmag.ravel(), bins=1000)

# +

#normmag = (X/X.mean(axis=(1,2), keepdims=True)).max(axis=0)
plt.figure(figsize=[18, 18])

#plt.imshow((normmag > threshold_otsu(normmag)-0.2).T)
plt.imshow((normmag > 1.5).T)
plt.imshow(smooth[xslice, yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
# -

plt.imshow((X/X.mean(axis=(1, 2), keepdims=True)).max(axis=0).T)

plt.imshow(to_KovesiRGB(X.T)/X.max())

plt.imshow(to_KovesiRGB((X/np.quantile(X, 0.99, axis=(1, 2), keepdims=True)).T))

# +

gs = []
for j, pk in enumerate(pks):
    da.map_blocks()
    g = cuGPA.wfr2_grad_opt(deformed, sigma,
                            pk[0], pk[1], kw=kw,
                            kstep=kstep)
    gs.append(g)
# -

clipped = np.clip(smooth[2500:2500+2048, 2500:2500+2048], 0.75, None)

plt.figure(figsize=[18, 18])
plt.imshow(to_KovesiRGB((magnitudes/np.quantile(magnitudes, 0.99, axis=(1, 2), keepdims=True)).T))
plt.imshow(clipped.T, cmap='gray', vmax=np.quantile(clipped, 0.98), alpha=0.5)
plt.colorbar()

normmags = (magnitudes/np.quantile(magnitudes, 0.99, axis=(1, 2), keepdims=True))
normmags.shape

plt.hist([x.ravel() for x in normmags], bins=250, range=(0, 1.5), histtype='barstacked')

u, gs = GPA.extract_displacement_field(clipped, pks, ksteps=3, kwscale=20, sigma=12,
                                       return_gs=True, wfr_func=GPA.wfr2_grad_opt)

# +
f, axs = plt.subplots(ncols=3, nrows=2, figsize=[18, 12])

magnitudes = np.stack([np.abs(g['lockin']) for g in gs])
angles = np.stack([np.angle(g['lockin']) for g in gs])
grads = np.stack([g['grad'] for g in gs])
gradmags = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs])
for i, ax in enumerate(axs.flat[:3]):
    ax.imshow(angles[i].T, cmap='twilight', interpolation='nearest')
    indicate_k(pks, i, ax=ax)
axs[1, 0].imshow(to_KovesiRGB((magnitudes/np.quantile(magnitudes, 0.99, axis=(1, 2), keepdims=True)).T))
axs[1, 1].imshow(clipped.T)
axs[1, 2].imshow(to_KovesiRGB((gradmags < 0.05).astype(float).T))

# +
f, axs = plt.subplots(ncols=3, nrows=2, figsize=[18, 12])

magnitudes = np.stack([np.abs(g['lockin']) for g in gs])
angles = np.stack([np.angle(g['lockin']) for g in gs])
grads = np.stack([g['grad'] for g in gs])
gradmags = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs])
for i, ax in enumerate(axs.flat[:3]):
    ax.imshow(angles[i].T, cmap='twilight', interpolation='nearest')
    indicate_k(pks, i, ax=ax)
axs[1, 0].imshow(to_KovesiRGB((magnitudes/np.quantile(magnitudes, 0.99, axis=(1, 2), keepdims=True)).T))
axs[1, 1].imshow(clipped.T)
axs[1, 2].imshow(to_KovesiRGB((gradmags < 0.05).astype(float).T))
# -

pks

fig, ax = plt.subplots(ncols=3, figsize=[18, 6])
cutoff = 0.05
for i in range(3):
    ax[i].hist2d(*grads[i][gradmags[i] < cutoff, :].T,
                 weights=(magnitudes[i]**2)[gradmags[i] < cutoff],
                 bins=50)
    print(np.average(grads[i][gradmags[i] < cutoff, :], weights=(
        magnitudes[i]**2)[gradmags[i] < cutoff], axis=0), pks[i])
    ax[i].set_aspect('equal')
    indicate_k(pks, i, ax=ax[i], inset=False, origin='lower')
    ax[i].scatter(*np.average(grads[i][gradmags[i] < cutoff, :],
                  weights=(magnitudes[i]**2)[gradmags[i] < cutoff], axis=0), c='red')

cutoff = 0.08
plt.figure(figsize=[12, 12])
plt.imshow(to_KovesiRGB(np.where(gradmags < cutoff, (cutoff-gradmags)/cutoff, 0).T))

[g.mean() for g in grads]
