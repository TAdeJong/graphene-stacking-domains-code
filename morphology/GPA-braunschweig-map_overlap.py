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
import numpy as np
import matplotlib.pyplot as plt
import pyGPA.geometric_phase_analysis as GPA
from pyGPA.imagetools import gauss_homogenize2, fftplot
import os
from dask_image.imread import imread
import dask_image
#from skimage.io import imread
from moisan2011 import per
import scipy.ndimage as ndi
import dask.array as da
from dask_cuda import LocalCUDACluster
from skimage.filters import threshold_otsu

# %matplotlib inline
# -

cluster = LocalCUDACluster(threads_per_worker=4, device_memory_limit=0.8)
cluster

from dask.distributed import Client
client = Client(cluster)
client

# +
folder = '/mnt/storage-linux/domainsdata/2019-10-22-Anna2017-020/20191022_175235_3.5um_537.1_sweep-STAGE_X-STAGE_Y'
name = 'stitch_v10_2022-02-14_1050_sobel_6_bw_200'

image = imread(os.path.join(folder, name+'.tif')).astype(float).squeeze().T.rechunk((1000,1000))
NMPERPIXEL = 2.23
# -

plt.imshow(image>0)

(image>0).sum().compute()

1152816 * 16.0 * 0.25 * 81 / (image>0).sum().compute()

sigma = 50
smooth = da.map_overlap(gauss_homogenize2, image,
                        da.ones_like(image),
                        depth=3*sigma, sigma=sigma)

f, ax = plt.subplots(figsize=[12,12])
cropped = smooth[2000:4000,1500:3500].compute()
ax.imshow(cropped.T, vmin=0.75)

plt.hist(cropped.ravel(), bins=100);

clipped = np.clip(cropped, 0.75, None)

p,s = per(clipped - clipped.mean(), inverse_dft=False)

fftim = np.fft.fftshift(np.abs(p))

f, ax = plt.subplots(figsize=[12,12])
im = fftplot(ndi.filters.gaussian_filter(fftim, sigma=3), pcolormesh=False, ax=ax, vmax=5e3)
fftr=0.2
plt.xlim(-fftr,fftr)
plt.ylim(-fftr,fftr)
plt.colorbar(im, ax=ax)

pks, _ = GPA.extract_primary_ks(clipped - clipped.mean(), plot=True, sigma=8, pix_norm_range=(70,200), DoG=False)

from latticegen.transformations import rotate
from pyGPA.imagetools import to_KovesiRGB, indicate_k
import pyGPA.cuGPA as cuGPA

# +
import functools

kw = np.linalg.norm(pks, axis=1).mean() / 4
kstep = kw / 6
sigma = 15

gfunc = functools.partial(cuGPA.wfr2_only_lockin, sigma=sigma, kw=kw, kstep=kstep)

gs = da.stack([da.map_overlap(gfunc, smooth, depth=4*sigma, dtype=smooth.dtype,
                     kvec=pk) for pk in pks])
# -

pks2 = rotate(pks/2., np.pi/6)



# +
sigma2 = 25

gfunc2 = functools.partial(cuGPA.wfr2_only_lockin, sigma=sigma2, kw=kw, kstep=kstep)

gs2 = da.stack([da.map_overlap(gfunc2, smooth, depth=4*sigma2, dtype=smooth.dtype,
                     kvec=pk) for pk in pks2])
# -

plt.scatter(*np.concatenate([pks2, -pks2]).T)
plt.scatter(*np.concatenate([pks, -pks]).T)
plt.gca().set_aspect('equal')

xslice = slice(2000, 2000+2048)
yslice = slice(1500, 1500+2048)

# + tags=[]
plt.figure(figsize=[12,12])
plt.imshow(image[xslice,yslice].T, cmap='gray')
# -

magnitudes = np.abs(gs)
angles = np.angle(gs)
X = magnitudes[:,xslice,yslice].compute()

gfunc2 = functools.partial(cuGPA.wfr2_only_grad, sigma=sigma, kw=kw, kstep=kstep)
grads =  da.stack([da.map_overlap(gfunc2, smooth, depth=[0,4*sigma,4*sigma], dtype=smooth.dtype, new_axis=2, chunks=(1000,1000,2),
                     kvec=pk) for pk in pks])

ws = grads[:,xslice,yslice].compute()
ws = np.moveaxis(ws, -1,1)
ws = ws / (2*np.pi)
ws = ws + pks[...,None,None]

# +

ws.shape
# -

normmag = (X/X.mean(axis=(1,2), keepdims=True)).max(axis=0)
wavelengths = np.where(X/X.mean(axis=(1,2), keepdims=True) == normmag,
                 np.linalg.norm(ws, axis=1),
                 np.nan)
relangles = np.arctan2(ws[:,1], ws[:,0])
relangles = np.where(X/X.mean(axis=(1,2), keepdims=True) == normmag,
                 relangles,
                 np.nan)

grads2.shape, grads.shape

magnitudes2 = np.abs(gs2)
angles2 = np.angle(gs2)
X2 = magnitudes2[:,xslice,yslice].compute()
gfunc3 = functools.partial(cuGPA.wfr2_only_grad, sigma=sigma2, kw=kw, kstep=kstep)
grads2 =  da.stack([da.map_overlap(gfunc3, smooth, depth=[0, 4*sigma2, 4*sigma2],
                                   dtype=smooth.dtype, new_axis=2, chunks=(1000,1000,2),
                     kvec=pk) for pk in pks2])
ws2 = grads2[:,xslice,yslice].compute()
ws2 = np.moveaxis(ws2 / (2*np.pi), -1, 1) + (pks[...,None,None] / 2.2)

fig, ax = plt.subplots(ncols=3, figsize=[18,6])
for i,phase in enumerate(angles2):
    ax[i].imshow(phase[xslice,yslice].compute(), cmap='twilight', interpolation='nearest')

fig, ax = plt.subplots(ncols=3, figsize=[18,6])
for i,phase in enumerate(angles2):
    ax[i].imshow(phase[xslice,yslice].compute(), cmap='twilight', interpolation='nearest')

plt.figure(figsize=[18,18])
plt.imshow(to_KovesiRGB((X2 / np.quantile(X2, 0.99, axis=(1,2), keepdims=True)).T))
#plt.imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
#plt.colorbar()

[np.rad2deg(np.nanmean(angle)) for angle in relangles]

plt.imshow(np.where(normmag > 1.2, relangles, np.nan).T)

(normmag > 1.5).sum() / (2048*2048)

plt.hist(np.where(normmag > 1.5, 1/wavelengths, np.nan).ravel(), bins=1000);

plt.hist(np.where(normmag > 1.5, 1/wavelengths, np.nan).ravel(), bins=1000);

plt.hist([np.where(normmag > 1.2, 0.246 / (2*NMPERPIXEL/wavelengths[i]) * 100, np.nan).ravel() for i in range(3)],
         histtype='barstacked', bins=1000, color=list('rgb'));
plt.xlabel('strain in percent')

plt.hist([np.where(normmag > 1.2, 0.246 / (2*NMPERPIXEL/wavelengths[i]) * 100, np.nan).ravel() for i in range(3)],
         histtype='barstacked', bins=1000, color=list('rgb'));
plt.xlabel('strain in percent')

allnormmag = (X/X.mean()) #axis=(1,2), keepdims=True))
normmag = allnormmag.max(axis=0)
cutoff = 1.3
rgbmasks = [(allnormmag[i] == np.max(allnormmag, axis=0)) & (normmag > cutoff) for i in range(3)]


[r.sum()/2048/2048 for r in rgbmasks]

sum([r.sum()/2048/2048 for r in rgbmasks])

plt.imshow(np.array(rgbmasks).T.astype(float))

plt.figure(figsize=[12,12])
#plt.imshow(normmag.T)
plt.imshow(np.array(rgbmasks).T.astype(float))
plt.imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.3, vmin=0.75)
#plt.contour(normmag.T, colors='white')

# +
fig, ax = plt.subplots(ncols=3, figsize=[18,6.2],
                      gridspec_kw={'width_ratios': [1, 1, 0.6]}, constrained_layout=True)
rgbim = to_KovesiRGB((X/np.quantile(X, 0.99, axis=(1,2), keepdims=True)).T)
#ax[0].imshow(rgbim)
ax[0].imshow(to_KovesiRGB(np.stack(rgbmasks).T).astype(float))
ax[0].imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)


ax[1].imshow(np.where(np.any(np.array(rgbmasks), axis=0),
                      1, 0).T, cmap='gray', interpolation='none')
im = ax[1].imshow(np.where(np.any(np.array(rgbmasks), axis=0),
                     np.nanmean( 0.246 / (2*NMPERPIXEL/wavelengths) * 100, axis=0), np.nan).T,
                  cmap='inferno', interpolation='none', vmax=0.7, vmin=0.2)
plt.colorbar(im, ax=ax[:2], fraction=0.05, aspect=50)
ax[1].imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
ax[2].hist([np.where(normmag > cutoff, 0.246 / (2*NMPERPIXEL/wavelengths[i]) * 100, np.nan).ravel() for i in [0,2,1]],
         histtype='barstacked', bins=1000, color=list('rbg'),
           orientation='horizontal', range=(0.2,0.7))
ax[2].set_ylabel('relative strain (%)')
ax[2].set_ylim(0.2,0.7)

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
for i,l in enumerate('abc'):
    ax[i].set_title(l, fontweight='bold', loc='left')
# -

from matplotlib_scalebar.scalebar import ScaleBar


# +
fig, ax = plt.subplots(ncols=3, figsize=[9,3.2],
                      gridspec_kw={'width_ratios': [1, 1, 0.6]}, constrained_layout=True)
rgbim = to_KovesiRGB((X/np.quantile(X, 0.99, axis=(1,2), keepdims=True)).T)
#ax[0].imshow(rgbim)
ax[0].imshow(to_KovesiRGB(np.stack(rgbmasks).T).astype(float))
ax[0].imshow(smooth[xslice,yslice].T, cmap='gray', 
             vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)


ax[1].imshow(np.where(np.any(np.array(rgbmasks), axis=0),
                      1, 0).T, cmap='gray', interpolation='none')
im = ax[1].imshow(np.where(np.any(np.array(rgbmasks), axis=0),
                     np.nanmean( 0.246 / (2*NMPERPIXEL/wavelengths) * 100, axis=0), np.nan).T,
                  cmap='inferno', interpolation='none', vmax=0.7, vmin=0.2)
plt.colorbar(im, ax=ax[:2], fraction=0.05, aspect=50)
ax[1].imshow(smooth[xslice,yslice].T, cmap='gray',
             vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
ax[2].hist([np.where(normmag > cutoff, 0.246 / (2*NMPERPIXEL/wavelengths[i]) * 100, np.nan).ravel() for i in [0,2,1]],
         histtype='barstacked', bins=1000, color=list('rbg'),
           orientation='horizontal', range=(0.2,0.7))
ax[2].set_ylabel('relative strain (%)')
ax[2].set_ylim(0.2,0.7)

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
for i,l in enumerate('abc'):
    ax[i].set_title(l, fontweight='bold', loc='left')
    
ax[2].axhline(0.3, label=r'$\epsilon_{c0}$', color='black', alpha=0.5)
leftx = ax[2].get_xlim()[1]
ax[2].annotate(r'$\epsilon_{c0}$', (leftx,0.3),
                ha='right', va='bottom',
                xytext=(-1.5, 1.5), textcoords='offset points')
ax[2].axhline(0.37, color='black', alpha=0.5)
ax[2].annotate(r'$\epsilon_{c1}$', (leftx,0.37),
                ha='right', va='bottom',
                xytext=(-1.5, 1.5), textcoords='offset points')
#plt.tight_layout()
plt.savefig(os.path.join('plots', 'BSstrainhist.pdf'), dpi=600)
# -

plt.hist([np.where(normmag > 1.2, 0.246 / (2*NMPERPIXEL/wavelengths[i]) * 100, np.nan).ravel() for i in range(3)],
         histtype='barstacked', bins=1000, color=list('rgb'));
plt.xlabel('strain in percent')

plt.hist(np.where(normmag > 1.5, 0.246 / (2*NMPERPIXEL/wavelengths) * 100, np.nan).ravel(), bins=1000);
plt.xlabel('strain in percent')

plt.hist([np.where(normmag > 1.5, np.rad2deg(relangles[i]), np.nan).ravel() for i in range(3)],
         bins=1000, histtype='barstacked', color=list('rgb'));

plt.hist([np.where(normmag > 1.2, np.rad2deg(relangles[i]) % 180, np.nan).ravel() for i in range(3)],
         bins=1000, histtype='barstacked', color=list('rgb'));

plt.figure(figsize=[18,18])
for i,angle in enumerate(relangles):
    plt.imshow(angle.T % np.pi/3, cmap='twilight',
               vmax=np.pi/3,
               vmin=0)
plt.colorbar()
plt.imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.3, vmin=0.75)

pks

newpks

normmag = (X/X.mean(axis=(1,2), keepdims=True)).max(axis=0)

weights.shape

ws[:,0].shape

# +
weights = np.where(np.logical_and(X/X.mean(axis=(1,2), keepdims=True) == normmag, normmag > 1.2),
                   normmag, 0)

#weights = (normmag > threshold_otsu(normmag)-0.2)

newpks = np.array([np.average(ws[:,i], axis=(-1,-2),
                              weights=weights) for i in range(2)]).T
# -

(weights > 0).sum() /  (weights > -1).sum()

import colorcet

pks

f, ax = plt.subplots(figsize=[12,12])
wxs = np.concatenate([ws[:,0][weights > 0], -ws[:,0][weights > 0]])
wys = np.concatenate([ws[:,1][weights > 0], -ws[:,1][weights > 0]])
plt.hist2d(wxs.ravel(),
           wys.ravel(),
           bins=1000, cmap='cet_fire_r',vmax=5e2, range=[(-fftr,fftr),(-fftr,fftr)]);
plt.scatter(*np.concatenate([pks, -pks]).T)

f, ax = plt.subplots(figsize=[12,12])
wxs = np.concatenate([ws[:,0], -ws[:,0]])
wys = np.concatenate([ws[:,1], -ws[:,1]])
plt.hist2d(wxs.ravel(),
           wys.ravel(),
           bins=500, cmap='cet_fire_r',vmax=1e4, range=[(-fftr/2,fftr/2),(-fftr/2,fftr/2)]);
plt.grid()

f, ax = plt.subplots(figsize=[12,12])
im = fftplot(ndi.filters.gaussian_filter(fftim, sigma=1), pcolormesh=False, ax=ax, vmax=3e3, origin='lower')
fftr=0.15
plt.xlim(-fftr,fftr)
plt.ylim(-fftr,fftr)
plt.scatter(*np.concatenate([pks, -pks]).T)
plt.scatter(*np.concatenate([newpks, -newpks]).T)
plt.gca().set_aspect('equal')

plt.figure(figsize=[18,18])
plt.imshow(to_KovesiRGB((X/np.quantile(X, 0.99, axis=(1,2), keepdims=True)).T))
plt.imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
plt.colorbar()

fig, ax = plt.subplots(ncols=2, figsize=[18,9])
rgbim = to_KovesiRGB((X/np.quantile(X, 0.99, 
                                    #axis=(1,2),
                                    keepdims=True)).T)
ax[0].imshow(rgbim)
#ax[0].imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
ax[1].imshow(np.where(normmag > 1.2, rgbim.T, np.nan).T)
ax[1].imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)

Y = angles[:,xslice,yslice].compute()
Y.shape

# +
f, axs = plt.subplots(ncols=3,  nrows=2, figsize=[18,12])

for i in range(3):
    axs[0,i].imshow(Y[i].T, cmap='twilight', interpolation='nearest')
    axs[1,i].imshow(X[i].T,)
# -

X.mean(axis=(1,2))

threshold_otsu(normmag)

plt.hist((X/X.mean(axis=(1,2), keepdims=True)).ravel(), bins=1000);
plt.hist(normmag.ravel(), bins=1000);

# +

#normmag = (X/X.mean(axis=(1,2), keepdims=True)).max(axis=0)
plt.figure(figsize=[18,18])

#plt.imshow((normmag > threshold_otsu(normmag)-0.2).T)
plt.imshow((normmag > 1.5).T)
plt.imshow(smooth[xslice,yslice].T, cmap='gray', vmax=np.quantile(cropped, 0.98), alpha=0.5, vmin=0.75)
# -

plt.imshow((X/X.mean(axis=(1,2), keepdims=True)).max(axis=0).T)

plt.imshow(to_KovesiRGB(X.T)/X.max())

plt.imshow(to_KovesiRGB((X/np.quantile(X, 0.99, axis=(1,2), keepdims=True)).T))

# +

gs = []
for j, pk in enumerate(pks):
    da.map_blocks()
    g = cuGPA.wfr2_grad_opt(deformed, sigma,
                            pk[0], pk[1], kw=kw,
                            kstep=kstep)
    gs.append(g)
# -

clipped = np.clip(smooth[2500:2500+2048,2500:2500+2048], 0.75, None)

plt.figure(figsize=[18,18])
plt.imshow(to_KovesiRGB((magnitudes/np.quantile(magnitudes, 0.99, axis=(1,2), keepdims=True)).T))
plt.imshow(clipped.T, cmap='gray', vmax=np.quantile(clipped, 0.98), alpha=0.5)
plt.colorbar()

normmags = (magnitudes/np.quantile(magnitudes, 0.99, axis=(1,2), keepdims=True))
normmags.shape

plt.hist([x.ravel() for x in normmags], bins=250, range=(0,1.5), histtype='barstacked');

u, gs = GPA.extract_displacement_field(clipped, pks, ksteps=3, kwscale=20, sigma=12, return_gs=True, wfr_func=GPA.wfr2_grad_opt)

# +
f, axs = plt.subplots(ncols=3, nrows=2, figsize=[18,12])

magnitudes = np.stack([np.abs(g['lockin']) for g in gs])
angles = np.stack([np.angle(g['lockin']) for g in gs])
grads = np.stack([g['grad'] for g in gs])
gradmags = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs])
for i, ax in enumerate(axs.flat[:3]):
    ax.imshow(angles[i].T, cmap='twilight', interpolation='nearest')
    indicate_k(pks, i, ax=ax)
axs[1,0].imshow(to_KovesiRGB((magnitudes/np.quantile(magnitudes, 0.99, axis=(1,2), keepdims=True)).T))
axs[1,1].imshow(clipped.T)
axs[1,2].imshow(to_KovesiRGB((gradmags < 0.05).astype(float).T))

# +
f, axs = plt.subplots(ncols=3, nrows=2, figsize=[18,12])

magnitudes = np.stack([np.abs(g['lockin']) for g in gs])
angles = np.stack([np.angle(g['lockin']) for g in gs])
grads = np.stack([g['grad'] for g in gs])
gradmags = np.stack([np.linalg.norm(g['grad'], axis=-1) for g in gs])
for i, ax in enumerate(axs.flat[:3]):
    ax.imshow(angles[i].T, cmap='twilight', interpolation='nearest')
    indicate_k(pks, i, ax=ax)
axs[1,0].imshow(to_KovesiRGB((magnitudes/np.quantile(magnitudes, 0.99, axis=(1,2), keepdims=True)).T))
axs[1,1].imshow(clipped.T)
axs[1,2].imshow(to_KovesiRGB((gradmags < 0.05).astype(float).T))
# -

pks

fig, ax = plt.subplots(ncols=3, figsize=[18,6])
cutoff = 0.05
for i in range(3):
    ax[i].hist2d(*grads[i][gradmags[i] < cutoff, :].T,
                 weights=(magnitudes[i]**2)[gradmags[i] < cutoff], 
                 bins=50)
    print(np.average(grads[i][gradmags[i] < cutoff, :], weights=(magnitudes[i]**2)[gradmags[i] < cutoff], axis=0), pks[i])
    ax[i].set_aspect('equal')
    indicate_k(pks, i, ax=ax[i], inset=False, origin='lower')
    ax[i].scatter(*np.average(grads[i][gradmags[i] < cutoff, :], weights=(magnitudes[i]**2)[gradmags[i] < cutoff], axis=0), c='red')

cutoff = 0.08
plt.figure(figsize=[12,12])
plt.imshow(to_KovesiRGB(np.where(gradmags < cutoff, (cutoff-gradmags)/cutoff, 0).T))

[g.mean() for g in grads]


