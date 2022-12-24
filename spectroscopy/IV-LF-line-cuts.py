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

# # Twisted bilayer graphene unit cell averaging
#
# In this notebook, we apply unit cell averaging to a spectroscopic LEEM dataset of Twisted Bilayer Graphene (TBG) with a twist angle of $\theta \approx 0.18^\circ$.
# The extracted reflectivity data for different spots in the unit cell is compared to calculated reflectivities for different relative stackings of bilayer graphene.

# +
import dask.array as da
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import numpy as np
import os
from moisan2011 import per
import colorcet  # noqa: F401
import scipy.ndimage as ndi
from itertools import combinations
import xarray as xr
import pandas as pd

from skimage.morphology import erosion, disk, binary_erosion
from dask.distributed import Client, LocalCluster
import ipywidgets as widgets
from ipywidgets import interactive


import pyGPA.geometric_phase_analysis as GPA
import pyGPA.unit_cell_averaging as uc
from pyGPA.imagetools import indicate_k, fftplot, trim_nans, gauss_homogenize2, trim_nans2, generate_mask, cull_by_mask
from pyGPA.mathtools import wrapToPi
import latticegen


from pyL5.analysis.DriftCorrection.StatRegistration import StatRegistration
from pyL5.analysis.CorrectChannelPlate.CorrectChannelPlate import CorrectChannelPlate
from pyL5.lib.analysis.container import Container


# %matplotlib inline

# +
folder = '/mnt/storage-linux/speeldata/20201008-XTBLG02'
names2 = ['20201008_203010_2.3um_445.7_IVhdr_largerFoV',
          '20201008_230111_2.3um_453.9_IVhdrregular_largerFoV_again',
          ]
for name in names2:
    script = CorrectChannelPlate(os.path.join(folder, name + '.nlp'))
    script.start()

data2 = []
EGY = []
MULTIPLIER = []
for name in names2:
    cont = Container(os.path.join(folder, name))
    original = cont.getStack('CPcorrected').getDaskArray()
    data2.append(original.squeeze())
    EGY.append(np.array(cont["EGYM"]))
    MULTIPLIER.append(np.array(cont["MULTIPLIER"]))
for X in [MULTIPLIER, data2, EGY]:
    X[0] = X[0][EGY[0] < 20.]
#data2[0] = data2[0][EGY[0] < 20.]
#EGY[0] = EGY[0][EGY[0] < 20.]
EGY = np.concatenate(EGY)
data2 = da.concatenate(data2, axis=0)
MULTIPLIER = np.concatenate(MULTIPLIER)
nmperpixel = 1.36
# -

cluster = LocalCluster(n_workers=2, threads_per_worker=8, memory_limit='28GB')
client = Client(cluster)
client

# + tags=[]
# data2.to_zarr(os.path.join(folder,names2[-1], 'combined.zarr'), overwrite=True)

# +
# script = StatRegistration(os.path.join(folder,names2[-1]))
# script.start()
# -

cont = Container(os.path.join(folder, names2[-1]))
data3 = cont.getStack('driftcorrectedcombined3').getDaskArray()
data3

smask = generate_mask(data3, 0)
plt.imshow(smask.T)
plt.show()

multiplier = np.array(cont["MULTIPLIER"])
rIV = 15
meanIV = data3[:, 800-rIV:800+rIV, 600-rIV:600+rIV].mean(axis=[1, 2]) / MULTIPLIER
meanIV = meanIV.compute()
meanIV = meanIV / meanIV[:20].mean()
plt.semilogy(EGY, meanIV)

start = np.array((102, 677))
end = np.array((1200, 365))
length = np.linalg.norm(start-end)
xs = np.linspace(start[0], end[0], int(length))
ys = np.linspace(start[1], end[1], int(length))
Es = np.arange(data3.shape[0])
plt.plot(xs, ys, '.')
plt.show()

calcdata = data3.astype(float)  # .compute()

GPAim = np.nanmean(np.where(calcdata[210:220] == 0, np.nan, calcdata[210:220]), axis=0).compute()
GPAcrop = trim_nans2(GPAim)
#GPAim = np.nan_to_num(GPAim)
plt.imshow(GPAcrop.T)
pks, _ = GPA.extract_primary_ks(GPAcrop, plot=True, pix_norm_range=(5, 100))

# +
kw = np.linalg.norm(pks, axis=1).mean() / 4
sigma = 30
kstep = kw / 4
dks = np.zeros_like(pks) + GPA.calc_diff_from_isotropic(pks)

# (re-)generate mask now sigma is known
mask = ~np.isnan(GPAim)
dr = 2*sigma
mask = binary_erosion(mask, disk(dr))
# GPAims is smoothed version of GPAim with mask
GPAims = gauss_homogenize2(GPAim, mask, 50, nan_scale=0)

GPAims = np.where(~mask, 0, GPAims - np.nanmean(GPAims))

# +
pks_iso = pks + dks
    
gs = [GPA.wfr2_grad_opt(GPAims, sigma, 
                        pk[0], pk[1], 
                        kw=kw, kstep=kstep) 
      for pk in pks]
# -

# regenerate smask with the right dr
smask = generate_mask(data3, 0, r=dr)
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(smask.T)
#mask[dr:-dr,dr:-dr] = 1.
ax[1].imshow(mask.T)
phases = np.stack([np.angle(g['lockin']) for g in gs])

wxs = np.array([g['w'][0] for g in gs])
wys = np.array([g['w'][1] for g in gs])

# + tags=[]
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=[16, 12])
for i in range(len(gs)):
    dx = pks[i][0] - gs[i]['w'][0]
    dy = pks[i][1] - gs[i]['w'][1]
    # 2*np.pi*(dx*xx+dy*yy)
    #phase = np.exp(1j*np.angle(gs[i]['lockin']))
    im = axs[2, i].imshow(phases[i][dr:-dr, dr:-dr].T)  # , cmap='twilight', interpolation='none')
    #plt.colorbar(im, ax=axs[0,i])
    axs[0, i].set_title(pks[i])
    indicate_k(pks, i, ax=axs[0, i])
    im = axs[0, i].imshow(np.where(mask, phases[i], np.nan)[dr:-dr, dr:-dr].T, cmap='twilight', interpolation='none')
    im = axs[1, i].imshow(np.where(True,
                                   np.sqrt(gs[i]['w'][0]**2 + gs[i]['w'][1]**2),
                                   np.nan)[dr:-dr, dr:-dr].T,
                          interpolation='none')
    #im = axs[2,i].imshow(dx.T, cmap='twilight', interpolation='none')
    #plt.colorbar(im, ax=axs[2,i])
    axs[1, i].set_title(f"{pks[i][0]:.3f}, {pks[i][1]:.3f}")

# + tags=[]
wadvs = []
for i in range(3):
    gphase = np.moveaxis(gs[i]['grad'], -1, 0)/2/np.pi
    w = gphase + pks_iso[i, :, None, None]
    wadvs.append(w)
wadvs = np.stack(wadvs)

wxs2 = np.concatenate([wadvs[:, 0], -wadvs[:, 0]])
wys2 = np.concatenate([wadvs[:, 1], -wadvs[:, 1]])
#wys = np.concatenate([wadvs[:,1], -wadvs[:,1]])
#wxs = np.clip(wxs, -0.15,0.15)
#wys = np.clip(wys, -0.15,0.15)
fig, axs = plt.subplots(ncols=3, figsize=[26, 8], sharex=True, sharey=True)
axs[0].hist2d(wxs2.ravel(),
              wys2.ravel(),
              bins=500, cmap='cet_fire_r', vmax=1000)
axs[0].set_aspect('equal')
axs[0].set_title(f'sigma={sigma}, kstep={kstep:.4f}')


axs[1].hist2d(wxs2.ravel()[np.stack([mask]*6).ravel()],
              wys2.ravel()[np.stack([mask]*6).ravel()],
              bins=500, cmap='cet_fire_r', vmax=1000)
axs[1].set_aspect('equal')
axs[1].set_title(f'sigma={sigma}, kstep={kstep:.4f}')

p, _ = per(GPAims, inverse_dft=False)
fftim = np.abs(np.fft.fftshift(p))
fftplot(fftim, ax=axs[2], pcolormesh=False, vmax=np.quantile(fftim, 0.9999),
        vmin=np.quantile(fftim, 0.01), cmap='cet_fire_r', interpolation='none',
        origin='lower')


# +


#phases = np.stack([np.angle(g['lockin']) for g in gs])
maskzero = 0.000001
weights = np.stack([np.abs(g['lockin']) for g in gs])*(mask+maskzero)

grads = np.stack([g['grad'] for g in gs])

lxx, lyy = np.mgrid[:phases.shape[1], :phases.shape[2]]
dks = GPA.calc_diff_from_isotropic(pks)
iso_grads = np.stack([g['grad'] - 2*np.pi*np.array([dk[0], dk[1]]) for g, dk in zip(gs, dks)])
iso_grads = wrapToPi(iso_grads)

iso_phases = phases + np.stack([2*np.pi*(dk[0]*lxx+dk[1]*lyy) for dk in dks])

unew_iso = GPA.reconstruct_u_inv_from_phases(pks+dks, iso_phases, weights)
# -

u_inv = GPA.invert_u_overlap(unew_iso)
xxh2, yyh2 = np.mgrid[:unew_iso.shape[1], :unew_iso.shape[2]]
reconstructed = ndi.map_coordinates(np.nan_to_num(GPAim), [xxh2+u_inv[0], yyh2+u_inv[1]], cval=np.nan)
reconstructed2 = ndi.map_coordinates(np.nan_to_num(GPAim), [xxh2+unew_iso[0], yyh2+unew_iso[1]], cval=np.nan)

# +
smask2 = (np.where(smask, data3[70], np.nan) > 4.9e4).compute()  # Yes, this is a magic value

fig, ax = plt.subplots(ncols=2, figsize=[9, 4.5*3/4], sharey=True, sharex=True)
ax[0].imshow(cull_by_mask(np.where(smask, GPAim, np.nan), smask).T, cmap='gray')
ax[0].imshow(cull_by_mask(np.where(~smask2 & smask, GPAim, np.nan), smask).T)  # , cmap='gray')
reconstructed3 = ndi.map_coordinates(np.nan_to_num(np.where(smask, GPAim, np.nan)),
                                     [xxh2+unew_iso[0], yyh2+unew_iso[1]],
                                     cval=np.nan)
reconstructed3 = np.where(reconstructed3 < 100.0, np.nan, reconstructed3)
im = ax[1].imshow(trim_nans(reconstructed3).T, cmap='gray')
u_culled = cull_by_mask(np.where(smask[None], unew_iso, np.nan), smask)
im.set_extent(np.array(im.get_extent()) + np.array([np.nanmin(u_culled[1, :, 0])]*2 + [np.nanmin(-u_culled[0, 0])]*2))

ax[1].autoscale_view()
ax[1].set_xlim([None, 1020])
ax[1].set_ylim([700, None])
a = 50

xx, yy = np.mgrid[a//2:u_culled.shape[1]:a, a//2:u_culled.shape[2]:a]
u_culled = -u_culled[:, a//2::a, a//2::a]

# TODO: check directions!!!
ax[0].quiver(xx, yy, *u_culled, color='green',
             angles='xy', scale_units='xy')  # , pivot='tip')
for a, l in zip(ax, 'ab'):
    a.set_title(l, loc='left', fontweight='bold')
    a.set_xlabel('x (pixels)')
ax[0].set_ylabel('y (pixels)')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'unitcellaveragintop.pdf'), dpi=300)
# -

lpks = (pks+dks)[:2]

recon = trim_nans2(np.where(reconstructed < 100, np.nan, reconstructed))
ucellim2 = uc.unit_cell_average(recon, lpks, z=1)
uc_averaged = uc.expand_unitcell(ucellim2, lpks, recon.shape, z=1)
fig, ax = plt.subplots(ncols=2, figsize=[15, 8])
ax[0].imshow(recon.T)
ax[1].imshow(uc_averaged.T,
             vmax=np.nanquantile(uc_averaged, 0.99),
             vmin=np.nanquantile(uc_averaged, 0.01))
indicate_k((pks+dks), 0, ax=ax[1])
indicate_k((pks+dks), 1, ax=ax[1])


# +
def generate_cut(lpks, z, rshift=np.array([0, 0]), direction=np.array([1, 1]), npoints=200):
    a_0 = np.linspace(0, 1, npoints)
    ucellcoords = np.stack([a_0, a_0], axis=-1) * direction
    rmin, rsize = uc.calc_ucell_parameters(lpks, z)
    scattercoords0 = z*(uc.backward_transform(ucellcoords, lpks).T)
    scattercoords = z*uc.cart_in_uc(rshift/z + rmin + scattercoords0.T, lpks, rmin=rmin).T
    return scattercoords


def find_shift(ucellim, sigma=0):
    mask = ~np.isnan(ucellim)
    mask = np.pad(mask, 1)
    mask = erosion(mask, selem=disk(3))[1:-1, 1:-1]
    if sigma == 0:
        argmin = np.argmin(np.where(mask, ucellim, np.nanmax(ucellim)))
    else:
        smooth = ndi.filters.gaussian_filter(np.where(mask, ucellim, np.nanmax(ucellim)), sigma=sigma)
        argmin = np.argmin(smooth)
    index = np.array(np.unravel_index(argmin, ucellim.shape))
    return index


# -

z = 1
fig, ax = plt.subplots(ncols=3, figsize=[18, 6])
for i, lpks in enumerate(combinations(pks+dks, 2)):
    lpks = np.array(lpks)
    lpks[0] = lpks[0]*np.sign(np.dot(*lpks))*-1
    rmin, rsize = uc.calc_ucell_parameters(lpks, z)
    ucellim = uc.unit_cell_average(np.where(mask, GPAim, np.nan),
                                   np.array(lpks),
                                   u=-unew_iso, z=z)  # unit_cell_average_distorted(GPAim, , z=1)
    ax[i].imshow(ucellim.T, cmap='gray',
                 vmax=np.nanquantile(ucellim, 0.99), vmin=np.nanquantile(ucellim, 0.001))
    for direction in np.array([[1, 1], [1, -2], [-2, 1]]):
        scattercoords = generate_cut(lpks, z, find_shift(ucellim, sigma=2), direction=direction)
        ax[i].scatter(*scattercoords, alpha=0.3)

smask2 = (np.where(smask, data3[70], np.nan) > 4.9e4).compute()
plt.figure(figsize=[12, 12])
plt.imshow(np.where(smask2, data3[210], np.nan).T)
plt.imshow(np.where(~smask2, data3[210], np.nan).T, cmap='gray')

maskedstack = cull_by_mask(da.where(smask2, data3.astype(float), np.nan), smask2)


# +
def stack_apply(images, inner_func, u):
    """
    Helper function to apply uc.unit_cell_average to a stack of images
    using da.map_blocks.
    First homogenizes the image.
    """
    if images.ndim == 3:
        for image in images:
            res = [inner_func(gauss_homogenize2(image, cull_by_mask(smask2, smask2), sigma=40), u) for i in images]
        res = np.stack(res)
    elif images.ndim == 2:
        res = inner_func(images, u, z)
    return res


def unit_cell_average_stack(images, ks, u, z=1):
    """Average images with a distortion u over all it's unit cells
    using a drizzle like approach, scaling the unit cell
    up by a factor z.
    Return an array containing the unit cell
    and the corresponding weight distrbution."""
    if images.ndim == 3:
        for image in images:
            res = [uc.unit_cell_average(i, ks, u, z) for i in images]
        res = np.stack(res)
    elif images.ndim == 2:
        res = uc.unit_cell_average(i, ks, u, z)[None]
    return res


# -

maskedu = cull_by_mask(unew_iso, smask2)

z = 3
ucellim = uc.unit_cell_average(maskedstack[100].compute(), (pks+dks)[:2], -maskedu, z=z)
func = uc.unit_cell_average(maskedstack[100], (pks+dks)[:2], -maskedu, z=z, only_generate_func=True)
ucellstack = da.map_blocks(stack_apply,
                           maskedstack, dtype=ucellim.dtype,
                           chunks=(1,)+ucellim.shape,
                           inner_func=func,
                           u=np.moveaxis(-maskedu, 0, -1))

corners = np.array([[0., 0.],
                    [0., 1.],
                    [1., 0.],
                    [1., 1.]])
cornervals = uc.backward_transform(corners, (pks+dks)[:2])
plt.scatter(*cornervals.T)
plt.gca().set_aspect('equal')
slicelength = np.linalg.norm(cornervals[-1]-cornervals[0])*nmperpixel

test = ucellstack[220].compute()
plt.figure(figsize=[8, 5])
plt.imshow(test.T, cmap='gray',
           vmax=np.nanquantile(test, 0.99), vmin=np.nanquantile(test, 0.01))

ucellstackc = ucellstack.compute()

stackingcolors = dict(AB='C3', BA='C3', SP='C4', AA='C5')

# +
index = 219
test2 = ucellstackc[index]  # .mean(axis=0)
fig, (ax, ax2) = plt.subplots(ncols=2, figsize=[9, 3])
im = ax.imshow(test2.T, vmax=np.nanquantile(test2, 0.99), vmin=np.nanquantile(test2, 0.01), cmap='gray')
slices = []
directions = np.array([[1, 1],
                       [2, -1],
                       [-1, 2]])
for i, direction in enumerate(directions/z):
    scattercoords = generate_cut((pks+dks)[:2], z, find_shift(test2, sigma=5),
                                 direction=direction, npoints=700)
    #ax.scatter(*scattercoords, alpha=0.8, cmap='plasma', c=np.arange(scattercoords.shape[1]),)
    ax.scatter(*scattercoords, alpha=0.8, color=f'C{i}', s=3)
    Es = np.arange(data3.shape[0])
    coords = np.broadcast_arrays(Es[:, None], scattercoords[[0], :], scattercoords[[1], :])
    newEslice = ndi.map_coordinates(np.nan_to_num(ucellstackc), coords)
    slicepos = np.linspace(0, slicelength, len(newEslice[index]))
    for loc, label in zip([-1, -1/3, 0, 1/3], ['AA', 'AB', 'SP', 'BA']):
        if i == 0 or label == 'SP':
            ax.annotate(label, scattercoords[:, int((loc+1)*700/2)-1],
                        fontweight='bold', ha="center", va="center",
                        # color=stackingcolors[label],
                        color='white',
                        bbox=dict(boxstyle="square,pad=0.2", lw=0, fc=stackingcolors[label]))
            if i == 0:
                ax2.annotate(label, [slicepos[int((loc+1)*700/2)-1], newEslice[index, int((loc+1)*700/2)-1]],
                             fontweight='bold', ha="center", va="center",
                             # color=stackingcolors[label],
                             color='white',
                             bbox=dict(boxstyle="square,pad=0.2", lw=0, fc=stackingcolors[label], alpha=0.7))

    ax2.scatter(slicepos, newEslice[index], s=10, alpha=0.7, marker='.', linewidths=0)
    slices.append(newEslice)

ax2.margins(x=0)
ax2.set_xlabel('Position along slice (nm)')
ax2.set_ylabel('Intensity (a.u.)')
#plt.colorbar(im, ax=ax)
ax.set_xlabel(f'x ({z} $\\times$ zoomed pix)')
ax.set_ylabel(f'y ({z} $\\times$ zoomed pix)')
ax.set_title(f'Average unit cell $E_0={EGY[index]:.1f}$eV')
ax.set_title('c', loc='left', fontweight='bold')
ax2.set_title('d', loc='left', fontweight='bold')

ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.tight_layout()
#plt.savefig(os.path.join('plots', 'unitcellaveraginbottom.pdf'), dpi=300)
# -

# ## Loading calculated reflectivity
# * Ab-initio scattering data available from https://doi.org/10.4121/16843510
# * TensorLEED data is from Hibino et al. PRB 80 085406 (2009) https://doi.org/10.1103/PhysRevB.80.085406

# +
ds = xr.load_dataset('../data/stacking_reflectivity_calculations_bilayer_graphene.nc')

xsmooth = ndi.gaussian_filter1d(ds.Reflectivity.data, sigma=1, axis=1)

pldat = np.log(xsmooth/xsmooth[8])
pldat = np.concatenate([pldat[9:][::-1], pldat], axis=0)
Krasovenergy = (ds.Energy.data-5)*1.04
pltenergies = [3.5, 14.2, 17, 19.5, 30, 32.5, 40, 44.8, 47, 53]
ds

# +
plt.subplots(figsize=[7.5, 4], constrained_layout=True)
plt.matshow(pldat,  # [:,:-140],
            aspect=2.5,
            extent=[Krasovenergy[0], Krasovenergy[-1], 8.5, -16.5], fignum=0, cmap='seismic',
            vmin=-2, vmax=2)
plt.colorbar(label=r'log ($I/{I_{AB}}$)')
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))

for e in pltenergies:
    plt.axvline(e, color='black', alpha=0.6)
plt.grid(which='major', axis='y')
plt.yticks([-16, -12, -8, -4, 0, 4, 8], ['AA', '', 'AC', 'SP', 'AB', '', 'AA'])
plt.xlabel('$E_0$')
# -

hibinofolder = '/mnt/storage-linux/stack/Promotie/data/hibino-data'
files = ['AA', 'AB', 'ABC', 'Boundary']
hibino = {}
for key in files:
    hibino[key] = pd.read_csv(os.path.join(hibinofolder, key+'.txt'), engine='python', delimiter=r"\s+")
hibino['SP'] = hibino['Boundary']


# +
def ks_to_avecs(ks):
    norms = np.linalg.norm(ks, axis=1)
    avecs = latticegen.transformations.rotate(ks/norms[:, None],
                                              angle=-np.pi/2)
    avecs = avecs * latticegen.transformations.r_k_to_a_0(norms)[:, None]
    return avecs

stackingcolors = dict(AB='C3', BA='C3', SP='C4', AA='C5')

fig, axs = plt.subplots(2, figsize=[9, 7], sharex=True)

for ind, l in zip([-8., -4., 8.], ['AB', 'SP', 'AA']):
    axs[0].semilogy(Krasovenergy, ds.Reflectivity.sel(dx=ind).data.T,
                    label=l, color=stackingcolors[l])
    axs[1].semilogy((hibino[l]['Energy']-11), hibino[l]['(0,0)']*2,
                    label=l, color=stackingcolors[l])
axs[0].semilogy(EGY, meanIV, '.', markersize=1, label='mean exp', color='black')
axs[1].semilogy(EGY, meanIV, '.', markersize=1, label='mean exp', color='black')
axs[1].legend(numpoints=8)
for a in axs:
    a.set_xlim(0, hibino[l]['Energy'].max()-11)
axs[0].set_title('Ab-initio scattering')
axs[1].set_title('TensorLEED')
axs[1].set_xlabel('$E_0$ (eV)')

for a, l in zip(axs, 'ac'):
    a.set_title(l, loc='left', fontweight='bold')
    a.xaxis.set_minor_locator(ticker.MultipleLocator(5))

ax = axs[0].inset_axes([105/155, 0., (155-105)/155, 1])

S = 1000
r_k = 0.065 * 500/S

theta = -4.5
xi = 0.
ks1 = latticegen.generate_ks(r_k, xi)[:-1]
ks2 = latticegen.generate_ks(r_k, theta+xi)[:-1]
shift = np.array([0, -200*S/500])
lattice1 = 0.7*latticegen.hexlattice_gen(r_k, xi, 1, shift=shift, size=S)
lattice2 = latticegen.hexlattice_gen(r_k, theta+xi, 1, shift=shift, size=S)
lattice1 = np.clip(lattice1 / lattice1.max(), 0, 1)
lattice2 = np.clip(lattice2 / lattice2.max(), 0, 1)
bicmap = 'BrBG_r'
#fig, ax = plt.subplots(figsize=[5,5])
ax.imshow(-lattice1.T, cmap=bicmap,
          vmax=1,
          vmin=-1,
          alpha=lattice1.T
          )
ax.imshow(lattice2.T, cmap=bicmap,
          vmax=1,
          vmin=-1,
          alpha=lattice2.T)
avecs = ks_to_avecs(ks2-ks1)
center = np.array(lattice1.shape)//2 - 0.5 - shift

ucell = mpl.patches.Polygon(center+np.stack([[0, 0],
                                             avecs[1],
                                             avecs[2] + avecs[1],
                                             avecs[2]]),
                            closed=True,
                            fc='none', ec='gray', lw=3)
ax.add_patch(ucell)
for a in avecs[[2, 1]]:
    ax.arrow(*center, *a,
             length_includes_head=True,
             # width=0.02,
             # linewidth=0,
             head_width=10,
             color='black'
             )


for i, label in enumerate(['AA', 'AB', 'BA', 'AA']):
    ax.annotate(label, center + i/3*(avecs[2]+avecs[1]),
                fontweight='bold', ha="center", va="center",
                # color=stackingcolors[label],
                color='white',
                bbox=dict(boxstyle="square,pad=0.2",
                          lw=0,
                          fc=stackingcolors[label],
                          alpha=0.7))
for vec in [avecs[1], avecs[2], avecs[2]+avecs[1]]:
    ax.annotate('SP', center + 0.5*(vec),
                fontweight='bold', ha="center", va="center",
                # color=stackingcolors[label],
                color='white',
                bbox=dict(boxstyle="square,pad=0.2",
                          lw=0,
                          fc=stackingcolors['SP'],
                          alpha=0.7))
ax.set_xlim(20, S-20)
ax.set_axis_off()
ax.set_title('b', loc='left', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'stackingtheorycurves.pdf'), dpi=600)

# +
fig, axs = plt.subplots(2, figsize=[6, 5.5])

for ind, l in zip([-8., -4., 8.], ['AB', 'SP', 'AA']):
    axs[1].semilogy(Krasovenergy, ds.Reflectivity.sel(dx=ind).data.T,
                    label=l, color=stackingcolors[l])
    #axs[1].semilogy((hibino[l]['Energy']-11), hibino[l]['(0,0)']*2,
    #                label=l, color=stackingcolors[l])
axs[1].semilogy(EGY, meanIV, '.', markersize=1, label='mean exp', color='black')
axs[1].legend(numpoints=8)

axs[1].set_title('Ab-initio scattering')
#axs[1].set_title('TensorLEED')
axs[1].set_xlabel('$E_0$ (eV)')

for a, l in zip(axs, 'ab'):
    a.set_title(l, loc='left', fontweight='bold')
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
axs[1].set_xlim(-5,100)

ax = axs[0] #axs[0].inset_axes([105/155, 0., (155-105)/155, 1])

r_k = 0.065
theta = -3.
xi = 0.
ks1 = latticegen.generate_ks(r_k, xi)[:-1]
ks2 = latticegen.generate_ks(r_k, theta+xi)[:-1]
shift = np.array([250, -150])
lattice1 = 0.7*latticegen.hexlattice_gen(r_k, xi, 1, shift=shift, size=(1250,500))
lattice2 = latticegen.hexlattice_gen(r_k, theta+xi, 1, shift=shift, size=(1250,500))
lattice1 = np.clip(lattice1 / lattice1.max(), 0, 1)
lattice2 = np.clip(lattice2 / lattice2.max(), 0, 1)
bicmap = 'PRGn'
#fig, ax = plt.subplots(figsize=[5,5])
ax.imshow(-lattice1.T, cmap=bicmap,
          vmax=1,
          vmin=-1,
          alpha=lattice1.T
          )
ax.imshow(lattice2.T, cmap=bicmap,
          vmax=1,
          vmin=-1,
          alpha=lattice2.T)
avecs = ks_to_avecs(ks2-ks1)
center = np.array(lattice1.shape)//2 - 0.5 - shift

iv = 3
iv2 = 2
ucell = mpl.patches.Polygon(center+np.stack([[0, 0],
                                             avecs[iv2],
                                             avecs[iv] + avecs[iv2],
                                             avecs[iv]]),
                            closed=True,
                            fc='none', ec='gray', lw=3)
ax.add_patch(ucell)
for a in avecs[[iv, iv2]]:
    ax.arrow(*center, *a,
             length_includes_head=True,
             # width=0.02,
             # linewidth=0,
             head_width=10,
             color='black'
             )

stackingcolors = dict(AB='C3', BA='C3', SP='C4', AA='C5')
for i, label in enumerate(['AA', 'AB', 'BA', 'AA']):
    ax.annotate(label, center + i/3*(avecs[iv]+avecs[iv2]),
                fontweight='bold', ha="center", va="center",
                # color=stackingcolors[label],
                color='white',
                bbox=dict(boxstyle="square,pad=0.2",
                          lw=0,
                          fc=stackingcolors[label],
                          alpha=0.5))
for vec in [avecs[iv2], avecs[iv], avecs[iv]+avecs[iv2]]:
    ax.annotate('SP', center + 0.5*(vec),
                fontweight='bold', ha="center", va="center",
                # color=stackingcolors[label],
                color='white',
                bbox=dict(boxstyle="square,pad=0.2",
                          lw=0,
                          fc=stackingcolors['SP'],
                          alpha=0.5))
#ax.set_xlim(10, 500-10)
ax.set_axis_off()
#ax.set_title('b', loc='left', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'stackingtheorycurves_no_hibino.pdf'), dpi=300)

# +
f, ax = plt.subplots(nrows=4, figsize=[9, 6], sharex=True, constrained_layout=True)
for i, newEslice in enumerate(slices):
    # Use MM to crop out edge outliers
    X = newEslice[:10].mean(axis=0, keepdims=True)
    #X = np.abs(X-X.mean()) > 100
    newEslice = newEslice / X
    # plt.figure(figsize=[12,4])
    ldat = np.log(newEslice / newEslice[:, 230:236].mean(axis=1, keepdims=True))
    print(np.diff(np.quantile(ldat, [0.01, 0.99], axis=1), axis=0).max())
    im = ax[i].imshow(  # np.concatenate([ldat]*2, axis=1).T,
        ldat.T,
        aspect='auto', extent=[EGY[0], EGY[-1], -1, 1],
        cmap='seismic', vmax=0.3, vmin=-0.3, interpolation='none')
    ax[i].set_yticks([-1, -1/3, 0, 1/3, 1], ['AA', 'AB', 'SP', 'BA', 'AA'], fontweight='bold')
    for axis in ['left', 'right']:
        ax[i].spines[axis].set_linewidth(4)
        ax[i].spines[axis].set_color(f'C{i}')
    ax[i].set_title('abcd'[i], fontweight='bold', loc='left')
    ax2 = ax[i].twinx()
    ax2.set_ylim(-0.5*slicelength, 0.5*slicelength)
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(30))
    if i == 1:
        ax2.set_ylabel('nm (approx.)')
plt.colorbar(im, ax=ax[:3], label=r'log ($I/{I_{AB}}$)')
print(pldat[:, :-140].max(), pldat[:, :-140].min())
im = ax[3].imshow(pldat[:, :-140], aspect='auto',
                  extent=[Krasovenergy[0], Krasovenergy[-140], 8.5, -16.5],
                  cmap='seismic',
                  vmin=-2, vmax=2)
plt.colorbar(im, ax=ax[3], label=r'log ($I/{I_{AB}}$)', extend='both', aspect=6)
ax[3].xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax[3].xaxis.set_major_locator(ticker.MultipleLocator(10))
ax[3].set_title('d', fontweight='bold', loc='left')
# pltenergies = [3.5, 14.2, 17,19.5,30,32.5, 40, 44.8, 47,53]
# for e in pltenergies:
#     ax[3].axvline(e, color='black', alpha=0.6)
ax[3].grid(which='major', axis='y')

ax[3].set_yticks([-16, -8, -4, 0, 8], ['AA',  'BA', 'SP', 'AB', 'AA'], fontweight='bold')
ax[3].set_xlabel('$E_0$ (eV)')
ax[3].set_title('Theory')
ax[0].set_title('Slices of unit cell averaged data')
# plt.tight_layout()
plt.savefig(os.path.join('plots', 'unitcellaveragedslices.pdf'))#, dpi=300)
# -

fig, ax = plt.subplots(ncols=3, figsize=[9, 3], sharey=True, constrained_layout=True)
ax[1].fill_between([-0.3, 0.3], [15, 15], [38, 38], color='C3', alpha=0.2, zorder=-10)
ax[1].fill_between([-0.3, 0.3], [15, 15], [-10, -10], color='C4', alpha=0.2, zorder=-10)
for i, newEslice in enumerate(slices):
    # Use MM to crop out edge outliers
    X = newEslice[:10].mean(axis=0, keepdims=True)
    #X = np.abs(X-X.mean()) > 100
    newEslice = newEslice / X
    ldat = np.log(newEslice / newEslice[:, 230:236].mean(axis=1, keepdims=True))
    cmax = 0.25
    res = np.apply_along_axis(lambda a: np.histogram(a, bins=100, range=(-cmax, cmax))[0], 0, ldat)
    ax[i].imshow(  # res.T,
        np.where(res > 0, res, np.nan).T,
        aspect='auto',
        vmax=80,
        extent=[-cmax, cmax, -0.5*slicelength, 0.5*slicelength, ],
        cmap='cet_fire_r', interpolation='none')
    ax[i].yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax[i].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[i].set_xlabel(r'log ($I/{I_{AB}}$)')
    for axis in ['left', 'right']:
        ax[i].spines[axis].set_linewidth(4)
        ax[i].spines[axis].set_color(f'C{i}')
ax[1].axhline(38, color='C3', alpha=0.8, linewidth=1.5, zorder=-10)
ax[1].annotate("", (-0.17, 38), (-0.17, 15),
               arrowprops=dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, color='black'))
ax[1].axhline(15, color='C4', alpha=0.5, linewidth=1.5, zorder=-10)
ax[1].axhline(15, color='C3', alpha=0.5, linewidth=1.5, zorder=-10)
ax[1].axhline(-10, color='C4', alpha=0.8, linewidth=1.5, zorder=-10)
ax[1].annotate("", (-0.2, -10), (-0.2, 15),
               arrowprops=dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, color='black'))
ax[0].set_ylabel('nm (approx.)')
ax2 = ax[2].twinx()
ax2.set_yticks([-1, -1/3, 0, 1/3, 1], ['AA', 'AB', 'SP', 'BA', 'AA'], fontweight='bold')
ax2.tick_params(
    right=False,
    labelright=True)
ax2.set_ylim(-1, 1)
# plt.tight_layout()
plt.savefig(os.path.join('plots', 'unitcellaveragedslicesprojected.pdf'))

# +
fig, ax = plt.subplots(ncols=1, figsize=[4., 3.], sharey=True, constrained_layout=True)
ax = [ax]
ax[0].fill_between([-0.3, 0.3], [15, 15], [38, 38], color='C3', alpha=0.2, zorder=-10)
ax[0].fill_between([-0.3, 0.3], [15, 15], [-10, -10], color='C4', alpha=0.2, zorder=-10)

# Use MM to crop out edge outliers
X = slices[1][:10].mean(axis=0, keepdims=True)
#X = np.abs(X-X.mean()) > 100
newEslice = slices[1] / X
ldat = np.log(newEslice / newEslice[:, 230:236].mean(axis=1, keepdims=True))
cmax = 0.25
res = np.apply_along_axis(lambda a: np.histogram(a, bins=100, range=(-cmax, cmax))[0], 0, ldat)
im = ax[0].imshow(  # res.T,
    np.where(res > 0, res, np.nan).T,
    aspect='auto',
    vmax=80, 
    vmin=0,
    extent=[-cmax, cmax, -0.5*slicelength, 0.5*slicelength, ],
    cmap='cet_fire_r', interpolation='none')
plt.colorbar(im, label='counts', extend='max')
ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(10))
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax[0].set_xlabel(r'log ($I/{I_{AB}}$)')
for axis in ['left', 'right']:
    ax[0].spines[axis].set_linewidth(4)
    ax[0].spines[axis].set_color(f'C{1}')
ax[0].axhline(38, color='C3', alpha=0.8, linewidth=1.5, zorder=-10)
ax[0].annotate("", (-0.17, 38), (-0.17, 15),
               arrowprops=dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, color='black'))
ax[0].axhline(15, color='C4', alpha=0.5, linewidth=1.5, zorder=-10)
ax[0].axhline(15, color='C3', alpha=0.5, linewidth=1.5, zorder=-10)
ax[0].axhline(-10, color='C4', alpha=0.8, linewidth=1.5, zorder=-10)
ax[0].annotate("", (-0.2, -10), (-0.2, 15),
               arrowprops=dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, color='black'))
ax[0].set_ylabel('nm (approx.)')
ax2 = ax[0].twinx()
ax2.set_yticks([-1, -1/3, 0, 1/3, 1], ['AA', 'AB', 'SP', 'BA', 'AA'], fontweight='bold')
ax2.tick_params(
    right=False,
    labelright=True)
ax2.set_ylim(-1, 1)
plt.savefig(os.path.join('plots', 'unitcellaveragedslicesprojected_reduced.pdf'))
# -


