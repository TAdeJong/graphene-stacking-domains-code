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

# # Comparison of stacking contrast in different bilayer graphene systems
#
# In this notebook stacking domains in quasi-freestanding bilayer graphene on silicon carbide (QFBLG), graphene on SiC (MLG), 1-on-1 twisted graphene and 1-on-2 twisted bilayer graphene are compared.

# +
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
import dask.array as da
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import os


import colorcet  # noqa: F401
import scipy.ndimage as ndi

from pyL5.analysis.DriftCorrection.StatRegistration import StatRegistration
from pyL5.analysis.CorrectChannelPlate.CorrectChannelPlate import CorrectChannelPlate
from pyL5.lib.analysis.container import Container
from dask.distributed import Client, LocalCluster
from pyGPA.imagetools import generate_mask, cull_by_mask

from skimage.morphology import erosion, disk

# %matplotlib inline
# -

rfolder = '/mnt/storage-linux/'
names = ['domainsdata/2017-11-20-QFBLG2/20171120_160356_3.5um_591.4_IVhdr',
         'domainsdata/2017-11-24-QFBLG2/20171124_115107_3.5um_577.8_IVhdr_BF',
         'speeldata/20200714-XTBLG02/20200714_111905_5.7um_494.7_IVhdr_layercounts',
         'speeldata/20201008-XTBLG02/20201008_220142_5.7um_444.6_IVhdr_multilayer',
         ]
for name in names:
    script = CorrectChannelPlate(os.path.join(rfolder, name + '.nlp'))
    script.start()

cluster = LocalCluster(n_workers=1, threads_per_worker=8, memory_limit='24GB')
client = Client(cluster)
client

for name in names:
    script = StatRegistration(os.path.join(rfolder, name + '.nlp'))
    script.start()

conts = [Container(os.path.join(rfolder, name + '.nlp')) for name in names]
data = [cont.getStack('driftcorrected.zarr').getDaskArray() for cont in conts]
multipliers = [np.array(cont["MULTIPLIER"]) for cont in conts]
energies = [np.array(cont["EGYM"]) for cont in conts]
energies[1] = energies[1] + conts[1]["mirror"][0] - conts[0]["mirror"][0]

conts[1]["mirror"][0] - conts[0]["mirror"][0]

smasks = [generate_mask(dat, 0) for dat in data]

cdata = [cull_by_mask(dat, smask) for dat, smask in zip(data, smasks)]
cdata[3] = cdata[2]
cmasks = [cull_by_mask(smask, smask) for smask in smasks]
cmasks[3] = cmasks[2]

nmperpixels = [2.23, 2.23, 3.7, 3.7]
labels = ['QFBLG', 'EMLG', '1-on-2 TBG', '1-on-1 TBG']

# +
pltEGY = [4.2, 17.3, 29.2, 30.4, 37.4, 45.7]
fig, axs = plt.subplots(2, len(pltEGY), figsize=[9, 3.4], constrained_layout=True)

for j in range(2):
    for i in range(len(pltEGY)):
        index = np.argmin(np.abs(energies[j] - pltEGY[i]))
        axs[j, i].imshow(cdata[j][index, 200+30*j:600+30*j, 300+50*j:700+50*j].T.compute(),
                         cmap='gray', interpolation='none')
        if j == 0:
            axs[j, i].set_title(f'$E_0 = {pltEGY[i]:.1f}$ eV')
        axs[j, i].tick_params(
            bottom=False,      # ticks along the bottom edge are off
            left=False,
            labelbottom=False,
            labelleft=False)

scalebar = ScaleBar(nmperpixels[0], "nm", length_fraction=0.3,
                    location="lower right", box_alpha=0.3,
                    width_fraction=0.03, color='white'
                    )
axs.flat[-1].add_artist(scalebar)

for i, ax in enumerate(axs.flat):
    ax.set_title('abcdefghijkl'[i], fontweight='bold', loc='left')

plt.savefig(os.path.join('plots', 'EGYslices.pdf'))
# -

3.7/2.23, 4.45/2.73

# +
fig, axs = plt.subplots(2, 2, figsize=[18, 15])

slicelims = [np.array([[400, 480], [600, 400]]),
             #np.array([[530,570], [650,450]]),
             np.array([[520, 550], [640, 430]]),
             #np.array([[480,480], [570,520]]),
             np.array([[485, 470], [575, 510]]),
             #np.array([[850,230], [1080,330]])
             np.array([[855, 230], [1085, 330]])
             ]

# equalize all lengths
sla = 160  # half the length in nm
newlims = []
for i, lims in enumerate(slicelims):
    center = (lims[0]+lims[1]) / 2
    direc = (lims[0]-lims[1]) / (np.linalg.norm(lims[0]-lims[1]) * nmperpixels[i])
    newlims.append(np.array([center+direc*sla, center-direc*sla]))

slicelims = newlims  # slicelims are now limits in pixels for a slice of 2*sla nm

coords = []
lengths = []
for i, ax, dat in zip([0, 1, 2, 3], axs.flat, cdata):
    ax.imshow(np.where(cmasks[i].T, dat[dat.shape[0]//4].T.compute(), np.nan))
    # ax.imshow(dat[3*dat.shape[0]//4].T.compute(),
    ax.imshow(dat[509].T.compute(),
              cmap='inferno', alpha=0.6)
    ax.grid()

for i in range(4):
    start, end = slicelims[i]
    length = np.linalg.norm(start-end)*2
    lengths.append(length)
    xs = np.linspace(start[0], end[0], int(length))
    ys = np.linspace(start[1], end[1], int(length))
    Es = np.arange(cdata[i].shape[0])
    coords.append(np.broadcast_arrays(Es[:, None], xs[None, :], ys[None, :]))
    axs.flat[i].plot(xs, ys)

lengths = np.array(lengths)  # lengths is 2 times the length in pixels

# +
fix, ax = plt.subplots(ncols=3, figsize=[9, 4])

for i in range(3):
    index = np.argmin(np.abs(energies[i] - 40))
    im = ax[i].imshow(cdata[i][index].T.compute(), cmap='gray')
    start, end = slicelims[i]*np.array(conts[i]["NMPERPIXEL"])[0]
    im.set_extent(np.array(im.get_extent())*np.array(conts[i]["NMPERPIXEL"])[0])

    xs = np.linspace(start[0], end[0], int(lengths[i]))
    ys = np.linspace(start[1], end[1], int(lengths[i]))
    ax[i].plot(xs.squeeze(), ys.squeeze(), color=f'C{i}')
# -

Eslices = [ndi.map_coordinates(cdata[i], c) for i, c in enumerate(coords)]

# +
fix, ax = plt.subplots(2, 2, figsize=[4.1, 4.5], constrained_layout=True)
ax = ax.flat


for i in range(3):
    index = np.argmin(np.abs(energies[i] - 38))
    print(index)
    im = ax[i].imshow(cdata[i][index].T.compute(), cmap='gray')
    start, end = slicelims[i] * nmperpixels[i] / 1e3
    im.set_extent(np.array(im.get_extent()) * nmperpixels[i] / 1e3)  # np.array(conts[i]["NMPERPIXEL"])[0]/1e3)

    xs = np.linspace(start[0], end[0], int(lengths[i]))
    ys = np.linspace(start[1], end[1], int(lengths[i]))
    ax[i].plot(xs, ys, color=f'C{i}', linewidth=3, alpha=0.8)
    r = 0.5
    ax[i].set_xlim(np.array([start, end]).mean(axis=0)[0]-r, np.array([start, end]).mean(axis=0)[0]+r)
    ax[i].set_ylim(np.array([start, end]).mean(axis=0)[1]+r, np.array([start, end]).mean(axis=0)[1]-r)


index = np.argmin(np.abs(energies[i] - 40))
im = ax[3].imshow(cdata[2][index].T.compute(), cmap='gray')

start, end = slicelims[3] * nmperpixels[i] / 1e3
im.set_extent(np.array(im.get_extent()) * nmperpixels[i] / 1e3)

xs = np.linspace(start[0], end[0], int(lengths[3]))
ys = np.linspace(start[1], end[1], int(lengths[3]))
ax[3].plot(xs, ys, color=f'C{3}', linewidth=3, alpha=0.7)
r = 0.5
ax[3].set_xlim(np.array([start, end]).mean(axis=0)[0]-r, np.array([start, end]).mean(axis=0)[0]+r)
ax[3].set_ylim(np.array([start, end]).mean(axis=0)[1]+r, np.array([start, end]).mean(axis=0)[1]-r)

scalebar = ScaleBar(1e3, "nm", length_fraction=0.25,
                    location="lower left", box_alpha=0.1,
                    width_fraction=0.03
                    )
ax[3].add_artist(scalebar)

for i in range(4):
    ax[i].set_title(labels[i])
    ax[i].tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
    ax[i].set_title('abcd'[i], fontweight='bold', loc='left')


plt.savefig(os.path.join('plots', 'multislice_locs.pdf'))
# -

[np.array(cont["NMPERPIXEL"])[0] for cont in conts]

# +
fig, axs = plt.subplots(nrows=5, figsize=[9, 8], constrained_layout=True, sharex=True,
                        gridspec_kw={'height_ratios': [1.3, 1, 1, 1, 1]})
Inorms = []

meanspectra = []
vmax = 0.12
for i, Eslice in enumerate(Eslices[:3]):
    x = np.linspace(-1, 1, Eslice.shape[1])
    res = np.polyfit(x, Eslice.T, 1)
    #Inormnew = (Eslice-res[0][:,None]*x[None])
    Inormnew = Eslice / (res[0][:, None]*x[None] + res[1][:, None])
    meanspectrum = res[1]  # Inormnew.mean(axis=1, keepdims=True)
    #Inormnew = Inormnew / meanspectrum
    meanspectra.append(meanspectrum.squeeze() / multipliers[i])
    Inorm = Inormnew  # Eslices[1] / Eslices[1].mean(axis=1, keepdims=True)
    Inorms.append(Inorm)
    length = lengths[i]/4*nmperpixels[i]  # np.array(conts[i]["NMPERPIXEL"])[0]
    print(np.abs(np.log(Inorm)).max())
    im = axs[i+1].imshow(np.log(Inorm).T,
                         aspect='auto', extent=[energies[i][0], energies[i][-1], -length, length],
                         vmax=vmax, vmin=-vmax,
                         cmap='PuOr_r',
                         # vmax=1.2, vmin=0.85
                         )
cbar = plt.colorbar(im, ax=axs[1:],
                    label=r'log(I / $\langle I\rangle$)',
                    extend='both')
tick_locator = ticker.MultipleLocator(0.02)
cbar.locator = tick_locator
cbar.update_ticks()



paper = True
if paper:
    axs[0].axvline(38,  alpha=0.5, color='black')
else:
    for E in pltEGY:
        axs[0].axvline(E, alpha=0.5, color='black')

labels = ['QFBLG', 'EMLG', '1-on-2 TBG', '1-on-1 TBG']
for ax, r, label in zip(axs[1:], nmperpixels, labels):
    ax.set_title(label)
    ax.set_ylabel(f'(nm)')
axs[-1].set_xlabel('$E_0$ (eV)')

x = np.linspace(-1, 1, Eslices[3].shape[1])
res = np.polyfit(x, Eslices[3].T, 1)
Inormnew = (Eslices[3]-res[0][:, None]*x[None])
meanspectrum = Inormnew.mean(axis=1, keepdims=True)
Inormnew = Inormnew / meanspectrum
meanspectra.append(meanspectrum.squeeze() / multipliers[2])
Inorm = Inormnew
Inorms.append(Inorm)
print(np.abs(np.log(Inorm)).max())
length = lengths[3]/4*nmperpixels[2]
axs[4].imshow(np.log(Inorm).T,
              aspect='auto', extent=[energies[2][0], energies[2][-1], -length, length],
              vmax=vmax, vmin=-vmax,
              cmap='PuOr_r',
              # vmax=1.2, vmin=0.85
              )
axs[4].set_xlim(-2, None)
axs[4].xaxis.set_minor_locator(ticker.MultipleLocator(5))

for i, EGY in enumerate(energies[:3]+[energies[2]]):
    axs[0].semilogy(EGY, meanspectra[i]*0.25**i, label=labels[i])
    axs[i+1].yaxis.set_major_locator(ticker.MultipleLocator(80))
axs[0].legend()
axs[0].set_ylabel(r'$\langle I\rangle$ (shifted)')

for i, ax in enumerate(axs):
    ax.set_title('abcde'[i], fontweight='bold', loc='left')
    
axs[0].set_yticks([1, 0.01, 0.0001])
axs[0].set_yticks([0.1, 0.001], labels=[], minor=True)

plt.savefig(os.path.join('plots', 'multislice_sigma=0_renorm_paper.pdf'))
# -

fig, axs = plt.subplots(4, figsize=[4.8, 4.5], sharex=True, sharey=True, constrained_layout=True)
for i, index in enumerate([428, 428, 478, 478]):
    x = np.linspace(-length, length, Inorms[i].shape[1])
    axs[i].plot(x, Inorms[i][index], color=f'C{i}')
    axs[i].yaxis.set_label_position("right")
    axs[i].yaxis.tick_right()
    axs[i].set_title('efgh'[i], fontweight='bold', loc='left')
    axs[i].set_xlim(-length, length)
axs[2].set_ylabel(r'                    relative intensity I/$\langle I \rangle$')
axs[3].set_xlabel('position along slice (nm)')
plt.savefig(os.path.join('plots', 'multislice_locsr.pdf'))
plt.tight_layout(pad=1.01)

# +
fig, axs = plt.subplots(2,2, figsize=[4.8, 2.5], sharex=True, sharey=True, constrained_layout=True)
axs = axs.flat
for i, index in enumerate([428, 428, 478, 478]):
    x = np.linspace(length, -length, Inorms[i].shape[1])
    axs[i].plot(x, Inorms[i][index], color=f'C{i}')
    
    axs[i].yaxis.tick_right()
    axs[i].set_title('efgh'[i], fontweight='bold', loc='left')
    axs[i].set_xlim(-length, length)
    axs[i].tick_params(axis='y', which='both', labelleft=False, labelright=(i%2 ==1))
    axs[i].xaxis.set_major_locator(ticker.MultipleLocator(80))

xticks = axs[3].xaxis.get_major_ticks()
xticks[1].label1.set_visible(False)
axs[3].yaxis.set_label_position("right")
axs[3].set_ylabel(r'                       relative intensity I/$\langle I \rangle$')
#axs[3].set_xlabel('position along slice (nm)')
axs[2].set_xlabel(r'                                                 position along slice (nm)')

plt.savefig(os.path.join('plots', 'multislice_locsr.pdf'))
#plt.tight_layout(pad=1.01)
# -

labels


