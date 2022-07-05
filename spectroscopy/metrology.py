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
#     display_name: pyGPA-cupy
#     language: python
#     name: pygpa-cupy
# ---

# +
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.io import imread
from moisan2011 import per
import colorcet
import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from pyL5.analysis.DriftCorrection.StatRegistration import StatRegistration
from pyL5.analysis.CorrectChannelPlate.CorrectChannelPlate import CorrectChannelPlate
from pyL5.lib.analysis.container import Container
from dask.distributed import Client, LocalCluster

import pyGPA.geometric_phase_analysis as GPA
import pyGPA.unit_cell_averaging as uc
from pyGPA.imagetools import indicate_k, fftplot, trim_nans, gauss_homogenize2, trim_nans2
from pyGPA.mathtools import wrapToPi
import pyGPA.cuGPA as cuGPA
from skimage.morphology import erosion, disk, binary_erosion
from skimage.morphology import erosion, disk
from registration.registration import register_stack, strided_register

# +
folder = '/mnt/storage-linux/2021TBGdata'
dbov_name = "20200713_163811_5.7um_501.2_sweep-STAGE_X-STAGE_Y_domainboundaries_stitch_v10_2020-11-20_1649_sobel_4_bw_200.tif"
dbov_image = imread(os.path.join(folder, dbov_name)).squeeze()

NMPERPIXEL = 3.7
# -

plt.figure(figsize=[15,9])
plt.imshow(dbov_image[3200:5000,4000:6500], vmin=2e4, vmax=5e4)
plt.colorbar()

# +
image = dbov_image[3200:5000,4000:6500].astype(float)

smooth = gauss_homogenize2(image, image>2.2e4, sigma=50)
plt.imshow(image>2.2e4)
# -

plt.hist(smooth.ravel(), bins=200);

plt.figure(figsize=[15,9])
plt.imshow(smooth> 0.88)#, vmin=0.9)
plt.colorbar()

smooth2 = gauss_homogenize2(image, smooth> 0.88, sigma=50)

plt.figure(figsize=[15,9])
plt.imshow(smooth, vmin=0.9)
plt.colorbar()

from matplotlib import patches, ticker

# +
fig, axs = plt.subplots(ncols=2, figsize=[9,4.2],gridspec_kw=dict(width_ratios=[8,4]))
axs = axs[::-1]
ax = axs[1]
im = ax.imshow(smooth2, vmin=0.95, vmax=1.05, cmap='gray', interpolation='none')
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1e3)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.annotate('DDW',
            xy=(7.8, 5.1), xycoords='data',
            xytext=(6.5, 5), textcoords='data',
            arrowprops=dict(facecolor='purple', shrink=0.05, ec='none'),
            horizontalalignment='center', verticalalignment='bottom',
           fontweight='bold', color='purple')
ax.annotate('  ',
            xy=(7.8, 5.55), xycoords='data',
            xytext=(6.5, 5), textcoords='data',
            arrowprops=dict(facecolor='purple', shrink=0.1, ec='none'),
            horizontalalignment='center', verticalalignment='bottom',
           fontweight='bold', color='purple')
ax.annotate('  ',
            xy=(7.7, 5.8), xycoords='data',
            xytext=(6.5, 5), textcoords='data',
            arrowprops=dict(facecolor='purple', shrink=0.1, ec='none'),
            horizontalalignment='center', verticalalignment='bottom',
           fontweight='bold', color='purple')
ax.annotate('  ',
            xy=(5.7, 4.3), xycoords='data',
            xytext=(6.5, 4.8), textcoords='data',
            arrowprops=dict(facecolor='purple', shrink=0.1, ec='none'),
            horizontalalignment='center', verticalalignment='top',
           fontweight='bold', color='purple')
axs[0].yaxis.set_label_position("right")
axs[0].yaxis.tick_right()
ax.set_xlabel(r'x ($\mu$m)')
ax.set_ylabel(r'y ($\mu$m)')

locs = [(5.16,3.66), (4.22,4.55), (3.92,3.3)]
r = 0.85
for loc in locs:
    e1 = patches.Arc(loc, 2*r, 2*r, theta1=-45.0, theta2=45,
                 angle=0, linewidth=3, fill=False, zorder=2, edgecolor='orange', alpha=0.7)

    ax.add_patch(e1)
#plt.colorbar(im, ax=ax)
#plt.grid()

Halbertal = imread(os.path.join('plots', 'metrologysource2.png')).squeeze()
axs[0].imshow(Halbertal, extent=[0,50,0,3000], aspect='auto')
axs[0].set_ylabel(r'$\kappa^{-1}(nm)$')
axs[0].set_xlabel(r'counts')
axs[0].axhline(r*1e3, color='orange', linewidth=3)
axs[0].annotate('This data ', (50,r*1e3),
               horizontalalignment='right', verticalalignment='bottom', 
               fontweight='bold', color='orange')
plt.tight_layout()
for ax, l in zip(axs,'ba'):
    ax.set_title(l, fontweight='bold', loc='left')
plt.savefig(os.path.join('plots', 'metrology.pdf'))
# -

plt.hist(smooth2.ravel(), bins=200);

np.exp(0.12)

np.exp(0.3)


