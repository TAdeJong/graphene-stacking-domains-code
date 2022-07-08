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

# # Showcase moiré metrology using LEEM data
#
# In this notebook we show that moiré metrology as described by [Halbertal et al. Nat Commun 12, 242 (2021).](https://doi.org/10.1038/s41467-020-20428-1) can also be applied to LEEM data. We use the data from [De Jong et al. Nat Commun 13, 70 (2022).](https://doi.org/10.1038/s41467-021-27646-1), as available from data.4tu.nl: https://doi.org/10.4121/16843510 .

# +
import matplotlib.pyplot as plt
from matplotlib import patches, ticker
import numpy as np
import os
from skimage.io import imread
import colorcet  # noqa: F401

from pyGPA.imagetools import gauss_homogenize2


# +
folder = '/mnt/storage-linux/2021TBGdata'
dbov_name = "20200713_163811_5.7um_501.2_sweep-STAGE_X-STAGE_Y_domainboundaries_stitch_v10_2020-11-20_1649_sobel_4_bw_200.tif"
dbov_image = imread(os.path.join(folder, dbov_name)).squeeze()

NMPERPIXEL = 3.7
# -

plt.figure(figsize=[15, 9])
plt.imshow(dbov_image[3200:5000, 4000:6500], vmin=2e4, vmax=5e4)
plt.colorbar()

# +
image = dbov_image[3200:5000, 4000:6500].astype(float)

smooth = gauss_homogenize2(image, image > 2.2e4, sigma=50)
plt.imshow(image > 2.2e4)
# -

plt.figure(figsize=[15, 9])
plt.imshow(smooth > 0.88)  # Magic number blocking out most of the folds
plt.colorbar()

smooth2 = gauss_homogenize2(image, smooth > 0.88, sigma=50)

plt.figure(figsize=[15, 9])
plt.imshow(smooth, vmin=0.9)
plt.colorbar()

# +
fig, axs = plt.subplots(ncols=2, figsize=[9, 4.2], gridspec_kw=dict(width_ratios=[8, 4]))
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

locs = [(5.16, 3.66), (4.22, 4.55), (3.92, 3.3)]
r = 0.85
for loc in locs:
    e1 = patches.Arc(loc, 2*r, 2*r, theta1=-45.0, theta2=45,
                     angle=0, linewidth=3, fill=False, zorder=2, edgecolor='orange', alpha=0.7)

    ax.add_patch(e1)

Halbertal = imread(os.path.join('plots', 'metrologysource2.png')).squeeze()
axs[0].imshow(Halbertal, extent=[0, 50, 0, 3000], aspect='auto')
axs[0].set_ylabel(r'$\kappa^{-1}(nm)$')
axs[0].set_xlabel(r'counts')
axs[0].axhline(r*1e3, color='orange', linewidth=3)
axs[0].annotate('This data ', (50, r*1e3),
                horizontalalignment='right', verticalalignment='bottom',
                fontweight='bold', color='orange')
plt.tight_layout()
for ax, l in zip(axs, 'ba'):
    ax.set_title(l, fontweight='bold', loc='left')
plt.savefig(os.path.join('plots', 'metrology.pdf'))
