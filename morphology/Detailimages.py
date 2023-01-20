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

# # Overviews and crops of details

# +
import napari
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from matplotlib_scalebar.scalebar import ScaleBar
from dask.distributed import Client
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
client = Client(cluster)
client

# +
folder = '/mnt/storage-linux/domainsdata/2019-04-09-CQFBLG_01_litho'
name = os.path.join('20190409_115642_3.5um_636.8_sweep-STAGE_X-STAGE_Y', 'stitch_v10_2021-11-15_1406_sobel_4_bw_200')

image = imread(os.path.join(folder, name+'.tif')).astype(float).squeeze().T.rechunk((1000, 1000))
images = dict(CQFBLG=image)

folder = '/mnt/storage-linux/domainsdata/2019-08-12-G1193/20190812_214011_2.3um_529.8_sweep-STAGE_X-STAGE_YBF_fullCP'
name = 'stitch_v10_2021-12-01_1400_sobel_2_bw_200.tif'


images['Linköping'] = imread(os.path.join(folder, name)).squeeze().astype(float).rechunk((1000, 1000))

folder = '/mnt/storage-linux/domainsdata/2019-10-22-Anna2017-020/20191022_175235_3.5um_537.1_sweep-STAGE_X-STAGE_Y'
name = 'stitch_v10_2022-02-14_1050_sobel_6_bw_200.tif'

images["Braunschweig"] = imread(os.path.join(folder, name)).squeeze().T.astype(float).rechunk((1000, 1000))

# -

nmperpixel = dict(CQFBLG=2.23, Braunschweig=2.23, Linköping=1.36)

for k in images.keys():
    print(np.array(images[k].shape)*nmperpixel[k]/1e3)

sigma = 50
smooths = {}
masks = {key: ~(image == 0) for key, image in images.items()}
for key, image in images.items():
    print(image.shape)
    smooths[key] = da.map_overlap(gauss_homogenize2,
                                  da.where(masks[key], image, np.nan), masks[key],
                                  depth=3*sigma, sigma=sigma)

# +

masks
# -

f, ax = plt.subplots(ncols=3, figsize=[9, 3.4], sharey=True, constrained_layout=True)
z0 = 1500
for i, key in enumerate(images.keys()):
    maxl = z0+int(3300*2.23/nmperpixel[key]/1.825)
    plim = images[key][z0:maxl, z0+500:maxl+500].compute()
    im = ax[i].imshow(plim.T, cmap='gray',
                      vmax=np.quantile(plim, 0.999),
                      vmin=np.quantile(plim, 0.001))  # , interpolation='none')
    im.set_extent(np.array(im.get_extent())*nmperpixel[key]/1000)
    # ax[i].set_title(key)
    ax[i].set_title("abc"[i], fontweight='bold', loc='left')
ax[0].set_ylabel('y ($\\mu$m)')
ax[1].set_xlabel('x ($\\mu$m)')
ax[0].yaxis.set_major_locator(MultipleLocator(1))
# plt.tight_layout()
#plt.savefig(os.path.join('plots', 'EGsampleoverviews.pdf'), dpi=300)

# +
f, ax = plt.subplots(ncols=3, figsize=[9, 3.5], sharey=True)

for i, key in enumerate(images.keys()):
    maxl = 1300+int(3300*2.23/nmperpixel[key])
    plim = images[key][1300:maxl, 1300:maxl].compute()
    ax[i].hist(plim.ravel(), bins=100)  # , interpolation='none')


# -

def gauss_homogenize(image, mask, sigma, nan_scale=None):
    """Homogenize image by dividing by a
    Gaussian filtered version of image, ignoring
    areas where mask is False.
    If nan_scale is given, scale all values not covered
    by the masked image by nan_scale.
    """
    VV = ndi.gaussian_filter(np.where(mask, image, 0),
                             sigma=sigma)
    VV /= ndi.gaussian_filter(mask.astype(image.dtype),
                              sigma=sigma)
    if nan_scale is not None:
        VV = np.nan_to_num(VV, nan=nan_scale)
    return image / VV


# +
f, ax = plt.subplots(ncols=2, figsize=[18, 9, ])  # , sharey=True)

for i, key in enumerate(['Linköping', 'Braunschweig']):
    maxl = 1300+int(3300*2.23/nmperpixel[key])
    plim = da.where(masks[key], images[key], np.nan)  # [1300:maxl,1300:maxl]#.compute()
    plim = plim / ndi.filters.gaussian_filter(np.nanmedian(plim,
                                                           axis=1, keepdims=True),
                                              sigma=300)
    plim = plim / ndi.filters.gaussian_filter(np.nanmedian(plim,
                                                           axis=0, keepdims=True),
                                              sigma=300)
    plim = plim.compute()
    im = ax[i].imshow(plim.T, cmap='gray',
                      vmax=np.nanquantile(plim, 0.9999),
                      vmin=np.nanquantile(plim, 0.0001)
                      # interpolation='none',
                      )
    # im.set_extent(np.array(im.get_extent())*nmperpixel[key]/1000)
# -

f, ax = plt.subplots(ncols=2, figsize=[18, 9, ], sharey=True)
smoothingresults = dict(Linköping={}, Braunschweig={})
for i, key in enumerate(['Linköping', 'Braunschweig']):
    maxl = 1300+int(3300*2.23/nmperpixel[key])
    oplim = da.where(masks[key], images[key], np.nan)  # images[key][1300:maxl,1300:maxl]#.compute()
    plim = oplim / ndi.filters.gaussian_filter(np.nanmedian(oplim,
                                                            axis=1, keepdims=True),
                                               sigma=300)
    plim = plim / ndi.filters.gaussian_filter(np.nanmedian(plim,
                                                           axis=0, keepdims=True),
                                              sigma=300)
    hist, bins = da.compute(*np.histogram(plim[masks[key]],  # bins=200, range=[0.0,3.5])
                                          # hist,bins,_ = ax[i].hist(plim.ravel(),
                                          bins=200,
                                          range=[plim[masks[key]].min(), plim[masks[key]].max()])
                            )
    otsu = threshold_otsu(hist=hist)  # fudgefactor
    thres = bins[otsu:otsu+1].mean()
    print(thres)
    lsigma = 50
    monolabel = (plim < thres).compute()
    # monolabels.append(monolabel)
    #oplim = da.where(monolabel, images[key], np.nan)
    plim = oplim / ndi.filters.gaussian_filter(np.nanmedian(da.where(monolabel, oplim, np.nan),
                                                            axis=1, keepdims=True),
                                               sigma=300)
    plim = plim / ndi.filters.gaussian_filter(np.nanmedian(da.where(monolabel, plim, np.nan),
                                                           axis=0, keepdims=True),
                                              sigma=300)
    hist, bins = da.compute(*np.histogram(plim[masks[key]],  # bins=200, range=[0.0,3.5])
                                          # hist,bins,_ = ax[i].hist(plim.ravel(),
                                          bins=200,
                                          range=[plim[masks[key]].min(), plim[masks[key]].max()])
                            )
    otsu = threshold_otsu(hist=hist)  # fudgefactor
    thres = bins[otsu:otsu+1].mean()

    monolabel = (plim < thres)
    homog = da.map_overlap(gauss_homogenize2,
                           plim, monolabel,
                           depth=3*lsigma, sigma=lsigma)
    #monoI = images[key][np.logical_and(monolabel, masks[key])].mean().compute()
    monoI = plim[np.logical_and(monolabel, masks[key])].mean().compute()

    homog2 = da.map_overlap(gauss_homogenize2,
                            plim, np.logical_and(~monolabel, masks[key]),
                            depth=3*lsigma, sigma=lsigma)
    biI = plim[np.logical_and(~monolabel, masks[key])].mean().compute()
    smoothingresults[key] = dict(homog=homog, homog2=homog2,
                                 monoI=monoI, biI=biI, monolabel=monolabel)
    #homog = gauss_homogenize2(plim, plim<thres, sigma=lsigma)
    # plim = np.where(plim < thres,
    #                homog, np.nan)
    plim = np.where(monolabel,
                    homog*monoI, homog2*biI*0.55).compute()  # fudgefactor
    im = ax[i].imshow(plim.T, cmap='gray',
                      vmax=np.nanquantile(plim, 0.999),
                      vmin=np.nanquantile(plim, 0.001)
                      # interpolation='none',
                      )
    im.set_extent(np.array(im.get_extent())*nmperpixel[key]/1000)

    # ax[i].axvline(thres)

    ax[i].set_title("({})".format("abc"[i]), fontweight='bold', loc='left')
plt.tight_layout()


# +
fig = plt.figure(figsize=[9, 13.], constrained_layout=True)
layout = """
    aa
    bc"""
ax_dict = fig.subplot_mosaic(layout,  gridspec_kw={
    # set the height ratios between the rows
    "height_ratios": [2.1, 1]}
)
for k, ax in ax_dict.items():
    ax.text(0.25, 0.25, "({})".format(k), transform=ax.transData,
            fontsize=14, #fontweight='bold', 
            va='top', ha='left',
            bbox=dict(facecolor='none', alpha=0.9, edgecolor='none'))
    ax.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
# for k,ax in ax_dict.items():
#     ax.set_title(k, fontweight='bold', loc='left')

plim2 = smooths['CQFBLG'].compute()
im = ax_dict['a'].imshow(plim2.T, cmap='gray',
                         vmax=np.nanquantile(plim2, 0.999),
                         vmin=np.nanquantile(plim2, 0.001)
                         )
im.set_extent(np.array(im.get_extent())*nmperpixel['CQFBLG']/1000)
key = 'Linköping'
plim = np.where(smoothingresults[key]['monolabel'],
                smoothingresults[key]['homog']*smoothingresults[key]['monoI'],
                smoothingresults[key]['homog2']*smoothingresults[key]['biI']*0.55).compute()  # fudgefactor
im = ax_dict['b'].imshow(plim.T, cmap='gray',
                         vmax=np.nanquantile(plim, 0.999),
                         vmin=np.nanquantile(plim, 0.001)
                         # interpolation='none',
                         )
im.set_extent(np.array(im.get_extent())*nmperpixel[key]/1000)
key = 'Braunschweig'
plim = np.where(smoothingresults[key]['monolabel'],
                smoothingresults[key]['homog']*smoothingresults[key]['monoI'],
                smoothingresults[key]['homog2']*smoothingresults[key]['biI']*0.55).compute()  # fudgefactor
im = ax_dict['c'].imshow(plim.T, cmap='gray',
                         vmax=np.nanquantile(plim, 0.999),
                         vmin=np.nanquantile(plim, 0.001)
                         # interpolation='none',
                         )
im.set_extent(np.array(im.get_extent())*nmperpixel[key]/1000)
ax_dict['b'].get_shared_y_axes().join(ax_dict['b'], ax_dict['c'])
ax_dict['b'].get_shared_x_axes().join(ax_dict['b'], ax_dict['c'])
ax_dict['c'].set_yticklabels([])

for k, ax in ax_dict.items():
    scalebar = ScaleBar(1e-6, "m", length_fraction=0.15,
                        location="lower left", box_alpha=0.1,
                        width_fraction=0.01
                        )
    ax.add_artist(scalebar)

plt.savefig(os.path.join('plots', 'EGsampleoverviews_smooth2.pdf'), dpi=600)
# -

for key in ['Linköping', 'Braunschweig']:
    area = masks[key].sum().compute() * nmperpixel[key]**2
    monoarea = smoothingresults[key]['monolabel'].sum().compute() * nmperpixel[key]**2
    print(key, area, monoarea, area - monoarea, monoarea/area)

key = 'CQFBLG'
area = masks[key].sum().compute() * nmperpixel[key]**2
print(area)

# +
f, ax = plt.subplots(ncols=3, figsize=[5, 1.8], sharey=True, constrained_layout=True)

r = 250
cs = [[4600, 2900],
      [1900, 2300],
      [6850, 4950]]
local_ims = np.array([smooths['CQFBLG'][c[0]-r:c[0]+r, c[1]-r:c[1]+r].compute() for c in cs])
vmax = np.quantile(local_ims, 0.999)
vmin = np.quantile(local_ims, 0.001)
for i, c in enumerate(cs):
    im = ax[i].imshow(local_ims[i].T, cmap='gray', vmax=vmax, vmin=vmin, interpolation='none')
    im.set_extent(np.array(im.get_extent())*nmperpixel['CQFBLG'])
#ax[1].set_xlabel('x (nm)')
#ax[0].set_ylabel('y (nm)')
for i, a in enumerate(ax):
    a.set_title("({})".format('abc'[i]), 
                #fontweight='bold',
                loc='left')
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
scalebar = ScaleBar(1e-9, "m", length_fraction=0.25,
                    location="lower right", box_alpha=0.5,
                    width_fraction=0.03
                    )
ax[-1].add_artist(scalebar)
# plt.tight_layout()
plt.savefig(os.path.join('plots', 'CQFBLGdetails_small.pdf'), dpi=600)
# -

smooths2 = {key: np.where(smoothingresults[key]['monolabel'],
                          smoothingresults[key]['homog']*smoothingresults[key]['monoI'],
                          smoothingresults[key]['homog2']*smoothingresults[key]['biI']*0.55) for key in ['Linköping', 'Braunschweig']}

plt.imshow(smooths['CQFBLG'][2000:3000, 2000:3000].compute().T, cmap='gray')

# +
f, ax = plt.subplots(ncols=4, figsize=[9, 2.5], sharey=True, constrained_layout=True)

r = 350
cs2 = [[2000, 6200],
       [4600, 3900],
       [6400, 5400],
       [4250, 4950]]
local_ims2 = np.array([smooths2['Linköping'][c[0]-r:c[0]+r, c[1]-r:c[1]+r].compute() for c in cs2])
vmax = np.quantile(local_ims2, 0.999)
vmin = np.quantile(local_ims2, 0.001)
for i, c in enumerate(cs2):
    im = ax[i].imshow(local_ims2[i].T,
                      cmap='gray', vmax=vmax, vmin=vmin, interpolation='none')
    im.set_extent(np.array(im.get_extent())*nmperpixel['Linköping'])
#ax[1].set_xlabel('x (nm)')
#ax[0].set_ylabel('y (nm)')
for i, a in enumerate(ax):
    a.set_title('abcd'[i], fontweight='bold', loc='left')
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
scalebar = ScaleBar(1e-9, "m", length_fraction=0.25,
                    location="lower right", box_alpha=0.5,
                    width_fraction=0.03
                    )
ax[-1].add_artist(scalebar)
# plt.tight_layout()
#plt.savefig(os.path.join('plots', 'linkopingdetails.pdf'))
# -

f, ax = plt.subplots(ncols=2, nrows=2, figsize=[5, 5.4], sharey=True, constrained_layout=True)
ax = ax.flat
r = 350
cs2 = [[2000, 6200],
       [4600, 3900],
       [6400, 5400],
       [4250, 4950]]
local_ims2 = np.array([smooths2['Linköping'][c[0]-r:c[0]+r, c[1]-r:c[1]+r].compute() for c in cs2])
vmax = np.quantile(local_ims2, 0.999)
vmin = np.quantile(local_ims2, 0.001)
for i, c in enumerate(cs2):
    im = ax[i].imshow(local_ims2[i].T,
                      cmap='gray', vmax=vmax, vmin=vmin, interpolation='none')
    im.set_extent(np.array(im.get_extent())*nmperpixel['Linköping'])
#ax[1].set_xlabel('x (nm)')
#ax[0].set_ylabel('y (nm)')
for i, a in enumerate(ax):
    a.set_title("({})".format('abcd'[i]),# fontweight='bold',
                loc='left')
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
scalebar = ScaleBar(1e-9, "m", length_fraction=0.25,
                    location="lower right", box_alpha=0.5,
                    width_fraction=0.03
                    )
ax[-1].add_artist(scalebar)
# plt.tight_layout()
plt.savefig(os.path.join('plots', 'linkopingdetails_small.pdf'),  dpi=600)

# +
f, ax = plt.subplots(ncols=4, figsize=[9, 2.5], sharey=True, constrained_layout=True)

r = 200
cs3 = [[3300, 1150],
       [2840, 4130],
       [4550, 3150],
       [3900, 4400], ]
local_ims3 = np.array([smooths2['Braunschweig'][c[0]-r:c[0]+r, c[1]-r:c[1]+r].compute() for c in cs3])
vmax = np.quantile(local_ims3, 0.999)
vmin = np.quantile(local_ims3, 0.001)
for i, c in enumerate(cs3):
    im = ax[i].imshow(local_ims3[i].T,
                      cmap='gray', vmax=vmax, vmin=vmin)
    im.set_extent(np.array(im.get_extent())*nmperpixel['Braunschweig'])
#ax[0].set_ylabel('y (nm)')
for i, a in enumerate(ax):
    a.set_title('abcd'[i], fontweight='bold', loc='left')
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
scalebar = ScaleBar(1e-9, "m", length_fraction=0.25,
                    location="lower right", box_alpha=0.5,
                    width_fraction=0.03
                    )
ax[-1].add_artist(scalebar)
#plt.tight_layout(pad=0.8, w_pad=0.5)
plt.savefig(os.path.join('plots', 'braunschweigdetails.pdf'), dpi=300)

# +
f, ax = plt.subplots(ncols=2, nrows=2, figsize=[5, 5.4], sharey=True, constrained_layout=True)
ax = ax.flat

r = 200
cs3 = [[3300, 1150],
       [2840, 4130],
       [4550, 3150],
       [3900, 4400], ]
local_ims3 = np.array([smooths2['Braunschweig'][c[0]-r:c[0]+r, c[1]-r:c[1]+r].compute() for c in cs3])
vmax = np.quantile(local_ims3, 0.999)
vmin = np.quantile(local_ims3, 0.001)
for i, c in enumerate(cs3):
    im = ax[i].imshow(local_ims3[i].T,
                      cmap='gray', vmax=vmax, vmin=vmin)
    im.set_extent(np.array(im.get_extent())*nmperpixel['Braunschweig'])
#ax[0].set_ylabel('y (nm)')
for i, a in enumerate(ax):
    a.set_title("({})".format('abcd'[i]),# fontweight='bold',
                loc='left')
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
scalebar = ScaleBar(1e-9, "m", length_fraction=0.25,
                    location="lower right", box_alpha=0.5,
                    width_fraction=0.03
                    )
ax[-1].add_artist(scalebar)
#plt.tight_layout(pad=0.8, w_pad=0.5)
plt.savefig(os.path.join('plots', 'braunschweigdetails_small.pdf'), dpi=600)
# -

nmperpixellist = [2.23, 1.36, 1.36, 2.23]

1.36*700, 400*2.23, 500*2.23


# +
f, ax = plt.subplots(ncols=4, figsize=[9, 2.5], constrained_layout=True)

for i, detim in enumerate([local_ims[1][80:230, 100:250],
                           local_ims2[0][400:646, 254:500],
                           local_ims2[0][100:346, 100:346],
                           local_ims3[0][100:250, 100:250]]):
    if i == 0:
        vmin = np.quantile(detim, 0.03)
    elif i in [1, 2]:
        vmin = np.quantile(local_ims2[0], 0.01)
    else:
        vmin = np.min(detim)
    smoothed = ndi.gaussian_filter(-detim, sigma=[3, 3, 3, 8][i])
    coordinates = peak_local_max(smoothed, min_distance=[8, 8, 8, 12][i])
    # print(coordinates)
    im = ax[i].imshow(detim.T, cmap='gray', 
                      vmin=vmin, interpolation='none')
    # im = ax[i].imshow(smoothed.T, cmap='gray',
    # vmin=vmin,
    #                  interpolation='none')
    im.set_extent(np.array(im.get_extent())*nmperpixellist[i])
    # ax[i].scatter(*coordinates.T*nmperpixellist[i])

for i, a in enumerate(ax):
    a.set_title('abcd'[i], fontweight='bold', loc='left')
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
scalebar = ScaleBar(1e-9, "m", length_fraction=0.25,
                    location="lower right", box_alpha=0.5,
                    width_fraction=0.03
                    )
ax[-1].add_artist(scalebar)
plt.savefig(os.path.join('plots', 'spiraldetails.pdf'), dpi=300)

# +
f, ax = plt.subplots(ncols=4, figsize=[5, 1.5], constrained_layout=True)

for i, detim in enumerate([local_ims[1][80:230, 100:250],
                           local_ims2[0][400:646, 254:500],
                           local_ims2[0][100:346, 100:346],
                           local_ims3[0][100:250, 100:250]]):
    if i == 0:
        vmin = np.quantile(detim, 0.03)
    elif i in [1, 2]:
        vmin = np.quantile(local_ims2[0], 0.01)
    else:
        vmin = np.min(detim)
    smoothed = ndi.gaussian_filter(-detim, sigma=[3, 3, 3, 8][i])
    coordinates = peak_local_max(smoothed, min_distance=[8, 8, 8, 12][i])
    # print(coordinates)
    im = ax[i].imshow(detim.T, cmap='gray',
                      vmin=vmin, interpolation='none',
                     vmax = np.quantile(detim, 0.995))
    # im = ax[i].imshow(smoothed.T, cmap='gray',
    # vmin=vmin,
    #                  interpolation='none')
    im.set_extent(np.array(im.get_extent())*nmperpixellist[i])
    # ax[i].scatter(*coordinates.T*nmperpixellist[i])

for i, a in enumerate(ax):
    a.set_title("({})".format('abcd'[i]),# fontweight='bold',
                loc='left')
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
scalebar = ScaleBar(1e-9, "m", length_fraction=0.25,
                    location="lower right", box_alpha=0.5,
                    width_fraction=0.03
                    )
ax[-1].add_artist(scalebar)
plt.savefig(os.path.join('plots', 'spiraldetails_small.pdf'), dpi=600)

# +


disldata = np.stack([local_ims[1][60:260, 50:250],
                     local_ims3[1][100:300, 200:],
                     local_ims3[3][50:250, :200],
                     local_ims3[3][50:250, 200:]])
viewer = napari.view_image(np.swapaxes(disldata, 1, 2))
pts_layer = viewer.add_points(name='points', ndim=3)
pts_layer.mode = 'add'
# -

pts_layer.data

# +
f, ax = plt.subplots(ncols=4, figsize=[5, 1.5], constrained_layout=True)

for i, detim in enumerate([local_ims[1][60:260, 50:250],
                           local_ims3[1][100:300, 200:],
                           local_ims3[3][50:250, :200],
                           local_ims3[3][50:250, 200:]]):
    if i == 0:
        vmin = np.quantile(detim, 0.03)
    elif i in [1, 2]:
        vmin = np.quantile(local_ims2[0], 0.01)
    else:
        vmin = np.min(detim)
    # print(coordinates)
    im = ax[i].imshow(detim.T, cmap='gray', vmin=vmin, interpolation='none')
    # im = ax[i].imshow(smoothed.T, cmap='gray',
    # vmin=vmin,
    #                  interpolation='none')
    im.set_extent(np.array(im.get_extent())*nmperpixellist[i])
    # ax[i].scatter(*coordinates.T*nmperpixellist[i])

#domains = [pts_layer.data[:4], pts_layer.data[4:7], pts_layer.data[7:]]
#for j, domain in enumerate(domains):
#    i = int(domain[0, 0])
#    ax[i].scatter(domain[:, 2]*nmperpixellist[i], domain[:, 1]*nmperpixellist[i], alpha=0.9, color=f'C{j}')

#p = pts_layer.data[3]
#i = int(p[0])
#ax[i].scatter(p[2]*nmperpixellist[i], p[1]*nmperpixellist[i],
#              alpha=0.6, color='C1', s=20)

for i, a in enumerate(ax):
    a.set_title(f"({'abcd'[i]})", loc='left')
    a.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
scalebar = ScaleBar(1e-9, "m", length_fraction=0.3,
                    location="lower right", box_alpha=0.5,
                    width_fraction=0.05
                    )
ax[-1].add_artist(scalebar)
plt.savefig(os.path.join('plots', 'dislocationdetails_small.pdf'), dpi=300)
# -

# 10 pix CQFBLG = 22nm
#
# 14 pix linkoping = 19nm
