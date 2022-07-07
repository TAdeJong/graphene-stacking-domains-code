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
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet  # noqa: F401

from pyL5.analysis.DriftCorrection.StatRegistration import StatRegistration
from pyL5.analysis.CorrectChannelPlate.CorrectChannelPlate import CorrectChannelPlate
from pyL5.lib.analysis.container import Container


# +
folder = '/mnt/storage-linux'

names = ['speeldata/20200512-XTBLG01/20200512_111934_0.66um_437.9_sweepOBJ_lowE_widerange',
         'speeldata/20200716-XTBLG02/20200717_133724_2.3um_480.6_sweepOBJ',
         'speeldata/20200916-XTBLG02/20201006_192818_2.3um_449.9_sweepOBJ',
         'domainsdata/2019-08-12-G1193/20190812_140334_2.3um_535.8_sweepOBJ']
# -

for name in names:
    script = StatRegistration(os.path.join(folder, name + '.nlp'))
    script.start()

conts = {name: Container(os.path.join(folder, name)) for name in names}
data = {name: conts[name].getStack('driftcorrected').getDaskArray() for name in names}

for name, cont in conts.items():
    print(name, cont['EGYM'][0])


f, axs = plt.subplots(ncols=3, nrows=3, figsize=[6, 6], constrained_layout=True)
for i, ax in enumerate(axs):
    name = names[i+1]
    objs = np.array(conts[name]['OBJ'])
    ldat = data[name][:, 650:1100, 150:500]
    ldat = ldat / np.array(conts[name]['MULTIPLIER'])[:, None, None]
    print(len(objs), len(objs)//2)
    ax[0].imshow(ldat[0].T, cmap='gray', vmax=ldat.max(), vmin=ldat.min())
    ax[0].set_title(f"{(objs[0] - objs[len(objs)//2])*1e3:.2f} mA")
    ax[1].imshow(ldat[len(objs)//2].T, cmap='gray', vmax=ldat.max(), vmin=ldat.min())
    ax[1].set_title("0.00 mA")
    ax[2].imshow(ldat[-1].T, cmap='gray', vmax=ldat.max(), vmin=ldat.min())
    ax[2].set_title(f"{(objs[-1] - objs[len(objs)//2])*1e3:+.2f} mA")
    ax[0].set_title('abc'[i], fontweight='bold', loc='left')
    for a in ax:
        a.tick_params(
            bottom=False,      # ticks along the bottom edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
    scalebar = ScaleBar(1.36e-9, "m", length_fraction=0.25,
                        location="lower right", box_alpha=0.3,
                        width_fraction=0.04
                        )
    ax[2].add_artist(scalebar)
plt.savefig(os.path.join('plots', 'defocusseries.pdf'), dpi=600)

# +
fig, axs = plt.subplots(ncols=5, figsize=[9, 2], constrained_layout=True)

ldat = data[names[0]]

ldat = ldat / np.array(conts[names[0]]['MULTIPLIER'])[:, None, None]

ldat = ldat[:, 550:1050, 250:750]

for i, ax in enumerate(axs):
    j = int(i*4.5+1)
    print(j)
    ax.imshow(ldat[j].T, cmap='gray', vmax=ldat.max(), vmin=ldat.min())
    ax.set_title(f"{(conts[names[0]]['OBJ'][j] - conts[names[0]]['OBJ'][10])*1e3:.2f} mA")
    ax.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
scalebar = ScaleBar(0.88e-9, "m", length_fraction=0.25,
                    location="lower right", box_alpha=0.3,
                    width_fraction=0.04
                    )
axs[-1].add_artist(scalebar)
plt.savefig(os.path.join('plots', 'defocusseries_large_angle.pdf'), dpi=600)
