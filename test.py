#%%
import xarray as xr
from pathlib import Path
from datetime import datetime
from pytz import UTC
import matplotlib.pyplot as plt
import numpy as np
# %%
win = '6300'
dir = Path('../data/l1c')
fns = list(dir.glob(f'**/*{win}*.nc'))
# %%
ds = xr.open_dataset(fns[0])
# %%
ds
# %%
ds.countrate.isel(tstamp = 0).plot(y='za', vmin = 0, vmax = 1700)
# %%
ds
# %%
