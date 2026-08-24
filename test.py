#%%
import xarray as xr
from pathlib import Path
from datetime import datetime
from pytz import UTC
import matplotlib.pyplot as plt
import numpy as np
# %%
win = '7774'
dir = Path('~/local_data_cedar/hmsao-v2/l1c').expanduser()
fns = list(dir.glob(f'*{win}*.nc'))
print(len(fns))
# %%
ds = xr.open_dataset(fns[0])
# %%
ds
# %%
ds.intensity.isel(tstamp = 0).plot(y='za', vmin = 0)
# %%
ds.noise.isel(tstamp = 0).plot(y='za', vmin = 0)
# %%
da = ds.intensity.isel(tstamp = 0, za =500)
# %%
da.plot()
# %%
np.log10(da.mean(skipna=True).values)

# %%
np.log10(da.max(skipna=True).values)
# %%
np.log10(da.max(skipna=True).values - da.mean(skipna=True).values)
# %%
