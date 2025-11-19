#convert level 1B -> 1C
#%%
#%%
from datetime import datetime
import numpy as np
import xarray as xr
from skimage import transform
from pathlib import Path
from secondary_straightening import secondary_straightening
from matplotlib import pyplot as plt
from tqdm import tqdm

# %%
######### user inputs #########
windows = ['5577', '6300']
datdir = Path('../data/l1b')
destdir = '' #if empty string, will replace 'l1b' with 'l1c' in datdir path. If None, will use './l1c' as destdir.
calibmap_dir = None #if empty string, will replace 'l1b' with 'calib_maps' in datdir path. If None, will use './calib_maps' as calibmap_dir.
photon = False # if true, calibrate to photons/s.cm2.sr, if false, calibrate to Rayleighs
##################################
for win in windows:
    print(f"Processing window: {win}")
    if calibmap_dir is None: # if None, check current folder
        calibmap_dir = Path('./')
    elif calibmap_dir == '': # if emply, check in datadir for calib_maps
        calibmap_dir = Path(str(datdir).replace('l1b','calib_maps'))
    
    calibfns = list(calibmap_dir.glob(f'*calib*{win}*.nc'))
    if len(calibfns) < 1:
        raise FileNotFoundError(f"No calibration map files found in {calibmap_dir} for window {win}.")
    elif len(calibfns) > 0:
        print(f"Multiple calibration map files found for window {win}. Using the first one: {calibfns[0].name}")
        cfn = calibfns[0]

    calibds = xr.open_dataset(cfn) #type: ignore

    if destdir is None:
        destdir = Path('./l1c')
    elif destdir == '' :
        destdir = Path(str(datdir).replace('l1b','l1c'))
    destdir.mkdir(exist_ok=True)
    print(f"Destination directory: {destdir}")
    
    fns = list(datdir.glob(f'**/*{win}*.nc'))
    print(f"Found {len(fns)} files to process.")
    fns.sort()
    
    if photon: 
        print ("Calibrating to photons/s.cm2.sr...")
        id = 'kp'
        units = 'photons/s.cm2.sr.nm'
    else: 
        print ("Calibrating to Rayleighs...")
        id = 'kr'
        units = 'Rayleighs/nm'

    for fn in fns:
        print(f"Processing file: {fn.name}...")
        ds = xr.open_dataset(fn)
        ss = ds.copy()
        dwl = np.mean(np.diff(ds.wavelength.data))
        ss.countrate.data = ss.countrate.data * calibds[id].data / dwl
        ss.noise.data = ss.noise.data * calibds[id].data / dwl

        all_vars = list(ds.coords) + list(ds.keys())
        for var in all_vars:
            attrs = {}
            for k,v in ds[var].attrs.items():
                if k not in ['unit']:
                    continue
                else: k = 'units'
                attrs[k] = v
            ss[var].attrs = attrs

        ss['countrate'].attrs.update({'units': units})
        ss['countrate'].attrs.update({'long_name': 'Calibrated Intensity'}) 
        ss = ss.rename_vars({'countrate':'intensity'})
        ss['noise'].attrs.update({'units': units})
        
        attrs = {k:v for k,v in ds.attrs.items() if k not in 'unit'}
        ss.attrs = attrs
        ss.attrs['DataProcessingLevel'] = 'L1c - Calibrated.'
        ss.attrs['FileCreationDate'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT")
        encoding = {var: {'zlib': True} for var in (*ss.data_vars.keys(), *ss.coords.keys())}
        outfn = destdir.joinpath(fn.stem.replace('l1b','l1c') + fn.suffix)
        ss.to_netcdf(outfn, encoding=encoding)
        print(f"Saved L1C file: {outfn.name}")

