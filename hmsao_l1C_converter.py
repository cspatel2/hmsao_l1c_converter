# convert level 1B -> 1C
# %%
# %%
from datetime import datetime
import numpy as np
import xarray as xr
from pathlib import Path
# %%
LOCALPATH = Path(__file__).parent
CALIBMAP_DIR = LOCALPATH / 'calib_maps'



def main(
    datdir: Path = Path('/home/charmi/locsststor/proc/hmsao/l1b'),
    destdir: str | Path | None = '',
    calibmap_dir: Path | None = CALIBMAP_DIR,
    photon:  bool = True
):
    if destdir is None:
        destd = Path('./l1c')
    elif isinstance(destdir, str) and destdir == '':
        destd = Path(str(datdir).replace('l1b', 'l1c'))
    elif isinstance(destdir, str):
        destd = Path(destdir)
    else:
        destd = destdir
    destd.mkdir(exist_ok=True)
    if calibmap_dir is None:  # if None, check current folder
        cmap_dir = LOCALPATH / 'calib_maps'
    else:
        cmap_dir = calibmap_dir

    calibfns = list(cmap_dir.glob(f'*calib*.nc'))
    if len(calibfns) < 1:
        raise FileNotFoundError(
            f"No calibration map files found in {cmap_dir}."
        )
    windows = set([fp.stem.split('_')[-1] for fp in calibfns])
    windows = sorted(list(windows))

    for win in windows:
        print(f"Processing window: {win}")
        calibfns = list(cmap_dir.glob(f'*{win}*.nc'))
        if len(calibfns) > 0:
            print(
                f"Multiple calibration map files found for window {win}. Using the first one: {calibfns[0].name}")
            cfn = calibfns[0]

        calibds = xr.open_dataset(cfn)  # type: ignore

        print(f"Destination directory: {destd}")

        fns = list(datdir.glob(f'**/*{win}*.nc'))
        print(f"Found {len(fns)} files to process.")
        fns.sort()

        if photon:
            print("Calibrating to photons/s.cm2.sr...")
            id = 'kp'
            units = 'photons/s.cm2.sr.nm'
        else:
            print("Calibrating to Rayleighs...")
            id = 'kr'
            units = 'Rayleighs/nm'

        for fn in fns:
            print(f"Processing file: {fn.name}...", end='', flush=True)
            ds = xr.open_dataset(fn)
            ss = ds.copy()
            dwl = np.mean(np.diff(ds.wavelength.data))
            ss.countrate.data = ss.countrate.data * calibds[id].data / dwl
            ss.noise.data = ss.noise.data * calibds[id].data / dwl

            all_vars = list(ds.coords) + list(ds.keys())
            for var in all_vars:
                attrs = {}
                for k, v in ds[var].attrs.items():
                    if k not in ['unit']:
                        continue
                    else:
                        k = 'units'
                    attrs[k] = v
                ss[var].attrs = attrs

            ss['countrate'].attrs.update({'units': units})
            ss['countrate'].attrs.update({'long_name': 'Calibrated Intensity'})
            ss = ss.rename_vars({'countrate': 'intensity'})
            ss['noise'].attrs.update({'units': units})

            attrs = {k: v for k, v in ds.attrs.items() if k not in 'unit'}
            ss.attrs = attrs
            ss.attrs['DataProcessingLevel'] = 'L1c - Calibrated.'
            ss.attrs['FileCreationDate'] = datetime.now().strftime(
                "%m/%d/%Y, %H:%M:%S EDT")
            encoding = {var: {'zlib': True}
                        for var in (*ss.data_vars.keys(), *ss.coords.keys())}
            print('\tsaving...', end='', flush=True)
            outfn = destd.joinpath(fn.stem.replace('l1b', 'l1c') + fn.suffix)
            ss.to_netcdf(outfn, encoding=encoding)
            print('\tDone.', flush=True)


# %%
