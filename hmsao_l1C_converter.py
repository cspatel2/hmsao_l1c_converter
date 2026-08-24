# convert level 1B -> 1C
# %%
# %%
from dataclasses import dataclass
from datetime import datetime
import sys
import hdf5plugin
import numpy as np
import tqdm
from tqdm import tqdm
import xarray as xr
from pathlib import Path
import argparse
xr.set_options(netcdf_engine_order=["h5netcdf", "netcdf4", "scipy"])
# %%
@dataclass
class L1CConfig:
    """ Configuration for L1C conversion.
    Arguments:
        rootdir (str|Path): Root directory containing L1B files.
        destdir (str|Path): Destination directory for L1C files. If not provided, will create a 'l1c' directory parallel to the rootdir.
        windows (list[str]): List of windows to process. If not provided, will process all windows found in the calibration map directory.
        calibmap_dir (str|Path): Directory containing calibration map files.
        photon (bool): If set, calibrate to photons/s.cm2.sr.nm. If not set, calibrate to Rayleighs/nm. Default is True.
    """
    rootdir: str | Path
    destdir: str | Path
    windows: list[str]
    calibmap_dir: str | Path
    photon: bool = True

def main(config: L1CConfig):
    """ Converts HMS data from L1b -> L1C.
    uses calibration maps to convert from countrate to physical units (photons/s.cm2.sr.nm or Rayleighs/nm).

    Args:
        config (L1CConfig): Configuration for L1C conversion.
            - rootdir (str|Path): Root directory containing L1B files.
            - destdir (str|Path): Destination directory for L1C files. If not provided, will create a 'l1c' directory parallel to the rootdir.
            - windows (list[str]): List of windows to process. If not provided, will process all windows found in the calibration map directory.
            - calibmap_dir (str|Path): Directory containing calibration map files.
            - photon (bool): If set, calibrate to photons/s.cm2.sr.nm. If not set, calibrate to Rayleighs/nm. Default is True.


    Raises:
        FileNotFoundError: If no calibration map files are found in the specified directory.
    """    
    #check Root dir
    if isinstance(config.rootdir, str):
        config.rootdir = Path(config.rootdir)
    config.rootdir = config.rootdir.expanduser()

    #check Destination Dir
    if isinstance(config.destdir, str):
        if config.destdir == '':
            config.destdir = Path(str(config.rootdir).replace('l1b', 'l1c'))
        else:
            config.destdir = Path(config.destdir)
    config.destdir = config.destdir.expanduser() 
    config.destdir.mkdir(exist_ok=True, parents=True)
    print(f"Destination directory: {config.destdir}")

    #check Calibration Map Dir
    if isinstance(config.calibmap_dir, str):
        config.calibmap_dir = Path(config.calibmap_dir)
    config.calibmap_dir = config.calibmap_dir.expanduser()

    calibfns = list(config.calibmap_dir.glob(f'*calib*.nc'))
    if len(calibfns) < 1:
        raise FileNotFoundError(
            f"No calibration map files found in {config.calibmap_dir}."
        )

    #check windows
    available_windows = set([fp.stem.split('_')[-1] for fp in calibfns])
    if config.windows == ['']:
        valid_windows = available_windows
    else:
        valid_windows = [w for w in config.windows if w in available_windows]
    print(f"Valid windows: {valid_windows}")

    #run conversion
    for win in valid_windows:
        # print(f"Processing window: {win}")
        calibfns = list(config.calibmap_dir.glob(f'*{win}*.nc'))
        if len(calibfns) > 0:
            print(
                f"Multiple calibration map files found for window {win}. Using the first one: {calibfns[0].name}")
            cfn = calibfns[0]

        calibds = xr.open_dataset(cfn)  # type: ignore

        fns = list(config.rootdir.glob(f'**/*{win}*.nc'))
        # print(f"Found {len(fns)} files to process.")
        fns.sort()

        if config.photon:
            print("Calibrating to photons/s.cm2.sr...")
            id = 'kp'
            units = 'photons/s.cm2.sr.nm'
        else:
            print("Calibrating to Rayleighs...")
            id = 'kr'
            units = 'Rayleighs/nm'

        for fn in tqdm(fns, desc=f"Processing window {win}", unit="file"):
            # print(f"Processing file: {fn.name}...", end='', flush=True)
            ds = xr.open_dataset(fn)
            ss = ds.copy()
            # ss = ss.sel(wavelength=calibds.wavelength.data)  # ADDED FOR NOW CHECK IT LATER
            ss = ss.sel(za = calibds.za.data)  # ADDED FOR NOW CHECK IT LATER
            dwl = np.mean(np.diff(ds.wavelength.data))
            ss.countrate.data = ss.countrate.data * calibds[id].data / dwl
            ss.noise.data = ss.noise.data * calibds[id].data / dwl # currently ignoring noise in calibration

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
            encoding = {var: hdf5plugin.Zstd(clevel=3)
                        for var in (*ss.data_vars.keys(), *ss.coords.keys())}
            # print('\tsaving...', end='', flush=True)
            outfn = config.destdir.joinpath(fn.stem.replace('l1b', 'l1c') + fn.suffix)
            ss.to_netcdf(outfn, encoding=encoding)
            # print('\tDone.', flush=True)
            sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert L1B files to L1C using calibration maps.")
    parser.add_argument('--rootdir', type=str, required=True, help="Root directory containing L1B files.")
    parser.add_argument('--destdir', type=str, default='', help="Destination directory for L1C files. If not provided, will create a 'l1c' directory parallel to the rootdir.")
    parser.add_argument('--windows', type=str, nargs='*', default=[''], help="List of windows to process. If not provided, will process all windows found in the calibration map directory.")
    parser.add_argument('--calibmap_dir', type=str, required=True, help="Directory containing calibration map files.")
    parser.add_argument('--photon',type = bool,required = False, default= True, help="If set, calibrate to photons/s.cm2.sr.nm. If not set, calibrate to Rayleighs/nm.")
    args = parser.parse_args()
    l1c_config = L1CConfig(
        rootdir=args.rootdir,
        destdir=args.destdir,
        windows=args.windows,
        calibmap_dir=args.calibmap_dir,
        photon=args.photon
    )
    main(l1c_config)

# %%
