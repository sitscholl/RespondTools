import xarray as xr
import numpy as np
from odc.geo.xr import xr_reproject

from pathlib import Path
import logging
from pyproj import CRS, Geod

logger = logging.getLogger(__name__)

class GridLoader:

    def __init__(self, max_resolution: float = 100, target_crs: str = "4326", resampling_method: str = 'bilinear'):
        self.max_resolution_m = max_resolution
        try:
            self.target_crs = CRS.from_user_input(target_crs)
        except Exception as exc:
            raise ValueError(f"Invalid target_crs: {target_crs}") from exc
        self.resampling_method = resampling_method

        self.files = []

    def register_files(self, files: list[str]):
        self.files = [Path(i) for i in files]

    def iter_datasets(self):
        for file in self.files:

            if not file.exists():
                logger.warning(f"File {file} does not exist. Skipping...")
                continue
            
            data = self.load(file)
            yield file, data

    def load(self, file: str | Path):
        file = Path(file)
        if not file.exists():
            raise ValueError(f"Could not find file at {file}")

        data = xr.open_dataset(file)

        if data.rio.crs is None:
            raise ValueError(f"No crs found for file {file}")

        data_vars = list(data.data_vars.keys())
        if len(data_vars) == 1 and data_vars[0] == 'band_data':
            data = data.rename({'band_data': file.stem})

        res = data.rio.resolution()
        if res is None:
            raise ValueError(f"Could not get resolution of file {file}")

        max_res = np.min(np.abs(res)).item()
        is_projected = data.rio.crs.is_projected
        if is_projected:
            allowed_max_res = self.max_resolution_m
        else:
            allowed_max_res = self._transform_resolution(self.max_resolution_m, data)

        data_crs = CRS.from_user_input(data.rio.crs)
        if (max_res < allowed_max_res) or data_crs != self.target_crs:
            data = self._resample_and_reproject(data, allowed_max_resolution = allowed_max_res)

        return data

    def _resample_and_reproject(self, data: xr.DataArray | xr.Dataset, allowed_max_resolution: float):
        """Change resolution of data to taret_res"""

        res = data.rio.resolution()
        max_res = np.min(np.abs(res)).item()

        if max_res >= allowed_max_resolution:
            target_res = 'same'
        else:
            target_res = allowed_max_resolution

        logger.info(
            f"Resampling data from crs {data.rio.crs.to_epsg()} and resolution {max_res} "
            f"to crs {self.target_crs.to_epsg()} and resolution {target_res}"
        )
        return xr_reproject(
            data,
            how=self.target_crs,
            resampling=self.resampling_method,
            resolution=target_res
        )

    def _transform_resolution(self, resolution_m, data: xr.DataArray | xr.Dataset):
        """Transform resolution in m to degrees using dataset center latitude."""
        try:
            minx, miny, maxx, maxy = data.rio.bounds()
            ref_lat = (miny + maxy) / 2.0
        except Exception:
            ref_lat = 0.0

        geod = Geod(ellps="WGS84")
        meters_per_degree = geod.inv(0.0, ref_lat, 1.0, ref_lat)[2]
        return resolution_m / meters_per_degree
