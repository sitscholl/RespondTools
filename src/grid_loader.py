import xarray as xr
import numpy as np
from odc.geo.xr import xr_reproject

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GridLoader:

    def __init__(self, max_resolution: float, target_crs: str, resampling_method: str = 'bilinear'):
        self.max_resolution_m = max_resolution
        self._max_resolution_deg = self._transform_resolution(max_resolution)
        self.target_crs = target_crs
        self.resampling_method = resampling_method

    def load(self, file: str | Path):

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
        allowed_max_res = self.max_resolution_m if is_projected else self._max_resolution_deg

        if (max_res < allowed_max_res) or data.rio.crs != self.target_crs:
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

        logger.info(f"Resampling data from crs {data.rio.crs.to_epsg()} and resolution {max_res} to crs {self.target_crs} and resolution {target_res}")
        return xr_reproject(data, how=self.target_crs, resampling=self.resampling_method, resolution = target_res)

    def _transform_resolution(self, resolution_m):
        """Transform resolution in m to degree for unprojected crs"""
        pass