import xarray as xr
import numpy as np
from odc.geo.xr import xr_reproject

from pathlib import Path
import logging
from pyproj import CRS, Geod, Transformer
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GridResolution:
    res: float
    crs: CRS
    unit: str
    reference_latitude: float

    @classmethod
    def from_array(cls, array: xr.DataArray | xr.Dataset):
        if array.rio.crs is None:
            raise ValueError("No CRS found on input array")
        res = np.min(np.abs(array.rio.resolution())).item()
        crs = CRS.from_user_input(array.rio.crs)
        unit = 'm' if crs.is_projected else 'deg'
        reference_latitude = cls._reference_lat(array)

        return cls(
            res = res,
            crs = crs,
            unit = unit,
            reference_latitude = reference_latitude
        )

    def to_deg(self, orig_res: float | None = None):
        """Transform resolution in m to degrees using dataset center latitude."""

        if orig_res is None and self.unit == 'deg':
            return self.res

        orig_res = self.res if orig_res is None else orig_res

        ref_lat = self.reference_latitude
        geod = Geod(ellps="WGS84")
        meters_per_degree = geod.inv(0.0, ref_lat, 1.0, ref_lat)[2]
        return orig_res / meters_per_degree

    def to_m(self, orig_res: float | None = None):
        """Transform resolution in degrees to meters using dataset center latitude."""

        if orig_res is None and self.unit == 'm':
            return self.res

        orig_res = self.res if orig_res is None else orig_res

        ref_lat = self.reference_latitude
        geod = Geod(ellps="WGS84")
        meters_per_degree = geod.inv(0.0, ref_lat, 1.0, ref_lat)[2]
        return orig_res * meters_per_degree

    @staticmethod
    def _reference_lat(data: xr.DataArray | xr.Dataset) -> float:
        """Return center latitude in degrees, transforming bounds if needed."""
        try:
            minx, miny, maxx, maxy = data.rio.bounds()
            data_crs = CRS.from_user_input(data.rio.crs)
            if data_crs.is_projected:
                transformer = Transformer.from_crs(data_crs, CRS.from_epsg(4326), always_xy=True)
                _, miny = transformer.transform(minx, miny)
                _, maxy = transformer.transform(maxx, maxy)
            ref_lat = (miny + maxy) / 2.0
            return float(ref_lat)
        except Exception as e:
            logger.exception(f"Faild to compute reference latitude. Using latitude of 0 as default. Error: {e}")
            return 0.0

class GridLoader:

    def __init__(
        self, 
        max_resolution: float = 100, 
        target_crs: str = "4326", 
        resampling_method: str = 'bilinear',
        filesize_threshold: int = 1_000_000,
        allow_quick_resample: bool = False
        ):

        self.max_resolution_m = max_resolution
        try:
            self.target_crs = CRS.from_user_input(target_crs)
        except Exception as exc:
            raise ValueError(f"Invalid target_crs: {target_crs}") from exc
        self.resampling_method = resampling_method
        self.filesize_threshold = filesize_threshold
        self.allow_quick_resample = allow_quick_resample

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

        filesize = file.stat().st_size
        if filesize > self.filesize_threshold:
            chunks = {"x": 2048, "y": 2048}
        else:
            chunks = None

        data = xr.open_dataset(file, chunks = chunks)

        if data.rio.crs is None:
            raise ValueError(f"No crs found for file {file}")

        data_vars = list(data.data_vars.keys())
        if len(data_vars) == 1 and data_vars[0] == 'band_data':
            data = data.rename({'band_data': file.stem})

        grid_res = GridResolution.from_array(data)
        if self.target_crs.is_projected:
            allowed_max_res = self.max_resolution_m
        else:
            allowed_max_res = grid_res.to_deg(self.max_resolution_m)

        data_crs = CRS.from_user_input(data.rio.crs)
        if self.target_crs.is_projected:
            source_res_in_target = grid_res.to_m()
        else:
            source_res_in_target = grid_res.to_deg()

        needs_resample = source_res_in_target < allowed_max_res
        needs_reproject = data_crs != self.target_crs

        if needs_resample and self.allow_quick_resample:
            resample_factor = max(1, int(np.floor(allowed_max_res / source_res_in_target)))
            if resample_factor > 1:
                data = data.isel(x=slice(None, None, resample_factor), y=slice(None, None, resample_factor))
                grid_res = GridResolution.from_array(data)
                data = data.rio.write_transform(data.rio.transform(recalc = True))
                data = data.rio.write_crs(data_crs)
            needs_resample = False

        if needs_resample or needs_reproject:
            data = self._resample_and_reproject(
                data,
                grid_res = grid_res,
                allowed_max_resolution = allowed_max_res
            )

        return data

    def _resample_and_reproject(self, data: xr.DataArray | xr.Dataset, grid_res: GridResolution, allowed_max_resolution: float):
        """Change resolution of data to taret_res"""

        resolution_units_stay = (
            (data.rio.crs.is_projected and self.target_crs.is_projected)
            or (not data.rio.crs.is_projected and not self.target_crs.is_projected)
        )

        if grid_res.res >= allowed_max_resolution and resolution_units_stay:
            target_res = "same"
        else:
            target_res = allowed_max_resolution

        logger.info(
            f"Resampling data from crs {data.rio.crs.to_epsg()} and resolution {grid_res.res} "
            f"to crs {self.target_crs.to_epsg()} and resolution {target_res}"
        )
        data_re = xr_reproject(
            data,
            how=self.target_crs,
            resampling=self.resampling_method,
            resolution=target_res
        )
        data_re = data_re.rio.write_crs(self.target_crs)
        return data_re
