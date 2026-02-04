import xarray as xr
import rioxarray
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)

_TRANSFORM_TEMPLATE = """scale = {scale}
offset = {offset}
data_min = {data_min}
data_max = {data_max}
out_min = {out_min}
out_max = {out_max}
out_nodata = {out_nodata}

transform back using: value * scale + offset
"""

@dataclass
class TransformParameters:
    data_max: float | int
    data_min: float | int
    out_max: float | int
    out_min: float | int
    out_nodata: float | int | None
    new_dtype: np.dtype

    @property
    def denom(self):
        denom = self.data_max - self.data_min
        denom = denom if not np.isclose(denom, 0) else 1
        return denom

    @property
    def scale(self):
        return (self.data_max - self.data_min) / (self.out_max - self.out_min)

    @property
    def offset(self):
        return (self.data_min - self.out_min * self.scale)

    def write(self, out_path: str | Path):
        with open(out_path, "w") as f:
            f.write(
                _TRANSFORM_TEMPLATE.format(
                    scale = self.scale,
                    offset = self.offset,
                    data_min = self.data_min,
                    data_max = self.data_max,
                    out_min = self.out_min,
                    out_max = self.out_max,
                    out_nodata = self.out_nodata
                )
            )

@dataclass
class Grid:
    data: xr.Dataset
    original_location: str | Path
    nodata_mask: xr.Dataset
    original_nodata: Dict[str, float | int | None]
    original_dtype: Dict[str, np.dtype]
    transformation: Dict[str, TransformParameters] | None = None
    transformed: bool = False

    def __post_init__(self):
        if not isinstance(self.data, xr.Dataset):
            raise ValueError(f"Data must be an xr.Dataset. Got {type(self.data)}")

    @classmethod
    def from_geotiff(cls, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Could not find geotiff file at {path}")
        data = xr.open_dataarray(path).squeeze(drop = True)

        if isinstance(data, xr.DataArray):
            data = data.to_dataset(name = path.stem)

        nodata_mask = []
        nodata_vals = {}
        dtype_vals = {}
        for var, arr in data.data_vars.items():
            nodata = getattr(arr.rio, "nodata", None)
            
            if nodata is None:
                if np.issubdtype(arr.dtype, np.floating):
                    mask = xr.where(arr.notnull(), 1, 0).astype(bool)
                    nodata = np.nan
                else:
                    mask = xr.ones_like(arr).astype(bool)
            elif np.isnan(nodata):
                mask = xr.where(arr.notnull(), 1, 0).astype(bool)
            else:
                mask = xr.where((arr != nodata) & arr.notnull(), 1, 0).astype(bool)

            mask.name = var
            nodata_mask.append(mask)
            nodata_vals[var] = nodata
            dtype_vals[var] = arr.dtype

        return cls(data, path, xr.merge(nodata_mask), nodata_vals, dtype_vals)

    @property
    def crs(self):
        return self.data.rio.crs.to_epsg() if self.data.rio.crs else None

    @property
    def resolution(self):
        return self.data.rio.resolution()

    @staticmethod
    def _resolve_output_nodata(out_dtype: np.dtype, output_nodata: str | float | int | None):
        if np.issubdtype(out_dtype, np.integer):
            out_info = np.iinfo(out_dtype)
        else:
            out_info = np.finfo(out_dtype)

        dtype_min, dtype_max = out_info.min, out_info.max

        if output_nodata is None:
            return None, dtype_min, dtype_max

        if isinstance(output_nodata, str):
            output_nodata = output_nodata.lower()
            if output_nodata == "min":
                nodata = dtype_min
            elif output_nodata == "max":
                nodata = dtype_max
            else:
                raise ValueError("output_nodata must be 'min', 'max', or a numeric value equal to the dtype min/max")
        else:
            nodata = output_nodata

        if np.issubdtype(out_dtype, np.floating) and np.isnan(nodata):
            return nodata, dtype_min, dtype_max

        if nodata == dtype_min:
            if np.issubdtype(out_dtype, np.integer):
                return nodata, dtype_min + 1, dtype_max
            return nodata, np.nextafter(dtype_min, dtype_max), dtype_max
        if nodata == dtype_max:
            if np.issubdtype(out_dtype, np.integer):
                return nodata, dtype_min, dtype_max - 1
            return nodata, dtype_min, np.nextafter(dtype_max, dtype_min)

        raise ValueError("output_nodata must be 'min', 'max', or a numeric value equal to the dtype min/max")

    def fit_transformation(self, out_dtype: np.dtype = np.uint8, output_nodata: str | float | int | None = None):
        out_nodata, out_min, out_max = self._resolve_output_nodata(out_dtype, output_nodata)

        transform_dict = {}
        for var, arr in self.data.data_vars.items():
            valid = arr.where(self.nodata_mask[var])

            data_min = valid.min().compute().item()
            data_max = valid.max().compute().item()

            transformation = TransformParameters(
                data_max = data_max,
                data_min = data_min,
                out_max = out_max,
                out_min = out_min,
                out_nodata = out_nodata,
                new_dtype = out_dtype
            )
            transform_dict[var] = transformation

        return Grid(
            data = self.data,
            original_location = self.original_location,
            nodata_mask = self.nodata_mask,
            original_nodata = self.original_nodata,
            original_dtype = self.original_dtype,
            transformation = transform_dict,
        )

    def transform(self):

        if self.transformation is None:
            raise ValueError("Fit transformation first before calling .transform()")

        scaled_arrays = []
        for var, arr in self.data.data_vars.items():

            if var not in self.transformation:
                logger.warning(f'No transformation stored for variable {var}, Will be skipped')
                continue

            transform = self.transformation[var]

            scaled = (arr - transform.data_min) / transform.denom
            scaled = scaled * (transform.out_max - transform.out_min) + transform.out_min

            if np.issubdtype(transform.new_dtype, np.integer):
                scaled = scaled.round()

            scaled = scaled.clip(transform.out_min, transform.out_max)

            fill_value = transform.out_nodata if transform.out_nodata is not None else transform.out_min
            if np.issubdtype(transform.new_dtype, np.integer):
                # Avoid invalid casts when NaN/Inf are present.
                scaled = scaled.where(np.isfinite(scaled), fill_value)

            scaled = scaled.astype(transform.new_dtype)

            # Preserve nodata
            scaled = scaled.where(self.nodata_mask[var], fill_value)

            scaled_arrays.append(scaled)

        scaled_ds = xr.merge(scaled_arrays)
        return Grid(
            data = scaled_ds,
            original_location = self.original_location,
            nodata_mask = self.nodata_mask[list(scaled_ds.keys())],
            original_nodata = {i:j for i,j in self.original_nodata.items() if i in scaled_ds.data_vars},
            original_dtype = {i:j for i,j in self.original_dtype.items() if i in scaled_ds.data_vars},
            transformation = {i:j for i,j in self.transformation.items() if i in scaled_ds.data_vars},
            transformed = True
        )

    def inverse_transform(self):

        if self.transformation is None:
            raise ValueError("Fit transformation first and then tranform before calling .inverse_transform()")

        if not self.transformed:
            raise ValueError("Transform first before calling .inverse_transform()")

        back_transformed_arrays = []
        for var, arr in self.data.data_vars.items():

            if var not in self.transformation:
                logger.warning(f"Could not find tranformation for {var}. Skipping")
                continue
            transform = self.transformation[var]

            if transform.out_nodata is None:
                inverse_mask = self.nodata_mask[var]
            elif np.issubdtype(arr.dtype, np.floating) and np.isnan(transform.out_nodata):
                inverse_mask = arr.notnull()
            else:
                inverse_mask = (arr != transform.out_nodata) & arr.notnull()

            restored = (arr.astype(np.float64) * transform.scale + transform.offset).astype(self.original_dtype[var])
            restored = xr.where(inverse_mask, restored, self.original_nodata[var])

            back_transformed_arrays.append(restored)

        restored_ds = xr.merge(back_transformed_arrays)
        return Grid(
            data=restored_ds,
            original_location=self.original_location,
            nodata_mask = self.nodata_mask[list(restored_ds.keys())],
            original_nodata = {i:j for i,j in self.original_nodata.items() if i in restored_ds.data_vars},
            original_dtype = {i:j for i,j in self.original_dtype.items() if i in restored_ds.data_vars},
            transformation = {i:j for i,j in self.transformation.items() if i in restored_ds.data_vars},
            transformed = True
        )

    def write(self, out_dir: str | Path, with_sidecar: bool = False):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents = True, exist_ok = True)

        for var, arr in self.data.data_vars.items():
            out_array = Path(f"{out_dir}/{var}.tif")
            out_sidecar = Path(f"{out_dir}/{var}_transformation.txt")

            if out_array.exists():
                raise ValueError(f"Cannot write {out_array}. File already exists")

            if with_sidecar and out_sidecar.exists():
                raise ValueError(f"Cannot write {out_sidecar}. File already exists")

            arr.rio.to_raster(out_array)
            if with_sidecar:
                if var not in self.transformation:
                    logger.warning(f"No transformation present for {var}. Sidecar cannot be written")
                else:
                    self.transformation[var].write(out_sidecar)

    def align(self, *others):
        pass

if __name__ == '__main__':

    original = Grid.from_geotiff("data/huglin_1981-2010_clip.tif")
    original = original.fit_transformation(output_nodata = 'min')
    transformed = original.transform()
    original2 = transformed.inverse_transform()

    transformed.write('data/transformed', with_sidecar = True)
    original2.data.rio.to_raster('data/transformed/original_back.tif')
