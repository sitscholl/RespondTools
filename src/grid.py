import xarray as xr
import rioxarray
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class TransformParameters:
    data_max: float | int
    data_min: float | int
    out_max: float | int
    out_min: float | int
    out_nodata: float | int | None
    original_dtype: np.dtype
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

@dataclass
class Grid:
    data: xr.Dataset
    original_location: str | Path
    nodata_mask: xr.Dataset
    nodata: Dict[str, float | int | None]
    transformation: Dict[str, TransformParameters] | None = None
    output_nodata_choice: str | float | int | None = None
    output_nodata: float | int | None = None

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
            data = data.to_dataset(name = data.name or 'variable')

        nodata_mask = []
        nodata_vals = {}
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

        return cls(data, path, xr.merge(nodata_mask), nodata_vals)

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
                original_dtype = arr.dtype,
                new_dtype = out_dtype
            )
            transform_dict[var] = transformation

        return Grid(
            data = self.data,
            original_location = self.original_location,
            nodata_mask = self.nodata_mask,
            nodata = self.nodata,
            transformation = transform_dict,
            output_nodata_choice = output_nodata,
            output_nodata = out_nodata
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
            nodata = {i:j for i,j in self.nodata.items() if i in scaled_ds.data_vars},
            transformation = {i:j for i,j in self.transformation.items() if i in scaled_ds.data_vars},
            output_nodata_choice = self.output_nodata_choice,
            output_nodata = self.output_nodata
        )

    # def inverse_transform(self, out_dtype: np.dtype | None = None):
    #     data = self.data
    #     target_dtype = out_dtype or self.original_dtype
    #     nodata = self.nodata
    #     if nodata is None:
    #         nodata = getattr(data.rio, "nodata", None)

    #     def _inverse(da: xr.DataArray) -> xr.DataArray:
    #         scale = da.attrs.get("scale", self.scale)
    #         offset = da.attrs.get("offset", self.offset)
    #         if scale is None or offset is None:
    #             raise ValueError("Missing scale/offset for inverse_transform.")
    #         restored = da.astype(np.float64) * scale + offset
    #         if target_dtype is not None and np.issubdtype(target_dtype, np.integer):
    #             restored = restored.where(np.isfinite(restored), 0)
    #         if target_dtype is not None:
    #             restored = restored.astype(target_dtype)
    #         if nodata is not None:
    #             restored = restored.where(np.isfinite(restored), nodata)
    #         return restored

    #     if isinstance(data, xr.Dataset):
    #         restored = data.map(_inverse)
    #     else:
    #         restored = _inverse(data)

    #     return Grid(
    #         data=restored,
    #         original_location=self.original_location,
    #         original_dtype=target_dtype,
    #         scale=None,
    #         offset=None,
    #         nodata=nodata,
    #     )

    def align(self, *others):
        pass

if __name__ == '__main__':

    original = Grid.from_geotiff("huglin_1981-2010_clip.tif")
    transformed = original.transform()
    original2 = transformed.inverse_transform(out_dtype = original.data.dtype)

    transformed.data.rio.to_raster('transformed.tif')
    original2.data.rio.to_raster('original2.tif')
