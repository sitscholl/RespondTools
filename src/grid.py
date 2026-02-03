import xarray as xr
import rioxarray
import numpy as np

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Grid:
    data: xr.Dataset | xr.DataArray
    original_location: str | Path
    original_dtype: np.dtype | None = None
    scale: float | None = None
    offset: float | None = None

    @classmethod
    def from_geotiff(cls, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Could not find geotiff file at {path}")
        data = xr.open_dataarray(path).squeeze(drop = True)
        return cls(data, path)

    @property
    def crs(self):
        return self.data.rio.crs.to_epsg() if self.data.rio.crs else None

    @property
    def resolution(self):
        return self.data.rio.resolution()

    def scale_array(self, da: xr.DataArray, out_dtype, nodata = None, dim = None) -> xr.DataArray:
        out_info = np.iinfo(out_dtype)
        out_min, out_max = out_info.min, out_info.max

        valid = da if nodata is None else da.where(da != nodata)

        if dim and dim in da.dims:
            data_min = valid.min(dim=["y", "x"])
            data_max = valid.max(dim=["y", "x"])
        else:
            data_min = valid.min()
            data_max = valid.max()

        # Avoid divide-by-zero if data is constant
        denom = data_max - data_min
        denom = xr.where(denom == 0, 1, denom)

        scaled = (da - data_min) / denom
        scaled = scaled * (out_max - out_min) + out_min
        scaled = scaled.clip(out_min, out_max)

        if np.issubdtype(out_dtype, np.integer):
            # Avoid invalid casts when NaN/Inf are present.
            scaled = scaled.where(np.isfinite(scaled), out_min)

        scaled = scaled.astype(out_dtype)

        # Preserve nodata
        if nodata is not None:
            scaled = scaled.where(da != nodata, out_min)

        # Store scale/offset as attrs for inversion
        scaled.attrs["scale"] = (data_max - data_min) / (out_max - out_min)
        scaled.attrs["offset"] = data_min - out_min * scaled.attrs["scale"]
        return scaled

    def transform(self, out_dtype = np.uint8, dim = None):

        data = self.data

        # Handle nodata if present via rioxarray
        nodata = getattr(data.rio, "nodata", None)

        if isinstance(data, xr.Dataset):
            transformed = data.map(lambda x: self.scale_array(x, out_dtype = out_dtype, nodata = nodata, dim = dim))
            scale = None
            offset = None
        else:
            transformed = self.scale_array(data, out_dtype = out_dtype, nodata = nodata, dim = dim)
            scale = transformed.attrs.get("scale")
            offset = transformed.attrs.get("offset")

        return Grid(
            data=transformed,
            original_location=self.original_location,
            original_dtype=self.data.dtype,
            scale = scale,
            offset = offset
        )

    def inverse_transform(self, out_dtype: np.dtype | None = None):
        data = self.data
        target_dtype = out_dtype or self.original_dtype

        def _inverse(da: xr.DataArray) -> xr.DataArray:
            scale = da.attrs.get("scale", self.scale)
            offset = da.attrs.get("offset", self.offset)
            if scale is None or offset is None:
                raise ValueError("Missing scale/offset for inverse_transform.")
            restored = da.astype(np.float64) * scale + offset
            if target_dtype is not None and np.issubdtype(target_dtype, np.integer):
                restored = restored.where(np.isfinite(restored), 0)
            if target_dtype is not None:
                restored = restored.astype(target_dtype)
            return restored

        if isinstance(data, xr.Dataset):
            restored = data.map(_inverse)
        else:
            restored = _inverse(data)

        return Grid(
            data=restored,
            original_location=self.original_location,
            original_dtype=target_dtype,
            scale=None,
            offset=None,
        )

    def align(self, *others):
        pass

if __name__ == '__main__':

    original = Grid.from_geotiff("huglin_1981-2010_clip.tif")
    transformed = original.transform()
    original2 = transformed.inverse_transform(out_dtype = original.data.dtype)

    transformed.data.rio.to_raster('transformed.tif')
    original2.data.rio.to_raster('original2.tif')
