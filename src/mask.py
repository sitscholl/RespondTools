import geopandas as gpd
import xarray as xr
import rioxarray
from rasterio.features import geometry_mask

import logging

logger = logging.getLogger(__name__)

class ArrayMasker:
    def __init__(self, aoi: gpd.GeoDataFrame):

        if len(aoi.geometry) > 1:
            logger.warning("Multiple regions in aoi are not supported. Dataframe will be dissolved to a single region")
            aoi = aoi.dissolve()

        self.aoi = aoi

    def create_mask(self, ds: xr.DataArray | xr.Dataset, all_touched: bool = True, invert: bool = True, x_dim: str = 'x', y_dim: str = 'y'):

        if ds.rio.crs is None:
            raise ValueError('ds has no coordinate system. Cannot create mask')

        if (x_dim not in ds.dims) or (y_dim not in ds.dims):
            raise ValueError(f"ds must contain x_dim and y_dim in dims. Got {ds.dims}")

        # Ensure the region is in the same CRS as the elevation data
        if self.aoi.crs != ds.rio.crs:
            region = self.aoi.to_crs(ds.rio.crs)
        else:
            region = self.aoi

        geom = [region.geometry.iloc[0]]
        
        # Create a mask from the geometries
        region_mask = geometry_mask(
            geom, 
            out_shape=(ds.sizes[y_dim], ds.sizes[x_dim]), 
            transform=ds.rio.transform(), 
            all_touched=all_touched,
            invert=invert
        )
        
        # Convert the mask to a DataArray with the same coordinates as the elevation data
        region_arr = xr.DataArray(
            region_mask, 
            dims=(y_dim, x_dim),
            coords={y_dim: ds[y_dim], x_dim: ds[x_dim]},
            attrs = ds.attrs
        )
        region_arr = region_arr.rio.write_crs(ds.rio.crs)
        
        return region_arr

    def clip(self, ds: xr.DataArray | xr.Dataset, all_touched: bool = True, invert: bool = False):

        # Ensure the region is in the same CRS as the elevation data
        if self.aoi.crs != ds.rio.crs:
            region = self.aoi.to_crs(ds.rio.crs)
        else:
            region = self.aoi

        geom = [region.geometry.iloc[0]]
        return ds.rio.clip(geom, all_touched=all_touched, drop=True, invert = invert)