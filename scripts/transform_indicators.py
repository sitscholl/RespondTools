from pathlib import Path
import logging
import sys

from src.grid import Grid
from src.mask import ArrayMasker

logger = logging.getLogger(__name__)

def transform_indicators(
    files: list[str | Path], 
    out_dir: str | Path, 
    output_nodata = 'min', 
    with_sidecar = True, 
    region_mask: ArrayMasker | None = None
    ):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    for file in files:
        try:
            logger.info(f"Processing file {file}")

            file = Path(file)
            if not file.exists():
                logger.warning(f"File {file} does not exist. Skipping...")
                continue

            grid = Grid.from_geotiff(file, mask = region_mask)
            grid = grid.fit_transformation(output_nodata = output_nodata)
            transformed = grid.transform()

            transformed.write(out_dir, with_sidecar = with_sidecar)
        except Exception as e:
            logger.exception(f"Error processing file {file}: {e}")


if __name__ == '__main__':

    from src.utils import load_config
    import geopandas as gpd

    logging.basicConfig(level = logging.INFO, force = True)

    config = load_config('config.yaml')
    
    input_dir = Path(config['transformation']['input'])
    out_dir = Path(config['transformation']['output'])
    pat = config['transformation']['file_pattern']

    files = list(input_dir.glob(pat))
    if len(files) == 0:
        logger.warning(f'No geotiff files found in {input_dir}')
        sys.exit(1)

    logger.info(f'Found {len(files)} files to process')

    region_mask = config['transformation'].get('region_mask')
    if region_mask is not None:
        region_gdf = gpd.read_file(region_mask)
        region_mask = ArrayMasker(region_gdf)
    else:
        logger.info('No region mask provided; arrays will not be masked')

    transform_indicators(files, out_dir = out_dir, region_mask = region_mask)
