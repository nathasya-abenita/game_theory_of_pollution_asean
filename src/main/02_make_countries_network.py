import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import rioxarray
from shapely.geometry import Point, LineString
import rasterio.features
import array_to_latex as a2l

crs_short = 32649
crs_long = "EPSG:32649"

if __name__ == '__main__':

    """ 1.1 Read countries polygons and set index """

    # Read countries vector file
    gdf_countries = gpd.read_file("./data/vector/countries.shp")
    gdf_countries = gdf_countries.to_crs(crs_long)

    # Standard indexing
    country_codes = {'Indonesia' : 0, 'Vietnam' : 1, 'Thailand' : 2, 'Malaysia' : 3, 
                     'Philippines' : 4, 'Singapore' : 5, 'Myanmar' : 6, 'Cambodia' : 7, 
                     'Laos' : 8, 'Brunei' : 9}
    
    # Reindexing GDF
    gdf_countries['index'] = gdf_countries['NAME'].map(country_codes)
    gdf_countries.sort_values(by='index', inplace=True)
    gdf_countries = gdf_countries.set_index('index', drop=False)
    print(gdf_countries['NAME'])

    # Plotting
    gdf_countries.plot(cmap='tab10', column='NAME', legend=True)
    plt.title('preview of countries')
    plt.show()

    """ 1.2. Read population count raster data """

    raster = rioxarray.open_rasterio("./data/raster/count_15_min.tif", masked=True).squeeze()
    raster = raster.rio.reproject(crs_long) # reprojecting
    raster.plot(robust=True)
    plt.xlim(-0.2e7, 0.55e7)
    plt.ylim(-0.2e7, 0.32e7)
    plt.title("Gridded population count")
    plt.show()

    """ 2. Create graph vertices """

    # Precompute
    pop = raster.data
    x = raster.x.values
    y = raster.y.values
    X, Y = np.meshgrid(x, y)

    centroids = gdf_countries.centroid.values
    centroids_population = []
    lines = []

    for geom, centroid in zip(gdf_countries.geometry, centroids):

        mask = rasterio.features.rasterize(
            [(geom, 1)],
            out_shape=pop.shape,
            transform=raster.rio.transform(),
            fill=0,
            all_touched=True
        ).astype(bool)

        valid = mask & (pop >= 0)

        w = pop[valid]
        cx = (w * X[valid]).sum() / w.sum()
        cy = (w * Y[valid]).sum() / w.sum()

        pwc = Point(cx, cy)
        centroids_population.append(pwc)
        lines.append(LineString([centroid, pwc]))

    # Convert outputs to GeoSeries
    centroids_population = gpd.GeoSeries(centroids_population, crs=gdf_countries.crs)
    lines = gpd.GeoSeries(lines, crs=gdf_countries.crs)

    # Plotting
    base = gdf_countries.plot(cmap = "pink")
    gdf_countries.centroid.plot(ax = base, marker = 'o', color = 'red', markersize = 15, label = "Basic centroid")
    centroids_population.plot(ax = base, marker = 'o', color = 'blue', markersize = 15, label = "Population-weighted centroid")
    lines.plot(ax = base, color = "orange")

    # Decoration
    plt.title("Centroids of ASEAN Countries")
    plt.axis("off")
    plt.legend(fontsize = 8)
    plt.show()

    print("Centroid for each countries:")
    print(centroids_population)