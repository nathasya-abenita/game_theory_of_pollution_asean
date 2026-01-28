import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import rioxarray
from shapely.geometry import Point, LineString
import rasterio.features
import array_to_latex as a2l

plt.style.use('dark_background') # use dark mode style

crs_short = 32649
crs_long = "EPSG:32649"

def create_gdf_countries(filename: str, country_codes: dict) -> gpd.GeoDataFrame:

    # Read countries vector file
    gdf_countries = gpd.read_file(filename)
    gdf_countries = gdf_countries.to_crs(crs_long)

    # Reindexing GDF
    gdf_countries['index'] = gdf_countries['NAME'].map(country_codes)
    gdf_countries.sort_values(by='index', inplace=True)
    gdf_countries = gdf_countries.set_index('index', drop=False)

    return gdf_countries

def create_graph_vertices(raster, X, Y, gdf_countries):

    # Initialize centroids
    centroids = gdf_countries.centroid.values   # basic centroids based on polygons
    centroids_population = []                   # population-weighted centroids
    lines = []

    # Iterate over each polygon and its centroid
    for geom, centroid in zip(gdf_countries.geometry, centroids):

        mask = rasterio.features.rasterize(
            [(geom, 1)],
            out_shape=raster.data.shape,
            transform=raster.rio.transform(),
            fill=0,
            all_touched=True
        ).astype(bool)

        valid = mask & (raster.data >= 0)

        w = raster.data[valid]
        cx = (w * X[valid]).sum() / w.sum()
        cy = (w * Y[valid]).sum() / w.sum()

        pwc = Point(cx, cy)
        centroids_population.append(pwc)
        lines.append(LineString([centroid, pwc]))

    # Convert outputs to GeoSeries
    centroids_population = gpd.GeoSeries(centroids_population, crs=gdf_countries.crs)
    lines = gpd.GeoSeries(lines, crs=gdf_countries.crs)
    return centroids_population, lines

def read_wind_raster_specific (str_name, str_code): # speed-1.tiff
    # Define file name
    str_final = str_name + "-" + str_code
    str_final = str_final + ".tiff"

    # Read data
    raster = rioxarray.open_rasterio("./data/raster_wind_850mb/" + str_final, masked=False).squeeze()
    raster = raster.rio.reproject(crs_long)
    return raster

def read_wind_raster_monthly (i):
    raster_u_arr = read_wind_raster_specific("u", str(i))
    raster_v_arr = read_wind_raster_specific("v", str(i))
    return raster_u_arr, raster_v_arr

if __name__ == '__main__':    
    """ 1.1 Read countries polygons and set index """

    # Standard indexing
    country_codes = {'Indonesia' : 0, 'Vietnam' : 1, 'Thailand' : 2, 'Malaysia' : 3, 
                     'Philippines' : 4, 'Singapore' : 5, 'Myanmar' : 6, 'Cambodia' : 7, 
                     'Laos' : 8, 'Brunei' : 9}
    
    # Read and plot
    gdf_countries = create_gdf_countries(filename="./data/vector/countries.shp", country_codes=country_codes)
    gdf_countries.plot(cmap='tab10', column='NAME', legend=True)
    plt.title('preview of countries')
    plt.show()

    """ 1.2. Read population count raster data """

    raster_pop = rioxarray.open_rasterio("./data/raster/count_15_min.tif", masked=True).squeeze()
    raster_pop = raster_pop.rio.reproject(crs_long) # reprojecting

    # Prepare population data and coordinates
    x = raster_pop.x.values
    y = raster_pop.y.values
    X, Y = np.meshgrid(x, y)

    # Plotting
    raster_pop.plot(robust=True)
    plt.xlim(-0.2e7, 0.55e7)
    plt.ylim(-0.2e7, 0.32e7)
    plt.title("Gridded population count")
    plt.show()

    """ 2. Create graph vertices """

    centroids_population, lines = create_graph_vertices(raster_pop, X, Y, gdf_countries)

    # Plotting
    base = gdf_countries.plot(cmap = "tab10")
    gdf_countries.centroid.plot(ax = base, marker = 'o', color = 'yellow', markersize = 15, label = "Basic centroid")
    centroids_population.plot(ax = base, marker = '*', color = 'yellow', markersize = 15, label = "Population-weighted centroid")
    lines.plot(ax = base, color = "yellow")

    # Decoration
    plt.title("Centroids of ASEAN Countries")
    plt.axis("off")
    plt.legend(fontsize = 8)
    plt.show()

    """ 3.1. Read resampled population count raster data """

    raster_pop = rioxarray.open_rasterio("./data/raster/resampled_count_15_min.tiff", masked=True).squeeze()
    raster_pop = raster_pop.rio.reproject(crs_long) # reprojecting

    """ 3.2. Read wind speed for each country """

    wind_vector_per_month_list = []

    # Iterate over each month
    for start in range (1,12+1):

        # Preparing raster data of wind
        raster_wind_u, raster_wind_v = read_wind_raster_monthly (start) 

        # Iterate over each country's polygon
        wind_vector_list = []
        for geom in gdf_countries.geometry:

            mask = rasterio.features.rasterize(
                [(geom, 1)],
                out_shape=raster_wind_u.data.shape,
                transform=raster_wind_u.rio.transform(),
                fill=0,
                all_touched=True
            ).astype(bool)

            valid = mask & (raster_pop.data >= 0)

            # Compute u-component
            w = raster_pop.data[valid]
            u_weighted = (w * raster_wind_u.data[valid]).sum() / w.sum()
            v_weighted = (w * raster_wind_v.data[valid]).sum() / w.sum()

            wind_vector_list.append([u_weighted, v_weighted])

        # Save information
        wind_vector_per_month_list.append(wind_vector_list)

    """ 3.3. Plot wind vectors """

    # Plot vertex
    fig, axs = plt.subplots(4,3, figsize=(10,10),tight_layout=True)
    for i in range (4):
        for j in range (3):
            gdf_countries.plot(ax = axs[i][j], cmap = "tab10") # cmap pink juga bagus
            centroids_population.plot(ax = axs[i][j], marker = 'o', color = 'blue', markersize = 20)

    
    ax_id_list = [[0,0], [0,1], [0,2],
                [1,0], [1,1], [1,2],
                [2,0], [2,1], [2,2],
                [3,0], [3,1], [3,2]]
    time_int_strings = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Iterate over each month
    for time_int_id in range (12):
        ax = axs[ax_id_list[time_int_id][0]][ax_id_list[time_int_id][1]]

        # Iterate over each country
        for i in range (len(centroids_population)):
            # Plot wind vector
            qv = ax.quiver(centroids_population[i].x, centroids_population[i].y,
                    wind_vector_per_month_list[time_int_id][i][0],
                    wind_vector_per_month_list[time_int_id][i][1],
                    color='white') # darkturquoise
            # Add legend of the quiver
            qk = ax.quiverkey(qv, 1, 1, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
        # Modify plotting style
        ax.axis("off")
        ax.set_title(time_int_strings[time_int_id], fontsize=10)
    plt.tight_layout()
    plt.show()