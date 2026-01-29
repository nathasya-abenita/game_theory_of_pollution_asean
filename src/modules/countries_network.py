import numpy as np
import geopandas as gpd
import rioxarray
from shapely.geometry import Point, LineString
import rasterio.features


# plt.style.use('dark_background') # use dark mode style

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

def compute_wind_speed_mean (geom, raster_wind_u, raster_wind_v, raster_pop):
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
    return u_weighted, v_weighted

def compute_speed_proportion (wind_vector_list_per_month, month_id, country_id, s_max = 4.3):
    wind_vector = np.array(wind_vector_list_per_month[month_id][country_id])
    speed = np.sqrt(np.sum(wind_vector**2))
    if speed < s_max:
        p = speed/s_max
    else:
        p = 1
    return p

def compute_angle_proportion (wind_vector_list_per_month, centroids, month_id, country_id_i, country_id_j):
    # Take wind vector
    w = wind_vector_list_per_month[month_id][country_id_i]
    w_x = w[0]
    w_y = w[1]

    # Compute distances on x- and y-axis
    x_delta = centroids[country_id_j].x - centroids[country_id_i].x
    y_delta = centroids[country_id_j].y - centroids[country_id_i].y

    # Compute dot product
    dot_p = (w_x * x_delta) + (w_y * y_delta)
    dot_p = dot_p / np.sqrt((w_x**2+w_y**2)*(x_delta**2+y_delta**2))

    # Apply function for angle proportion based on dot product
    if dot_p >= 0:
        p = dot_p
    else:
        p = 0
    return p

def compute_distance_proportion (centroids, country_id_i, country_id_j, d_max = 1_964_972.7):

    x_delta = centroids[country_id_i].x - centroids[country_id_j].x
    y_delta = centroids[country_id_i].y - centroids[country_id_j].y
    norm = np.sqrt(x_delta**2 + y_delta**2)

    if norm <= d_max:
        p = np.cos(norm/d_max * np.pi/2)
        #p = norm/d_max
    else:
        p = 0
    return p