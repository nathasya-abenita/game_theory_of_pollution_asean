import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import rioxarray
from shapely.geometry import Point, LineString
import rasterio.features
import array_to_latex as a2l

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
    plt.savefig('./output/figs/preview_countries.png')

    """ 1.2. Read population count raster data """

    raster_pop = rioxarray.open_rasterio("./data/raster/count_15_min.tif", masked=True).squeeze()
    raster_pop = raster_pop.rio.reproject(crs_long) # reprojecting

    # Prepare population data and coordinates
    x = raster_pop.x.values
    y = raster_pop.y.values
    X, Y = np.meshgrid(x, y)

    # # Plotting
    # raster_pop.plot(robust=True)
    # plt.xlim(-0.2e7, 0.55e7)
    # plt.ylim(-0.2e7, 0.32e7)
    # plt.title("Gridded population count")
    # plt.savefig('./output/figs/population_preview.png')

    """ 2. Create graph vertices """

    centroids_population, lines = create_graph_vertices(raster_pop, X, Y, gdf_countries)

    # Plotting
    base = gdf_countries.plot(cmap = "pink")
    gdf_countries.centroid.plot(ax = base, marker = 'o', color = 'blue', markersize = 15, label = "Basic centroid")
    centroids_population.plot(ax = base, marker = 'o', color = 'red', markersize = 15, label = "Population-weighted centroid")
    lines.plot(ax = base, color = "orange")

    # Decoration
    plt.title("Centroids of ASEAN Countries")
    plt.axis("off")
    plt.legend(fontsize = 8)
    plt.savefig('./output/figs/centroids.png')

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
            # Compute mean and save it
            u_weighted, v_weighted = compute_wind_speed_mean(geom, raster_wind_u, raster_wind_v, raster_pop)
            wind_vector_list.append([u_weighted, v_weighted])

        # Save information
        wind_vector_per_month_list.append(wind_vector_list)

    """ 3.3. Plot wind vectors """

    # Plot vertex
    fig, axs = plt.subplots(4,3, figsize=(10,10),tight_layout=True)
    for i in range (4):
        for j in range (3):
            gdf_countries.plot(ax = axs[i][j], cmap = "pink") # cmap pink or tab10
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
                    color='darkturquoise') # white
            # Add legend of the quiver
            legend_arrow_length = 1
            qk = ax.quiverkey(qv, 1, 1, legend_arrow_length, rf'{legend_arrow_length} m/s', labelpos='E',
                   coordinates='axes')
        # Modify plotting style
        ax.axis("off")
        ax.set_title(time_int_strings[time_int_id], fontsize=10)
    plt.tight_layout()
    plt.savefig('./output/figs/network.png')

    """ 3.4. Save wind vectors values """
    np.save("./output/vars/wind_vector_per_month_list.npy", wind_vector_per_month_list)

    """ 4.1. Compute psi_{ij} or graph's weight """

    psi_list = [] # indexes: month, country_i, contry_j
    # Iterate over each month
    for month_id in range (12): # 12 months
        # Initialize 2-by-2 matrix for country-i ot country-j
        psi = np.zeros((len(gdf_countries), len(gdf_countries)))
        for country_id_i in range (len(gdf_countries)):
            for country_id_j in range (len(gdf_countries)):
                if country_id_i == country_id_j:
                    p = 0
                else:
                    p_speed = compute_speed_proportion (wind_vector_per_month_list, month_id, country_id_i)
                    p_angle = compute_angle_proportion (wind_vector_per_month_list, centroids_population, month_id, country_id_i, country_id_j)
                    p_distance = compute_distance_proportion (centroids_population, country_id_i, country_id_j)
                    p = p_speed * p_angle * p_distance
                psi[country_id_i][country_id_j] = p
        # Save matrix for that month
        psi_list.append(psi)
    
    """ 4.2. Save psi list  """
    np.save("./output/vars/psi_list.npy", psi_list)

    """ 4.3. Print psi list in LaTeX """

    with open("./output/vars/psi_list.txt", "w") as f:
        for i in range(12):
            f.write(f"month {i+1}\n")
            latex_str = a2l.to_ltx(
                psi_list[i],
                frmt='{:6.2f}',
                arraytype='bmatrix',
                print_out=False
            )
            f.write(latex_str + "\n\n")

    