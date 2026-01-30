if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import array_to_latex as a2l
    from modules.countries_network import *

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

    