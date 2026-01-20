if __name__ == '__main__':
    import geopandas as gpd
    import matplotlib.pyplot as plt
    
    # Initialize country names 
    chosen_countries = ['Indonesia', 'Vietnam', 'Thailand', 'Malaysia', 'Philippines', 'Singapore', 'Myanmar', 'Cambodia', 'Laos', 'Brunei']

    # Read raw data of countries polygons
    gdf = gpd.read_file('./data/vector/ne_10m_admin_0_countries.zip')
    
    # Decide filter indexes
    idxs = gdf['NAME'].isin(chosen_countries)

    # Print filtered data
    print(gdf[idxs])
    gdf_filtered = gdf[idxs].copy()

    # Save as a new shapefile
    gdf_filtered.to_file('./data/vector/countries.zip')

    # Plot map
    fig, ax = plt.subplots()
    gdf_filtered.plot(ax=ax)                        # Plot polygons
    gdf_filtered.boundary.plot(ax=ax, color='k')    # Plot countries' boundary
    plt.show()