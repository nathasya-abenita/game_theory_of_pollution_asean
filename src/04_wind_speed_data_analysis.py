import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    """
    1. Loading wind speed data 

    time interval 
    ```wind_vector_list_over_intervals[i]``` : $i=0,1,2,3$ 

    countries
    ```wind_vector_list_over_intervals[i][j]``` : $j=0,...,9$

    u, v value
    ```wind_vector_list_over_intervals[i][j][k]``` : $k=0,1$

    """

    vector_per_month_array = np.load("./output/vars/wind_vector_per_month_list.npy")
    n_country              = len(vector_per_month_array[0])

    """ 2. Compute speed from vector per month list """

    # Iterate over each month
    speed_per_month_array = np.empty((len(vector_per_month_array), len(vector_per_month_array[0])))
    for i in range (len(vector_per_month_array)):
        # Iterate over each country
        speed_per_month = []
        for j in range (len(vector_per_month_array[0])):
            u = vector_per_month_array[i][j][0]
            v = vector_per_month_array[i][j][1]
            # Save information
            speed_per_month_array[i, j] = np.sqrt(u**2 + v**2)

    """ 3. Print statistics of the speed """
    print("max:", np.max(speed_per_month_array))
    print("min:", np.min(speed_per_month_array))
    print("mean:", np.mean(speed_per_month_array))

    """ 4. Create monthly wind speed time series for each country """

    # Month number axis
    month_list = [i for i in range (1,12+1)]

    # Plotting line chart
    fig, ax = plt.subplots(figsize = (7,4))

    # Define country names
    country_names = ["Indonesia",
                    "Vietnam",
                    "Thailand",
                    "Malaysia",
                    "The Philippines",
                    "Singapore",
                    "Myanmar",
                    "Cambodia",
                    "Laos",
                    "Brunei"]

    # Plot line for each country
    for i in range (n_country):
        plt.plot(month_list, speed_per_month_array[:, i], label=country_names[i])
    
    # Decorate plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 8); plt.xlim(1, 12)
    plt.xlabel('Month Index'); plt.ylabel('Wind Speed (m/s)')
    plt.title("Monthly Wind Speed")
    plt.grid()
    plt.tight_layout()
    plt.savefig('./output/figs/wind_speed_time_series.png')

    """ 5. Create histogram """

    # Creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.25, .85)})

    # Assigning a graph to each ax
    ax_hist.hist(speed_per_month_array.flatten(), bins = 8, range = (0,8))
    bp = ax_box.boxplot(speed_per_month_array.flatten(), vert = False)

    # Removing borders
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    ax_box.spines['left'].set_visible(False)
    ax_box.set_yticks([])

    ax_hist.grid()
    plt.ylabel("Frequency")
    plt.xlabel("Speed (m/s)")
    plt.xlim(0,8)
    plt.suptitle("Wind Speed Boxplot and Histogram")
    plt.savefig('./output/figs/wind_speed_boxplot_hist.png')

    # Printing box plot values
    print('median:', np.median(speed_per_month_array))
    print('upper quartile:', np.percentile(speed_per_month_array, 75))
    print('lower quartile:', np.percentile(speed_per_month_array, 25))
    print('q80:', np.percentile(speed_per_month_array, 80))

    
