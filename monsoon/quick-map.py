import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_map():
    # Create a new figure
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Define bounds of the map (for India)
    ax.set_extent([67, 98, 5, 40])
    
    # Add country borders
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Add coastlines and land
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    
    # Add gridlines for latitudes and longitudes
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Set the title
    ax.set_title("Detailed Map of India")
    
    # Display the plot
    plt.show()

plot_map()
