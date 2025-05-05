import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Step 1: Define your grid bounding box
lat_min, lat_max = 40.70, 40.88
lon_min, lon_max = -74.02, -73.91
lat_step = 0.0005
lon_step = 0.0005

# Step 2: Create mesh grid of coordinates
lat_grid = np.arange(lat_min, lat_max, lat_step)
lon_grid = np.arange(lon_min, lon_max, lon_step)
lon_coords, lat_coords = np.meshgrid(lon_grid, lat_grid)
grid_points = np.vstack([lat_coords.ravel(), lon_coords.ravel()]).T

# Step 3: Convert grid points to GeoDataFrame
grid_gdf = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in grid_points],
    crs="EPSG:4326"
)

# Step 4: Load borough shapefile
nyc = gpd.read_file("nybb.shp")
manhattan = nyc[nyc['BoroName'].str.lower() == 'manhattan'].to_crs("EPSG:4326")

# Step 5: Filter grid points within Manhattan
manhattan_polygon = manhattan.geometry.union_all()
points_in_manhattan = grid_gdf[grid_gdf.geometry.within(manhattan_polygon)].copy()

# Step 6: Load and clean Airbnb data
df = pd.read_excel("../data/Airbnb_clean_dist_cent_sub.xlsx")
df_manhattan = df[df['neighbourhood group'].str.lower() == 'manhattan'].copy()
df_manhattan['price'] = df_manhattan['price'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

# Step 7: Convert listings to GeoDataFrame
gdf_airbnb = gpd.GeoDataFrame(
    df_manhattan,
    geometry=[Point(xy) for xy in zip(df_manhattan['long'], df_manhattan['lat'])],
    crs="EPSG:4326"
)

# Step 8: Project to a metric CRS
projected_crs = "EPSG:2263"
gdf_airbnb_proj = gdf_airbnb.to_crs(projected_crs)
manhattan_proj = manhattan.to_crs(projected_crs)

# Step 9: Randomly sample 200 Airbnb listings
gdf_sampled = gdf_airbnb_proj.sample(n=3).copy()
gdf_sampled.reset_index(drop=True, inplace=True)

# Step 10: Compute pairwise distances (in meters)
coords = np.array([[geom.x, geom.y] for geom in gdf_sampled.geometry])
dist_matrix_meters = distance_matrix(coords, coords) * 0.3048
# Step 11: Plot sampled points
fig, ax = plt.subplots(figsize=(12, 12))
manhattan_proj.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
gdf_sampled.plot(ax=ax, color='blue', markersize=10, label="Sampled Listings")

ax.set_title("Sampled Airbnb Listings in Manhattan (200 Points)")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
ax.set_aspect("equal")
plt.legend()
plt.show()

# Step 12: (Optional) Print or store distance matrix
# For example, print the first 5 distances of the first point
print("Example distances from first point (in meters):")
print(dist_matrix_meters[0][:5])
