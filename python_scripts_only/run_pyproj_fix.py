import os
import geopandas as gpd
import pyproj

# Manually force the correct proj data path
os.environ["PROJ_LIB"] = "/home/kwameatt/anaconda3/envs/cs231n/share/proj"
pyproj.datadir.set_data_dir("/home/kwameatt/anaconda3/envs/cs231n/share/proj")
gdf = gpd.GeoDataFrame(
    {"LON": [10], "LAT": [50]},
    geometry=gpd.points_from_xy([10], [50]),
    crs="EPSG:4326"
)

print("Success! CRS is:", gdf.crs)
