from __future__ import division
import numpy as np
import pickle
import torch, random, copy
import torch.optim as optim
from torch import nn
from sh_utils_local import *
from basic_structure_local import IGNNK, IGNNK_ModOptionB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import pandas as pd
import re
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
import geopandas as gpd
from datetime import datetime

import matplotlib.pyplot as plt


set_seed(42)

# Load training data = Anchor + grid nodes
grid_gdf_path = "/mnt/e/GNN_paper/2025_IGNNK/data/grids_train/all_grid_data_1p0r.csv"
anchor_data_path = "/mnt/e/GNN_paper/2025_IGNNK/data/anchors_train/WSM2025_EU.csv"

#generate, clean, and process data, then construct the graph adjacency and feature matrix 
sh_dat = generate_shmax_train_data(anchor_data_path, grid_gdf_path)
A, X_target, grid_gdf, anchor_gdf, test_anchor_data, node_info = load_shmax_train_data()

df_cols = ['ID', 'LON', 'LAT', 'DEPTH', 'DIST', 'PLATE', 'QUALITY', 'REGIME', 'AZI', 'geometry']
grid_gdf = grid_gdf[df_cols] 
anchor_gdf = anchor_gdf[df_cols]
node_info = node_info[df_cols]
test_anchor_data = test_anchor_data[df_cols]

# plates_gdf.boundary.plot(edgecolor='black')
grid_gdf.plot(column='PLATE', markersize=5, legend=True)
# grid_gdf.plot(column='BOUNDARY', markersize=5, legend=True)
plt.show()