import pandas as pd
import numpy as np 
import scipy.sparse as sp  
import torch
from math import radians, cos, sin, asin, sqrt
import os, yaml
import re 
import pickle
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import geopandas as gpd
from typing import Union 

os.environ["PROJ_LIB"] = "/home/kwameatt/anaconda3/envs/cs231n/share/proj"

STRESS_DIR = 'data/wsm2016'


# These files were obtained from Bird (2002)
PB2002_boundary_path = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/PB2002_boundaries.dig.txt"
PB2002_plates_path = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/PB2002_plates.dig.txt"
PB2002_orogen_path = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/PB2002_orogens.dig.txt"
PB2002_steps = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/PB2002_step.dat.txt"
### these are also obtained from the mordenized version of PB2002 model

###333333===========

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def encode_azimuth(theta_degrees):
    """Encodes azimuth angle into two channels cos/sin"""
    theta_radians = np.radians(2 * theta_degrees)
    return np.cos(theta_radians), np.sin(theta_radians)

def decode_azimuth(cos_2theta, sin_2theta):
    """
    decodes the final azimuth into a single channel of theta. 
    """
    angles_2 =np.arctan2(sin_2theta, cos_2theta)     # shape [N], radians
    angles_2_deg = angles_2 * 180.0 / np.pi          # shape [N], degrees in [-180,180]
    angles_shmax = (angles_2_deg % 360.0) / 2.0      # shape [N] in [0,180]

    return angles_shmax

# AZI_COLS = ("AZI", "ENC_COS", "ENC_SIN")

# def get_wsm_dataset_dataframe(csv_path: str) -> pd.DataFrame:
#     """
#     Read the WSM-2016 CSV and return a clean anchor-node DataFrame
#     with columns
#         ['id','longitude','latitude','depth_m',
#          'azimuth_deg','enc_cos','enc_sin',
#          'regime','plate','dist','boundary'].
#     """

#     needed_col = ["ID", "LON", "LAT", "DEPTH", "AZI",
#             "REGIME", "DIST", "PLATE", "BOUNDARY"]
#     df = pd.read_csv(csv_path, encoding='ISO-8859-1', usecols=needed_col)

#     df = df.dropna(subset=["AZI", "LON", "LAT", "PLATE"])        # essential fields only
#     df = df[(df["AZI"] <= 180.0)]   # and >180 degrees

#     df = df.astype({
#         "ID": str,
#         "LON": float,
#         "LAT": float,
#         "DEPTH": float,
#         "AZI": float,
#         "REGIME": str,
#         "DIST": float,
#         "PLATE": str,
#         "BOUNDARY": str
#     }, errors="ignore")

#     enc = df["AZI"].apply(encode_azimuth)
#     df["ENC_COS"], df["ENC_SIN"] = zip(*enc)

#     return df

def get_wsm_dataset_geodataframe(csv_path: str) -> gpd.GeoDataFrame:
    """
    Read the WSM-2016 CSV and return a clean GeoDataFrame of anchor nodes with:
        ['ID','LON','LAT','DEPTH','AZI','ENC_COS','ENC_SIN',
         ,'PLATE','DIST', 'QUALITY','BOUNDARY','geometry']
    """

    needed_col = ["ID", "LON", "LAT", "DEPTH", "AZI", "DIST","QUALITY", "PLATE", "BOUNDARY"]
    
    df = pd.read_csv(csv_path, encoding='ISO-8859-1', usecols=needed_col)

    # Drop invalid entries
    df = df.dropna(subset=["AZI", "LON", "LAT", "PLATE", "BOUNDARY", "QUALITY", "DIST", "DEPTH"])
    df = df[df["AZI"] <= 180.0]

    # Enforce data types
    df = df.astype({
        "ID": str,
        "LON": float,
        "LAT": float,
        "DEPTH": float,
        "AZI": float,
        "REGIME": str,
        "DIST": float,
        "PLATE": str,
        "BOUNDARY": str
    }, errors="ignore")

    enc = df["AZI"].apply(encode_azimuth)

    df["ENC_COS"], df["ENC_SIN"] = zip(*enc)

    df["GEOMETRY"] = df.apply(lambda row: Point(row["LON"], row["LAT"]), axis=1)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="GEOMETRY", crs="EPSG:4326")

    return gdf


def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000

def load_steps_table(path):
    """
    Parses pb2002_step.dat.txt into a DataFrame with:
    plate_pair, lon1, lat1, lon2, lat2, boundary_type
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14:
                continue
            try:
                plate_pair = parts[1].strip(':')
                lon1, lat1 = float(parts[2]), float(parts[3])
                lon2, lat2 = float(parts[4]), float(parts[5])
                boundary_type = parts[-1].strip(':')
                data.append({
                    "plate_pair": plate_pair,
                    "lon1": lon1,
                    "lat1": lat1,
                    "lon2": lon2,
                    "lat2": lat2,
                    "boundary_type": boundary_type
                })
            except ValueError:
                continue
    return pd.DataFrame(data)

def parse_pb2002_boundaries_with_type(dig_filepath):
    """
    Parses PB2002_boundaries.dig.txt and adds heuristic boundary_type codes ('SUB', 'CCB', 'OSR', etc.)
    """
    segments_list = []
    current_segment = None
    coords_buffer = []

    with open(dig_filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('*** end of line segment ***'):
                if current_segment and coords_buffer:
                    current_segment['coordinates'] = coords_buffer
                    segments_list.append(current_segment)
                current_segment = None
                coords_buffer = []
                continue

            match_title = re.match(r'^([A-Z]{2})([/\\-])([A-Z]{2})', line)
            if match_title:
                plate_left = match_title.group(1)
                boundary_sym = match_title.group(2)
                plate_right = match_title.group(3)

                subd = boundary_sym if boundary_sym in ['/', '\\'] else None

                if subd:
                    boundary_type = "SUB"
                elif plate_left in ["AF", "EU", "NA"] and plate_right in ["AF", "EU", "NA"]:
                    boundary_type = "CCB"
                elif plate_left in ["PA", "NZ", "AN", "IN", "SO", "AU"] and plate_right in ["PA", "NZ", "AN", "IN", "SO", "AU"]:
                    boundary_type = "OSR"
                else:
                    boundary_type = "UNKNOWN"

                current_segment = dict(
                    plate_left=plate_left,
                    plate_right=plate_right,
                    subduction=subd,
                    boundary_type=boundary_type,
                    coordinates=[]
                )
                coords_buffer = []
                continue

            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    try:
                        lon_val = float(parts[0])
                        lat_val = float(parts[1])
                        coords_buffer.append((lon_val, lat_val))
                    except ValueError:
                        continue

    return segments_list

def assign_boundary_types_from_steps(dig_segments, steps_df):
    """
    For each segment in dig_segments, find closest step (same plate pair) and assign its boundary_type.
    """
    for seg in dig_segments:
        coords = seg["coordinates"]
        if not coords:
            continue
        midpt = coords[len(coords) // 2]
        mid_latlon = (midpt[1], midpt[0])  # (lat, lon)

        pair1 = f"{seg['plate_left']}-{seg['plate_right']}"
        pair2 = f"{seg['plate_right']}-{seg['plate_left']}"

        matches = steps_df[
            (steps_df["plate_pair"] == pair1) | (steps_df["plate_pair"] == pair2)
        ]

        best_dist = float('inf')
        best_type = seg.get("boundary_type", "UNKNOWN")

        for _, row in matches.iterrows():
            step_mid = ((row.lat1 + row.lat2)/2, (row.lon1 + row.lon2)/2)
            try:
                dist = geodesic(mid_latlon, step_mid).kilometers
                if dist < best_dist:
                    best_dist = dist
                    best_type = row.boundary_type
            except Exception:
                continue

        seg["boundary_type"] = best_type
    return dig_segments

def find_distance_to_nearest_boundary(nodes_df, segments_list):
    """
    Adds 'DIST' (in km) and 'BOUNDARY_IDX' (index in segments_list)
    to each row of nodes_df based on nearest plate boundary.
    """
    nodes_df = nodes_df.copy()
    line_geoms = []
    for i, seg in enumerate(segments_list):
        coords = seg.get('coordinates', [])
        if len(coords) >= 2:
            try:
                line_geoms.append((i, LineString(coords)))
            except Exception:
                continue

    for idx, row in nodes_df.iterrows():
        try:
            node_lat = row['LAT']
            node_lon = row['LON']
            node_point = Point(node_lon, node_lat)
            node_coords = (node_lat, node_lon)

            min_dist_km = float('inf')
            nearest_idx = None

            for seg_idx, line in line_geoms:
                projected = line.interpolate(line.project(node_point))
                closest_coords = (projected.y, projected.x)

                if not (-90 <= closest_coords[0] <= 90 and -180 <= closest_coords[1] <= 180):
                    continue

                dist_km = geodesic(node_coords, closest_coords).kilometers
                if dist_km < min_dist_km:
                    min_dist_km = dist_km
                    nearest_idx = seg_idx

            nodes_df.loc[idx, 'DIST'] = min_dist_km
            # nodes_df.loc[idx, 'BOUNDARY_IDX'] = nearest_idx
        except Exception:
            nodes_df.loc[idx, 'DIST'] = None
            # nodes_df.loc[idx, 'BOUNDARY_IDX'] = None

    return nodes_df

# def assign_boundary_type_by_index(nodes_df, segments_list, idx_column='BOUNDARY_IDX'):
#     """
#     For each node, use the BOUNDARY_IDX to assign a BOUNDARY_TYPE label
#     from the corresponding segment in segments_list.
#     Adds column: 'BOUNDARY' (cleaned type, without '*')
#     """
#     nodes_df = nodes_df.copy()
#     boundary_types = []

#     for idx in nodes_df[idx_column]:
#         if pd.isna(idx):
#             boundary_types.append(None)
#         else:
#             idx = int(idx)
#             if 0 <= idx < len(segments_list):
#                 btype = segments_list[idx].get('boundary_type', None)
#                 if isinstance(btype, str):
#                     btype = btype.strip('*')  # Remove trailing '*' if present
#                 boundary_types.append(btype)
#             else:
#                 boundary_types.append(None)

#     nodes_df.loc[:, 'BOUNDARY'] = boundary_types
#     return nodes_df

def quality_assignment(anchor_df, grid_df,
                             lon="LON", lat="LAT", qual="QUALITY",
                             R_km=200.0,             # search radius
                             cutoff_frac=0.10,       # "close" = alpha * R
                             weight_mode="linear",   # 'linear' | 'inverse' | 'none'
                             hi_qual_set=("A", "B", "C"),
                             unknown_label="U"):
    """
    Stress2Grid-inspired QUALITY assignment with:
      1. Majority vote on A/B/C anchors within cut-off distance
      2. Otherwise distance-weighted vote over all anchors within R

    Returns a copy of grid_df with QUALITY column filled.
    """
    R_earth = 6371.0                                    # km
    anchors_rad = np.radians(anchor_df[[lat, lon]].values)
    tree = BallTree(anchors_rad, metric='haversine')
    grid_rad = np.radians(grid_df[[lat, lon]].values)

    idx_within_R = tree.query_radius(grid_rad, r=R_km / R_earth)

    grid_quality = []

    for g_idx, anchor_idxs in enumerate(idx_within_R):
        if len(anchor_idxs) == 0:
            grid_quality.append(unknown_label)
            continue

        # Distances to those anchors
        dists_km = tree.query(grid_rad[g_idx].reshape(1, -1),
                              k=len(anchor_idxs))[0][0] * R_earth
        quals = anchor_df.iloc[anchor_idxs][qual].values

        # ---------- Stage 1: majority of A/B/C within cut-off ----------
        close_mask = dists_km <= cutoff_frac * R_km
        close_hi = [q for q, d in zip(quals, dists_km)
                    if d <= cutoff_frac * R_km and q in hi_qual_set]

        if close_hi:
            # Count occurrences of A, B, C
            counts = {q: close_hi.count(q) for q in hi_qual_set}
            # Pick the letter with highest count (tie -> alphabetical)
            best_quality = max(counts, key=counts.get)
            grid_quality.append(best_quality)
            continue  # done with this grid node

        # ---------- Stage 2: distance-weighted vote over all anchors ----------
        if weight_mode == "none":
            weights = np.ones_like(dists_km)
        elif weight_mode == "linear":
            weights = 1 - dists_km / R_km
        elif weight_mode == "inverse":
            weights = R_km / (dists_km + 1e-6)
        else:
            raise ValueError("weight_mode must be 'linear', 'inverse', or 'none'.")

        # Apply Stress2Grid cut-off: anchors inside alpha*R get max weight
        weights[dists_km <= cutoff_frac * R_km] = weights.max()

        # Sum weights per QUALITY
        total_w = {}
        for q, w in zip(quals, weights):
            total_w[q] = total_w.get(q, 0.0) + w

        best_quality = max(total_w, key=total_w.get)
        grid_quality.append(best_quality)

    grid_out = grid_df.copy()
    grid_out[qual] = grid_quality

    # If grid_df is a GeoDataFrame, preserve geometry and CRS
    if isinstance(grid_df, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(grid_out, geometry=grid_df.geometry, crs=grid_df.crs)
    else:
        return grid_out

# ------------------------------------------------------------------

# def _build_grid(grid_spec):
#     """Return a DataFrame with columns id, longitude, latitude, depth_m."""
#     lon_vals = np.arange(grid_spec['min_lon'],
#                          grid_spec['max_lon']+1e-9,
#                          grid_spec['spacing_deg'])
#     lat_vals = np.arange(grid_spec['min_lat'],
#                          grid_spec['max_lat']+1e-9,
#                          grid_spec['spacing_deg'])
#     records, gid = [], 0
#     for z in grid_spec.get('depths', [0.0]):        # allow 3-D grids
#         for lat in lat_vals:
#             for lon in lon_vals:
#                 records.append((f'grid_{gid:06d}', lon, lat, z))
#                 gid += 1
#     return pd.DataFrame(records,
#                         columns=['ID', 'LON', 'LAT', 'DEPTH'])

def _build_grid(grid_spec):
    """
    Return a GeoDataFrame with columns: ID, LON, LAT, DEPTH, geometry (Point).
    
    grid_spec: dict with keys:
        - min_lon, max_lon
        - min_lat, max_lat
        - spacing_deg
        - depths (optional list of depth values)
    """
    lon_vals = np.arange(grid_spec['min_lon'],
                         grid_spec['max_lon'] + 1e-9,
                         grid_spec['spacing_deg'])
    lat_vals = np.arange(grid_spec['min_lat'],
                         grid_spec['max_lat'] + 1e-9,
                         grid_spec['spacing_deg'])

    records, geometries = [], []
    gid = 0
    for z in grid_spec.get('depths', [0.0]):  # allow 3-D grids
        for lat in lat_vals:
            for lon in lon_vals:
                records.append((f'grid_{gid:06d}', lon, lat, z))
                geometries.append(Point(lon, lat))
                gid += 1

    df = pd.DataFrame(records, columns=['ID', 'LON', 'LAT', 'DEPTH'])
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return gdf

# def _process_built_grid():

#     return gdf


def generate_shmax_data(anchor_df_path, grid_spec_file):
    """
    Builds X (Nx2) and A (NxN) matrices for kriging sHmax orientations.
    Saves: shmax_X.npy, shmax_A.npy, shmax_node_info.pkl
    """

    anchor_df = pd.read_csv(anchor_df_path, encoding='ISO-8859-1')
    assert {'ID','LON','LAT','DEPTH', 'QUALITY','AZI', 'DIST', 'BOUNDARY'}.issubset(anchor_df.columns)


    anchor_gdf = gpd.GeoDataFrame(
    anchor_df.copy(),
    geometry=gpd.points_from_xy(anchor_df.LON, anchor_df.LAT),
    crs="EPSG:4326"
)

    # build grid from spec as GeoDataFrame
    if os.path.isfile(grid_spec_file):
        with open(grid_spec_file) as f:
            grid_spec = yaml.safe_load(f)
        grid_gdf = _build_grid(grid_spec)  # this returns a GeoDataFrame with EPSG:4326
    else:
        grid_gdf = gpd.GeoDataFrame(columns=['ID','LON','LAT','DEPTH','geometry'], geometry=[], crs="EPSG:4326")

    # --- Load PB2002 tectonic models ---
    # dict_plates = build_plate_polygons_from_dig(PB2002_plates_path)
    # dict_orogen = build_orogen_polygons_from_dig(PB2002_orogen_path)
    steps_df = load_steps_table(PB2002_steps)
    segments = parse_pb2002_boundaries_with_type(PB2002_boundary_path)
    segments = assign_boundary_types_from_steps(segments, steps_df)

    # --- Fill missing values ---
    grid_gdf['DEPTH'] = 0.0
    grid_gdf['PLATE'] = 'UN'
    # grid_gdf['BOUNDARY'] = 'UNK'
    grid_gdf['AZI'] = np.nan
    grid_gdf['QUALITY'] = np.nan

    anchor_999_gdf = anchor_gdf[((anchor_gdf['AZI'] > 180)&(anchor_gdf['AZI'].notna()))] # this will be retrieved and saved at the call of this function
    anchor_gdf = anchor_gdf[~(anchor_gdf['AZI'] > 180)]

    # Path to your shapefile extracted from tectonicplates-master
    plates_path = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/tectonicplates-master/tectonicplates-master/PB2002_plates.shp"
    plates_gdf = gpd.read_file(plates_path)


    # Join grid points with plate polygons
    joined = gpd.sjoin(grid_gdf, plates_gdf, how="left", predicate="within")
    # The plate ID should now be in joined['PLATEID']
    grid_gdf["PLATE"] = joined["Code"].fillna("UN")

    # --- Quality assignment ---
    anchor_gdf = anchor_gdf[~(anchor_gdf['QUALITY'].isna())]
    grid_gdf = quality_assignment(
        anchor_gdf, grid_gdf,
        R_km=200, cutoff_frac=0.10,
        weight_mode="linear"
    )

    grid_gdf = grid_gdf[~(grid_gdf['QUALITY'] == 'U')]

    # --- Plate and boundary assignments ---
    # grid_gdf = assign_node_plate_and_orogen(grid_gdf, dict_plates, orogen_polygons=dict_orogen)
    grid_gdf = find_distance_to_nearest_boundary(grid_gdf, segments)
    # grid_gdf = assign_boundary_type_by_index(grid_gdf, segments)

    # --- Merge and filter region of interest---
    gdf = pd.concat([anchor_gdf, grid_gdf], ignore_index=True)
    gdf.reset_index(drop=True, inplace=True)

    lon_min, lon_max = grid_gdf['LON'].min(), grid_gdf['LON'].max()
    lat_min, lat_max = grid_gdf['LAT'].min(), grid_gdf['LAT'].max()
    gdf = gdf[(gdf['LON'] >= lon_min) & (gdf['LON'] <= lon_max) &
            (gdf['LAT'] >= lat_min) & (gdf['LAT'] <= lat_max)]

    # Feature matrix (X) ---
    N = len(gdf)
    X = np.zeros((N, 2), dtype=np.float32)
    anchor_mask = gdf['AZI'].notna()
    X[anchor_mask.values] = np.vstack(gdf.loc[anchor_mask, 'AZI'].apply(encode_azimuth))

    # Affinity Matrix (A) or adjacency matrix ---
    A = np.zeros((N, N), dtype=np.float32)
    anchor_coords = gdf.loc[anchor_mask, ['LON', 'LAT']].values
    hdists = [
        haversine(lon1, lat1, lon2, lat2)
        for i, (lon1, lat1) in enumerate(anchor_coords)
        for j, (lon2, lat2) in enumerate(anchor_coords[i+1:], start=i+1)
    ]
    sigma_h = np.median(hdists) if hdists else 100_000.0
    sigma_z = 500.0

    lons = gdf['LON'].values
    lats = gdf['LAT'].values
    depths = gdf['DEPTH'].fillna(0.0).values

    for i in range(N):
        for j in range(i+1, N):
            d_h = haversine(lons[i], lats[i], lons[j], lats[j])
            d_z = abs(depths[i] - depths[j])
            w = np.exp(-(d_h/sigma_h)**2 - (d_z/sigma_z)**2)
            A[i, j] = w

    A = A + A.T + np.eye(N, dtype=np.float32)

    np.save(os.path.join(STRESS_DIR, 'shmax_X.npy'), X)
    np.save(os.path.join(STRESS_DIR, 'shmax_A.npy'), A)
    with open(os.path.join(STRESS_DIR, 'shmax_node_info.pkl'), 'wb') as f:
        pickle.dump(gdf, f)
    with open(os.path.join(STRESS_DIR, 'wsm_test_data.pkl'), 'wb') as f:
        pickle.dump(anchor_999_gdf, f)

def generate_shmax_train_data(anchor_df_path: str, grid_gdf_path: Union[str, gpd.GeoDataFrame]):
    """
    Builds encoded orientation matrix X (Nx2) and affinity matrix A (NxN) 
    for kriging maximum horizontal stress orientation (SHmax).
    
    Parameters:
    - anchor_df_path : path to CSV file with anchor SHmax data
    - grid_gdf_path  : path to CSV file or GeoDataFrame with grid node info
    
    Saves:
    - shmax_X_test.npy
    - shmax_A_test.npy
    - shmax_node_info_test.pkl
    """

    # Load anchor dataset and convert to GeoDataFrame
    anchor_df = pd.read_csv(anchor_df_path, encoding='ISO-8859-1')
    assert {'ID','LON','LAT','DEPTH','QUALITY','AZI','DIST','REGIME','BOUNDARY'}.issubset(anchor_df.columns)

    anchor_gdf = gpd.GeoDataFrame(
        anchor_df.copy(),
        geometry=gpd.points_from_xy(anchor_df["LON"], anchor_df["LAT"]),
        crs="EPSG:4326"
    )
    anchor_999_gdf = anchor_gdf[((anchor_gdf['AZI'] > 180)&(anchor_gdf['AZI'].notna()))] # this will be retrieved and saved at the call of this function
    anchor_gdf = anchor_gdf[(anchor_gdf["AZI"] <= 180.0) & anchor_gdf["AZI"].notna()]
    anchor_gdf = anchor_gdf[anchor_gdf["QUALITY"].notna()]

    # Load or validate grid input
    if isinstance(grid_gdf_path, str):
        # Load the grid CSV file
        grid_df = pd.read_csv(grid_gdf_path)

        # Assign unique grid point IDs
        grid_df = grid_df.reset_index(drop=True)
        grid_df['ID'] = [f'grid_{i:06d}' for i in range(len(grid_df))]

        # Convert to GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(
            grid_df,
            geometry=gpd.points_from_xy(grid_df['LON'], grid_df['LAT']),
            crs="EPSG:4326"
            )
    else:
        grid_gdf = grid_gdf_path.copy()
        
    print("Before reassignment:", grid_gdf["BOUNDARY"].isna().sum(), "/", len(grid_gdf))


    grid_gdf["DEPTH"] = 0.0
    grid_gdf["PLATE"] = "UN"
    grid_gdf["AZI"] = np.nan
    grid_gdf["QUALITY"] = np.nan

    # Quality assignment
    grid_gdf = quality_assignment(
        anchor_gdf, grid_gdf,
        R_km=200, cutoff_frac=0.10, weight_mode="linear"
    )
    
    # Path to your shapefile extracted from tectonicplates-master
    plates_path = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/tectonicplates-master/tectonicplates-master/PB2002_plates.shp"
    plates_gdf = gpd.read_file(plates_path)


    # Join grid points with plate polygons
    joined = gpd.sjoin(grid_gdf, plates_gdf, how="left", predicate="within")
    # The plate ID should now be in joined['PLATEID']
    grid_gdf["PLATE"] = joined["Code"].fillna("UN")
    

    # steps_df = load_steps_table(PB2002_steps)
    segments = parse_pb2002_boundaries_with_type(PB2002_boundary_path)
    grid_gdf = find_distance_to_nearest_boundary(grid_gdf, segments)
    # Filter out unqualified/unlabeled
    grid_gdf = grid_gdf[grid_gdf["QUALITY"] != "U"]
    
    
    print("After reassignment:", grid_gdf["BOUNDARY"].isna().sum(), "/", len(grid_gdf))
    grid_gdf = grid_gdf[grid_gdf["BOUNDARY"] != "UUU"]


    # Combine anchor and grid data
    gdf = pd.concat([anchor_gdf, grid_gdf], ignore_index=True)
    gdf.reset_index(drop=True, inplace=True)


    # Limit to bounding box
    lon_min, lon_max = grid_gdf["LON"].min(), grid_gdf["LON"].max()
    lat_min, lat_max = grid_gdf["LAT"].min(), grid_gdf["LAT"].max()
    gdf = gdf[
        (gdf["LON"] >= lon_min) & (gdf["LON"] <= lon_max) &
        (gdf["LAT"] >= lat_min) & (gdf["LAT"] <= lat_max)
    ]

    # Orientation encoding
    N = len(gdf)
    X = np.zeros((N, 2), dtype=np.float32)
    anchor_mask = gdf["AZI"].notna()
    X[anchor_mask.values] = np.vstack(gdf.loc[anchor_mask, "AZI"].apply(encode_azimuth))

    # Affinity matrix
    A = np.zeros((N, N), dtype=np.float32)
    anchor_coords = gdf.loc[anchor_mask, ["LON", "LAT"]].values
    hdists = [
        haversine(lon1, lat1, lon2, lat2)
        for i, (lon1, lat1) in enumerate(anchor_coords)
        for lon2, lat2 in anchor_coords[i+1:]
    ]
    sigma_h = np.median(hdists) if hdists else 100_000.0
    sigma_z = 500.0

    lons = gdf["LON"].values
    lats = gdf["LAT"].values
    depths = gdf["DEPTH"].fillna(0.0).values

    for i in range(N):
        for j in range(i + 1, N):
            d_h = haversine(lons[i], lats[i], lons[j], lats[j])
            d_z = abs(depths[i] - depths[j])
            w = np.exp(-(d_h / sigma_h) ** 2 - (d_z / sigma_z) ** 2)
            A[i, j] = w

    A = A + A.T + np.eye(N, dtype=np.float32)
        
    np.save(os.path.join(STRESS_DIR, 'shmax_X_train.npy'), X)
    np.save(os.path.join(STRESS_DIR, 'shmax_A_train.npy'), A)
    with open(os.path.join(STRESS_DIR, 'shmax_node_info_train.pkl'), 'wb') as f:
        pickle.dump(gdf, f)

    with open(os.path.join(STRESS_DIR, 'wsm_train_data.pkl'), 'wb') as f:
        pickle.dump(anchor_999_gdf, f)



def generate_shmax_test_data(anchor_df_path, grid_spec_file):
    """
    Builds X (Nx2) and A (NxN) matrices for kriging sHmax orientations.
    Saves: shmax_X_test.npy, shmax_A_test.npy, shmax_node_info_test.pkl
    """

    anchor_df = pd.read_csv(anchor_df_path, encoding='ISO-8859-1')
    assert {'ID','LON','LAT','DEPTH', 'QUALITY','AZI', 'DIST'}.issubset(anchor_df.columns)


    anchor_gdf = gpd.GeoDataFrame(
    anchor_df.copy(),
    geometry=gpd.points_from_xy(anchor_df.LON, anchor_df.LAT),
    crs="EPSG:4326"
)

    # build grid from spec as GeoDataFrame
    if os.path.isfile(grid_spec_file):
        with open(grid_spec_file) as f:
            grid_spec = yaml.safe_load(f)
        grid_gdf = _build_grid(grid_spec)  # this returns a GeoDataFrame with EPSG:4326
    else:
        grid_gdf = gpd.GeoDataFrame(columns=['ID','LON','LAT','DEPTH','geometry'], geometry=[], crs="EPSG:4326")

    # --- Load PB2002 tectonic models ---
    # dict_plates = build_plate_polygons_from_dig(PB2002_plates_path)
    # dict_orogen = build_orogen_polygons_from_dig(PB2002_orogen_path)
    steps_df = load_steps_table(PB2002_steps)
    segments = parse_pb2002_boundaries_with_type(PB2002_boundary_path)
    segments = assign_boundary_types_from_steps(segments, steps_df)

    # --- Fill missing values ---
    grid_gdf['DEPTH'] = 0.0
    grid_gdf['PLATE'] = 'UN'
    # grid_gdf['BOUNDARY'] = 'UNK'
    grid_gdf['AZI'] = np.nan
    grid_gdf['QUALITY'] = np.nan

    # anchor_999_gdf = anchor_gdf[((anchor_gdf['AZI'] > 180)&(anchor_gdf['AZI'].notna()))] # this will be retrieved and saved at the call of this function
    anchor_gdf = anchor_gdf[~(anchor_gdf['AZI'] > 180)]

    # --- Quality assignment ---
    anchor_gdf = anchor_gdf[~(anchor_gdf['QUALITY'].isna())]
    grid_gdf = quality_assignment(
        anchor_gdf, grid_gdf,
        R_km=200, cutoff_frac=0.10,
        weight_mode="linear"
    )

    grid_gdf = grid_gdf[~(grid_gdf['QUALITY'] == 'U')]

    # --- Plate and boundary assignments ---
    # grid_gdf = assign_node_plate_and_orogen(grid_gdf, dict_plates, orogen_polygons=dict_orogen)
    grid_gdf = find_distance_to_nearest_boundary(grid_gdf, segments)
    # grid_gdf = assign_boundary_type_by_index(grid_gdf, segments)

    # --- Merge and filter region of interest---
    gdf = pd.concat([anchor_gdf, grid_gdf], ignore_index=True)
    gdf.reset_index(drop=True, inplace=True)

    lon_min, lon_max = grid_gdf['LON'].min(), grid_gdf['LON'].max()
    lat_min, lat_max = grid_gdf['LAT'].min(), grid_gdf['LAT'].max()
    gdf = gdf[(gdf['LON'] >= lon_min) & (gdf['LON'] <= lon_max) &
            (gdf['LAT'] >= lat_min) & (gdf['LAT'] <= lat_max)]

    # Feature matrix (X) ---
    N = len(gdf)
    X = np.zeros((N, 2), dtype=np.float32)
    anchor_mask = gdf['AZI'].notna()
    X[anchor_mask.values] = np.vstack(gdf.loc[anchor_mask, 'AZI'].apply(encode_azimuth))

    # Affinity Matrix (A) or adjacency matrix ---
    A = np.zeros((N, N), dtype=np.float32)
    anchor_coords = gdf.loc[anchor_mask, ['LON', 'LAT']].values
    hdists = [
        haversine(lon1, lat1, lon2, lat2)
        for i, (lon1, lat1) in enumerate(anchor_coords)
        for j, (lon2, lat2) in enumerate(anchor_coords[i+1:], start=i+1)
    ]
    sigma_h = np.median(hdists) if hdists else 100_000.0
    sigma_z = 500.0

    lons = gdf['LON'].values
    lats = gdf['LAT'].values
    depths = gdf['DEPTH'].fillna(0.0).values

    for i in range(N):
        for j in range(i+1, N):
            d_h = haversine(lons[i], lats[i], lons[j], lats[j])
            d_z = abs(depths[i] - depths[j])
            w = np.exp(-(d_h/sigma_h)**2 - (d_z/sigma_z)**2)
            A[i, j] = w

    A = A + A.T + np.eye(N, dtype=np.float32)

    # --- Save Outputs ---
    np.save(os.path.join(STRESS_DIR, 'shmax_X_test.npy'), X)
    np.save(os.path.join(STRESS_DIR, 'shmax_A_test.npy'), A)
    with open(os.path.join(STRESS_DIR, 'shmax_node_info_test.pkl'), 'wb') as f:
        pickle.dump(gdf, f)

# def generate_shmax_train_data(anchor_df_path, grid_gdf_path):
#     """
#     Builds X (Nx2) and A (NxN) matrices for kriging sHmax orientations.
#     Saves: shmax_X_test.npy, shmax_A_test.npy, shmax_node_info_test.pkl
#     """

#     anchor_df = pd.read_csv(anchor_df_path, encoding='ISO-8859-1')
#     assert {'ID','LON','LAT','DEPTH' ,'QUALITY','AZI', 'DIST', 'REGIME','BOUNDARY'}.issubset(anchor_df.columns)


#     anchor_gdf = gpd.GeoDataFrame(
#     anchor_df.copy(),
#     geometry=gpd.points_from_xy(anchor_df.LON, anchor_df.LAT),
#     crs="EPSG:4326"
# )

# # Load the grid CSV file
#     grid_df = pd.read_csv(grid_gdf_path)

# # Convert to GeoDataFrame (assuming LON, LAT are in degrees)
#     grid_gdf = gpd.GeoDataFrame(
#         grid_df,
#         geometry=[Point(xy) for xy in zip(grid_df['LON'], grid_df['LAT'])],
#         crs="EPSG:4326"
#     )

#     segments = parse_pb2002_boundaries_with_type(PB2002_boundary_path)
#     # segments = assign_boundary_types_from_steps(segments, steps_df)

#     # --- Fill missing values ---
#     grid_gdf['DEPTH'] = 0.0
#     grid_gdf['PLATE'] = 'UN'
#     grid_gdf['AZI'] = np.nan
#     grid_gdf['QUALITY'] = np.nan

#     anchor_999_gdf = anchor_gdf[((anchor_gdf['AZI'] > 180)&(anchor_gdf['AZI'].notna()))] # this will be retrieved and saved at the call of this function
#     anchor_gdf = anchor_gdf[~(anchor_gdf['AZI'] > 180)]

#     # --- Quality assignment ---
#     anchor_gdf = anchor_gdf[~(anchor_gdf['QUALITY'].isna())]
#     grid_gdf = quality_assignment(
#         anchor_gdf, grid_gdf,
#         R_km=200, cutoff_frac=0.10,
#         weight_mode="linear"
#     )

#     # --- Plate and boundary assignments ---
#     # grid_gdf = assign_boundary_type_by_index(grid_gdf, segments)
#     grid_gdf = grid_gdf[~(grid_gdf['QUALITY'] == 'U')]
# #     grid_gdf = grid_gdf[~(grid_gdf['BOUNDARY'] == 'UUU')]

#     # --- Merge and filter region of interest---
#     gdf = pd.concat([anchor_gdf, grid_gdf], ignore_index=True)
#     gdf.reset_index(drop=True, inplace=True)

#     lon_min, lon_max = grid_gdf['LON'].min(), grid_gdf['LON'].max()
#     lat_min, lat_max = grid_gdf['LAT'].min(), grid_gdf['LAT'].max()
#     gdf = gdf[(gdf['LON'] >= lon_min) & (gdf['LON'] <= lon_max) &
#             (gdf['LAT'] >= lat_min) & (gdf['LAT'] <= lat_max)]

#     # Feature matrix (X) ---
#     N = len(gdf)
#     X = np.zeros((N, 2), dtype=np.float32)
#     anchor_mask = gdf['AZI'].notna()
#     X[anchor_mask.values] = np.vstack(gdf.loc[anchor_mask, 'AZI'].apply(encode_azimuth))

#     # Affinity Matrix (A) or adjacency matrix ---
#     A = np.zeros((N, N), dtype=np.float32)
#     anchor_coords = gdf.loc[anchor_mask, ['LON', 'LAT']].values
#     hdists = [
#         haversine(lon1, lat1, lon2, lat2)
#         for i, (lon1, lat1) in enumerate(anchor_coords)
#         for j, (lon2, lat2) in enumerate(anchor_coords[i+1:], start=i+1)
#     ]
#     sigma_h = np.median(hdists) if hdists else 100_000.0
#     sigma_z = 500.0

#     lons = gdf['LON'].values
#     lats = gdf['LAT'].values
#     depths = gdf['DEPTH'].fillna(0.0).values

#     for i in range(N):
#         for j in range(i+1, N):
#             d_h = haversine(lons[i], lats[i], lons[j], lats[j])
#             d_z = abs(depths[i] - depths[j])
#             w = np.exp(-(d_h/sigma_h)**2 - (d_z/sigma_z)**2)
#             A[i, j] = w

#     A = A + A.T + np.eye(N, dtype=np.float32)

#     # --- Save Outputs ---
    # np.save(os.path.join(STRESS_DIR, 'shmax_X_train.npy'), X)
    # np.save(os.path.join(STRESS_DIR, 'shmax_A_train.npy'), A)
    # with open(os.path.join(STRESS_DIR, 'shmax_node_info_train.pkl'), 'wb') as f:
    #     pickle.dump(gdf, f)

    # with open(os.path.join(STRESS_DIR, 'wsm_train_data.pkl'), 'wb') as f:
    #     pickle.dump(anchor_999_gdf, f)
    

    

# ------------------------------------------------------------------
def load_shmax_data():
    """
    Returns  A, X, node_info (DataFrame).
    Automatically calls 'generate_shmax_data()' if artefacts are missing.
    """
    need = [os.path.join(STRESS_DIR, f0) for f0 in
            ('shmax_X.npy', 'shmax_A.npy', 'shmax_node_info.pkl', 'wsm_test_data.pkl')]
    if not all(os.path.isfile(p) for p in need):
            generate_shmax_data()


    X = np.load(os.path.join(STRESS_DIR, 'shmax_X.npy')).astype(np.float32)
    A = np.load(os.path.join(STRESS_DIR, 'shmax_A.npy')).astype(np.float32)
    with open(os.path.join(STRESS_DIR, 'shmax_node_info.pkl'), 'rb') as f1:
        node_info = pickle.load(f1)
    with open(os.path.join(STRESS_DIR, 'wsm_test_data.pkl'), 'rb') as f2:
        test_data_999_gdf = pickle.load(f2)    

    #grid
    grid_gdf = node_info[(node_info['AZI'].isna())]
    # anchor_gdf = anchor_gdf[~(anchor_gdf['AZI'] > 180)]

    #anchor 
    anchor_gdf = node_info[~(node_info['AZI'].isna())]
    # test_data_999_gdf = node_info[((node_info['AZI'] > 180)&(node_info['AZI'].notna()))]
    # node_info = node_info[~(node_info['AZI'] > 180)]

    ###======= running this again just to retrieve the bad azimuth values==========
    # test_data_999_gdf = generate_shmax_data()
    #==============

    return A, X, grid_gdf, anchor_gdf, test_data_999_gdf, node_info

def load_shmax_train_data():
    """
    Returns  A, X, node_info (DataFrame).
    Automatically calls 'generate_shmax_data()' if artefacts are missing.
    """
    need = [os.path.join(STRESS_DIR, f0) for f0 in
            ('shmax_X_train.npy', 'shmax_A_train.npy', 'shmax_node_info_train.pkl', 'wsm_train_data.pkl')]
    if not all(os.path.isfile(p) for p in need):
            generate_shmax_data()


    X = np.load(os.path.join(STRESS_DIR, 'shmax_X_train.npy')).astype(np.float32)
    A = np.load(os.path.join(STRESS_DIR, 'shmax_A_train.npy')).astype(np.float32)
    with open(os.path.join(STRESS_DIR, 'shmax_node_info_train.pkl'), 'rb') as f1:
        node_info = pickle.load(f1)
    with open(os.path.join(STRESS_DIR, 'wsm_train_data.pkl'), 'rb') as f2:
        train_data_999_gdf = pickle.load(f2)    

    #grid
    grid_gdf = node_info[(node_info['AZI'].isna())]
    # anchor_gdf = anchor_gdf[~(anchor_gdf['AZI'] > 180)]

    #anchor 
    anchor_gdf = node_info[~(node_info['AZI'].isna())]

    return A, X, grid_gdf, anchor_gdf, train_data_999_gdf, node_info



def test_error(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension

    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs

    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index

    o = np.zeros([test_data.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]]) #Separate the test data into several h period

    for i in range(0, test_data.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        T_inputs = inputs*missing_inputs
        T_inputs = T_inputs/E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis = 0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))

        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    o = o*E_maxvalue
    truth = test_inputs_s[0:test_data.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1]
    test_mask =  1 - missing_index_s[0:test_data.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    o_ = o[:,list(unknow_set)]

    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]
    
    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    return MAE, RMSE, R2, o


def test_error_optionB(STmodel,
                       unknow_set,
                       test_data,          # (h_total , N)
                       A_s,               # (N , N)
                       E_maxvalue=1.0,
                       Missing0=False,
                       h_target=2):
    """
    Evaluate an IGNNK_ModOptionB model on a static snapshot.

    Parameters
    ----------
    STmodel     : trained IGNNK_ModOptionB
    unknow_set  : indices of grid (unsampled) nodes
    test_data   : full data tensor with auxiliary features
                  shape = (h_total , N)
    A_s         : full adjacency matrix
    E_maxvalue  : scaling factor for targets (1 if cos/sin already scaled)
    Missing0    : kept for API compatibility (ignored here)
    h_target    : number of target channels (default 2)

    Returns
    -------
    MAE, RMSE, R2, o
        o : reconstructed target tensor  (h_target , N)
    """
    unknow_set = set(unknow_set)
    time_dim   = h_target                     # identical to STmodel.time_dimension

    # ---- 1. Build masks -------------------------------------------------
    N = test_data.shape[1]
    test_inputs = test_data.astype('float32')       # no 0-masking needed here

    missing_index = np.ones((time_dim, N), dtype=np.float32)
    missing_index[:, list(unknow_set)] = 0          # 0 for grid nodes

    # ---- 2. Forward pass  ----------------------------------------------
    T_inputs = test_inputs / E_maxvalue            # scale
    T_inputs = T_inputs[np.newaxis, ...]           # add batch dim
    T_inputs = torch.from_numpy(T_inputs)

    A_q = torch.from_numpy(calculate_random_walk_matrix(A_s).T.astype('float32'))
    A_h = torch.from_numpy(calculate_random_walk_matrix(A_s.T).T.astype('float32'))

    with torch.no_grad():
        imputation = STmodel(T_inputs, A_q, A_h).cpu().numpy()  # (1, h_target, N)
    o = imputation[0] * E_maxvalue                             # (h_target , N)

    # ---- 3. Replace anchor positions with ground truth -----------------
    truth = test_inputs[:h_target] * E_maxvalue
    o[missing_index == 1] = truth[missing_index == 1]

    # ---- 4. Metrics  ----------------------------------------------------
    test_mask = 1 - missing_index
    MAE  = np.sum(np.abs(o - truth)) / np.sum(test_mask)
    RMSE = np.sqrt(np.sum((o - truth) ** 2) / np.sum(test_mask))

    var  = np.sum((truth - truth.mean()) ** 2)
    R2   = np.nan if var < 1e-9 else 1 - np.sum((o - truth) ** 2) / var

    return MAE, RMSE, R2, o

# ---------------------------------------------------------------------
def test_error_optionB_trained(STmodel,
                       unknow_set,
                       full_data,            # shape (h_total , N)
                       A_full,
                       h_target=2,
                       Missing0=False,
                       E_maxvalue=1.0):
    """
    Evaluate an IGNNK-OptionB model.

    Parameters
    ----------
    STmodel     : trained IGNNK_ModOptionB model
    unknow_set  : indices of grid / unsampled nodes
    full_data   : ndarray (h_total , N)  –- targets+features for *all* nodes
    A_full      : ndarray (N , N) adjacency
    h_target    : #channels to reconstruct (default 2)
    Missing0    : keep for API compatibility (ignored for static task)
    E_maxvalue  : scaling factor (1.0 if cos/sin already in –1…1)

    Returns
    -------
    MAE, RMSE, R2, o   (same order as original test_error)
       where `o` is reconstructed tensor of shape (h_target , N)
    """

    unknow_set = set(unknow_set)
    # ---------- masks -------------------------------------------------
    N          = full_data.shape[1]
    missing_idx  = np.ones((h_target, N), dtype=np.float32)
    missing_idx[:, list(unknow_set)] = 0        # 0 = to be predicted

    inp = torch.from_numpy(full_data[np.newaxis, ...].astype('float32'))
    A_q = torch.from_numpy(calculate_random_walk_matrix(A_full).T.astype('float32'))
    A_h = torch.from_numpy(calculate_random_walk_matrix(A_full.T).T.astype('float32'))

    with torch.no_grad():
        out_pred = STmodel(inp, A_q, A_h)[0].cpu().numpy()   # (h_target , N)

    truth = full_data[:h_target]                 # (h_target , N)
    o     = out_pred.copy()
    o[missing_idx == 1] = truth[missing_idx == 1]   # keep anchors intact
    o   *= E_maxvalue
    truth *= E_maxvalue

    mask_eval = 1 - missing_idx                  # evaluate only unknown
    MAE  = np.sum(np.abs(o-truth)) / np.sum(mask_eval)
    RMSE = np.sqrt(np.sum((o-truth)**2) / np.sum(mask_eval))

    # avoid zero-variance crash
    var = np.sum((truth - truth.mean())**2)
    R2  = np.nan if var < 1e-9 else 1 - np.sum((o-truth)**2) / var

    return MAE, RMSE, R2, o


def angular_mae(pred_cos, pred_sin, true_cos, true_sin, mask=None):
    """
    Mean absolute mis-orientation (degrees) between predicted and true σHmax.

    Parameters
    ----------
    pred_cos, pred_sin : ndarray (T , N)
        Predicted cos 2\theta and sin 2\theta channels.
    true_cos, true_sin : ndarray (T , N)
        Ground-truth channels.
    mask : ndarray (T , N) of {\theta,1}, optional
        1 = include this entry in the metric, \theta = ignore.
        If None, all elements are used.

    Returns
    -------
    mae_deg : float
        Mean |\Delta\theta| in degrees over the selected entries.
    """

    theta_pred = decode_azimuth(pred_cos, pred_sin)
    theta_true = decode_azimuth(true_cos, true_sin)
    diff = np.abs(theta_pred - theta_true)
    diff = np.minimum(diff, 180.0 - diff)         
    if mask is not None:
        diff = diff * mask
        return diff.sum() / (mask.sum() + 1e-8)
    else:
        return diff.mean()


## Build plate dictionary from the .dig.txt file from Bird (2003)
def build_plate_polygons_from_dig(filepath):
    """
    Parses PB2002_plates.dig.txt where each polygon starts with a plate name,
    followed by coordinates, and ends with '*** end of line segment ***'.

    Returns:
        dict: { plate_name (str) : shapely.geometry.Polygon or MultiPolygon }
    """
    plate_geoms = defaultdict(list)
    current_plate = None
    coord_buffer = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Start of a new plate block (line with no commas)
            if ',' not in line and '*** end' not in line:
                current_plate = line
                coord_buffer = []
                continue

            # Coordinate line
            elif ',' in line:
                try:
                    lon_str, lat_str = line.split(',')
                    lon = float(lon_str)
                    lat = float(lat_str)
                    coord_buffer.append((lon, lat))
                except ValueError:
                    continue
                continue

            # End of segment — create polygon from current buffer
            elif line.startswith('*** end of line segment ***'):
                if current_plate and coord_buffer:
                    # Ensure closure
                    if coord_buffer[0] != coord_buffer[-1]:
                        coord_buffer.append(coord_buffer[0])
                    try:
                        poly = Polygon(coord_buffer)
                        # Fix if invalid
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if poly.is_valid and not poly.is_empty:
                            plate_geoms[current_plate].append(poly)
                        else:
                            print(f"Skipped invalid geometry for plate {current_plate}")
                    except Exception as e:
                        print(f"Polygon creation error for {current_plate}: {e}")
                coord_buffer = []
                current_plate = None
                continue

    # Merge multiple segments into one polygon per plate
    plate_polygons = {}
    for plate, geoms in plate_geoms.items():
        if len(geoms) == 1:
            plate_polygons[plate] = geoms[0]
        else:
            merged = unary_union(geoms)
            plate_polygons[plate] = merged

    return plate_polygons

def build_orogen_polygons_from_dig(filepath):
    """
    Parses PB2002_orogens.dig.txt and returns a dict of Shapely Polygon objects,
    keyed by orogen name.

    Args:
        filepath (str): path to PB2002_orogens.dig.txt

    Returns:
        dict: { orogen_name (str) : shapely.geometry.Polygon }
    """
    orogen_polygons = {}
    current_orogen = None
    coord_buffer = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if 'Peter Bird' in line:
                current_orogen = line.split('Peter Bird')[0].replace('by', '').strip()
                coord_buffer = []
                continue

            elif ',' in line:
                try:
                    lon_str, lat_str = line.split(',')
                    lon = float(lon_str)
                    lat = float(lat_str)
                    coord_buffer.append((lon, lat))
                except ValueError:
                    continue
                continue

            elif line.startswith('*** end of line segment ***'):
                if current_orogen and coord_buffer:
                    try:
                        # Close polygon if not closed
                        if coord_buffer[0] != coord_buffer[-1]:
                            coord_buffer.append(coord_buffer[0])

                        polygon = Polygon(coord_buffer)

                        # Fix invalid polygon if needed
                        if not polygon.is_valid or polygon.is_empty:
                            fixed = polygon.buffer(0)
                            if fixed.is_valid and not fixed.is_empty:
                                polygon = fixed
                                print(f"Fixed invalid polygon for: {current_orogen}")
                            else:
                                print(f"Could not fix polygon for: {current_orogen}")
                                polygon = None

                        if polygon:
                            orogen_polygons[current_orogen] = polygon

                    except Exception as e:
                        print(f"Error building polygon for {current_orogen}: {e}")

                current_orogen = None
                coord_buffer = []
                continue

    return orogen_polygons

    #Assigns each node on the grid to the plate polygon it belongs.
def assign_node_plate_and_orogen(nodes_df, plate_polygons, orogen_polygons=None):
    """
    For each node in 'nodes_df' (which is e.g. a Pandas DataFrame
    with columns 'LON' and 'LAT'), find to which plate polygon it belongs.

    If orogen_polygons are provided, check if it falls in an orogen.

    Mutates 'nodes_df' in-place adding columns: 'plate_id', 'in_orogen'.
    """
    plate_keys = list(plate_polygons.keys())

    for idx, row in nodes_df.iterrows():
        x = row['LON']
        y = row['LAT']
        pt= Point(x,y)

        # default
        assigned_plate = None
        assigned_orogen= False

        # check orogens first, if you want
        if orogen_polygons:
            for orog_name, orog_poly in orogen_polygons.items():
                if orog_poly and orog_poly.contains(pt):
                    assigned_orogen = True
                    break

        # find which plate polygon contains the node
        # (some orogens might overlap multiple plates, but we can keep it simple)
        for pcode in plate_keys:
            poly = plate_polygons[pcode]
            if poly is not None and poly.contains(pt):
                assigned_plate = pcode
                break

        # store results
        nodes_df.at[idx, 'PLATE']  = assigned_plate if assigned_plate else 'Unknown'
