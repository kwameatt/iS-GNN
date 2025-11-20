import pandas as pd
import numpy as np 
import scipy.sparse as sp  
import torch
from math import radians, cos, sin, asin, sqrt
import os, yaml
import re 
import pickle, random
# from collections import defaultdict
# from shapely.geometry import Polygon
# from shapely.ops import unary_union
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import geopandas as gpd
from typing import Union 

os.environ["PROJ_LIB"] = "/home/kwameatt/anaconda3/envs/cs231n/share/proj"

STRESS_DIR = '/mnt/e/GNN_paper/2025_IGNNK/data/num_array_files'


# These files were obtained from Bird (2002)
PB2002_boundary_path = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/PB2002_boundaries.dig.txt"
PB2002_plates_path = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/PB2002_plates.dig.txt"
PB2002_orogen_path = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/PB2002_orogens.dig.txt"
PB2002_steps = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/PB2002_step.dat.txt"
### these are also obtained from the mordenized version of PB2002 model

###333333===========

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # raises error for non-deterministic ops


def current_mask_count(epoch, ramp_epochs, n_nodes,
                       p0=0.20, p1=0.50):
    """
    Returns how many nodes to mask this epoch.
     -epoch         : 0-based training epoch
     - ramp_epochs   : epochs over which to increase masking
     - n_nodes       : n_o_n_m  (number of nodes in sub-graph)
     - p0, p1        : start and end mask fractions
    """
    frac = p0 if epoch <= 0 else (
            p1 if epoch >= ramp_epochs
            else p0 + (p1 - p0) * (epoch / ramp_epochs)
        )
    return max(1, int(round(frac * n_nodes)))


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


def get_wsm_dataset_geodataframe(csv_path: str) -> gpd.GeoDataFrame:
    """
    Read the WSM-2016 CSV and return a clean GeoDataFrame of anchor nodes with:
        ['ID','LON','LAT','DEPTH','AZI','ENC_COS','ENC_SIN',
         ,'PLATE','DIST', 'QUALITY','BOUNDARY','geometry']
    """


    needed_col = ["ID", "LON", "LAT", "DEPTH", "AZI", "DIST","QUALITY", "PLATE"]
    
    df = pd.read_csv(csv_path, encoding='ISO-8859-1', usecols=needed_col)

    # Drop invalid entries
    df = df.dropna(subset=["AZI", "LON", "LAT", "PLATE", "QUALITY", "DIST",])
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


##############################3
def _axial_delta_deg_from_enc(c1, s1, c2, s2):
    """
    Smallest axial angle difference Δθ in degrees using encoded (cos2θ, sin2θ).
    dot = cos(2(θ1-θ2)); Δθ = 0.5 * arccos(dot).
    """
    dot = np.clip(c1 * c2 + s1 * s2, -1.0, 1.0)
    d_rad = 0.5 * np.arccos(dot)
    return np.degrees(d_rad)

def build_adjacency_bilateral_knn_rbf(
    lon_deg, lat_deg,
    is_anchor,           # bool array shape (N,)
    enc_cos, enc_sin,    # arrays shape (N,), encoded targets; grids can be 0
    kNN=12,
    sigma_km=300.0,          # spatial length scale (km)
    sigma_theta_deg=20.0,    # orientation length scale (deg) — smaller => sharper boundaries
    add_self_loops=True,
    scale_to_one=True,
    mode="anchor_only",      # "anchor_only" or "hybrid"
    k_anchor_avg=8           # used only if mode="hybrid"
):
    """
    Bilateral adjacency: w_ij = exp(-(d_ij/sigma_km)^2) * exp(-(Delta theta_ij/sigma_theta)^2),
    where Delta theta_ij is computed from axial encodings. By default, Delta θ is applied only on
    anchor–anchor edges (mode="anchor_only"); grids use spatial weight only.
    """
    lon_deg = np.asarray(lon_deg); lat_deg = np.asarray(lat_deg)
    is_anchor = np.asarray(is_anchor, dtype=bool)
    enc_cos = np.asarray(enc_cos); enc_sin = np.asarray(enc_sin)

    N = len(lon_deg)
    if N <= 1:
        return np.eye(N, dtype=np.float32)

    # kNN on sphere
    coords_rad = np.column_stack([np.radians(lat_deg), np.radians(lon_deg)])
    tree = BallTree(coords_rad, metric="haversine")
    k_eff = int(min(max(kNN, 1), max(1, N - 1)))
    dist_rad, idx = tree.query(coords_rad, k=k_eff + 1)
    nbrs = idx[:, 1:]                          # drop self
    dist_km = (dist_rad[:, 1:] * 6371.0).astype(np.float32)

    # Spatial weight
    W_spatial = np.exp(- (dist_km / float(sigma_km))**2).astype(np.float32)  # (N, k_eff)

    # Optional: get an orientation for grid nodes (hybrid mode)
    if mode == "hybrid":
        # approximate grid orientation by weighted avg of nearby anchors
        # (so edges crossing sharp orientation changes get down-weighted even for grids)
        grid_est_c = enc_cos.copy()
        grid_est_s = enc_sin.copy()
        for i in np.where(~is_anchor)[0]:
            nbr_i = nbrs[i]
            w_i   = W_spatial[i]
            maskA = is_anchor[nbr_i]
            if not maskA.any():
                continue
            wA = w_i[maskA]
            cs = enc_cos[nbr_i][maskA]
            ss = enc_sin[nbr_i][maskA]
            if wA.sum() > 0:
                c_bar = (wA * cs).sum() / wA.sum()
                s_bar = (wA * ss).sum() / wA.sum()
                norm = np.hypot(c_bar, s_bar) + 1e-8
                grid_est_c[i] = c_bar / norm
                grid_est_s[i] = s_bar / norm
        C = grid_est_c; S = grid_est_s
    else:
        C = enc_cos; S = enc_sin

    # Orientation weight (anchor_only => only when both endpoints are anchors)
    W_orient = np.ones_like(W_spatial, dtype=np.float32)
    for i in range(N):
        j_idx = nbrs[i]
        if mode == "anchor_only":
            mask = is_anchor[i] & is_anchor[j_idx]
        else:
            mask = np.ones_like(j_idx, dtype=bool)
        if not np.any(mask):
            continue
        Del_theta = _axial_delta_deg_from_enc(C[i], S[i], C[j_idx[mask]], S[j_idx[mask]])
        W_orient[i, mask] = np.exp(- (Del_theta / float(sigma_theta_deg))**2).astype(np.float32)

    # Combine
    W = W_spatial * W_orient

    # Assemble dense symmetric A
    A = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), k_eff)
    cols = nbrs.ravel()
    A[rows, cols] = W.ravel()
    A = np.maximum(A, A.T)

    if add_self_loops:
        np.fill_diagonal(A, 1.0)
    if scale_to_one and A.max() > 0:
        A /= A.max()

    return A

def auto_sigma_km_from_knn(lon_deg, lat_deg, kNN=12, percentile=70, w0=0.5):
    # pick r0 as a percentile of kNN distances, then set sigma so exp(-(r0/sigma)^2)=w0
    coords = np.column_stack([np.radians(lat_deg), np.radians(lon_deg)])
    tree = BallTree(coords, metric="haversine")
    dist_rad, _ = tree.query(coords, k=kNN+1)
    nn_dists_km = (dist_rad[:, 1:] * 6371.0).ravel()
    r0 = np.percentile(nn_dists_km, percentile)
    sigma_km = r0 / np.sqrt(-np.log(w0))
    return sigma_km

def generate_shmax_train_data(anchor_df_path: str,
                              grid_gdf_path: Union[str, gpd.GeoDataFrame],
                              kNN: int = 12,
                            #   sigma_km: float = 100.0,
                              qual_R_km = 200,
                              Q_cut_frac = 0.10,
                              Q_mode='linear',
                              theta_bound_deg=20,
                              th_boundary_mode="hybrid",
                              boundary_bound_km = 75.0,
                              kernel_w_0 = 0.7, #to aid sigma_km auto selection
                              kernel_percentile_r_0 = 0.6 #to aid sigma_km auto selection
                              ):
    """
    Builds encoded orientation matrix X (N,2) and affinity matrix A (N,N) for SHmax interpolation.

    Parameters
    ----------
    anchor_df_path : str
        Path to CSV with anchor SHmax data (must include ID,LON,LAT,DEPTH,QUALITY,AZI,DIST,REGIME).
    grid_gdf_path : str | GeoDataFrame
        Path to CSV for grid nodes or a GeoDataFrame with LON,LAT (other cols optional).
    kNN : int, default=12
        Nearest neighbors per node for adjacency (clipped to [1, N-1]).
    sigma_km : float, default=400.0
        RBF length scale (km) used to weight kNN edges.
    add_self_loops : bool, default=True
        If True, fills diagonal of A with 1.0 (kept consistent with downstream random-walk).
    scale_to_one : bool, default=True
        If True, rescales A to max=1.0 for numerical stability.

    Saves
    -----
    shmax_X_train.npy
    shmax_A_train.npy
    shmax_node_info_train.pkl

    Returns
    -------
    X : np.ndarray, shape (N, 2)
    A : np.ndarray, shape (N, N)
    gdf : GeoDataFrame (anchors + grids, row order matches X and A)
    """
    # Load and clean anchors
    anchor_df = pd.read_csv(anchor_df_path, encoding='ISO-8859-1')
    assert {'ID','LON','LAT','DEPTH','QUALITY','AZI','DIST','REGIME'}.issubset(anchor_df.columns)

    anchor_gdf = gpd.GeoDataFrame(
        anchor_df.copy(),
        geometry=gpd.points_from_xy(anchor_df["LON"], anchor_df["LAT"]),
        crs="EPSG:4326"
    )
    # anchor_999_gdf = anchor_gdf[((anchor_gdf['AZI'] > 180) & (anchor_gdf['AZI'].notna()))]  # retained if you use it later
    anchor_gdf = anchor_gdf[(anchor_gdf["AZI"] <= 180.0) & anchor_gdf["AZI"].notna()]
    anchor_gdf = anchor_gdf[anchor_gdf["QUALITY"].notna()]
    anchor_gdf = anchor_gdf[~(anchor_gdf["QUALITY"].isin(['E', 'D']))]
    anchor_gdf = anchor_gdf[~(anchor_gdf["REGIME"] == 'U')]

    # Load / validate grid
    if isinstance(grid_gdf_path, str):
        grid_df = pd.read_csv(grid_gdf_path).reset_index(drop=True)
        grid_df['ID'] = [f'grid_{i:06d}' for i in range(len(grid_df))]
        grid_gdf = gpd.GeoDataFrame(
            grid_df,
            geometry=gpd.points_from_xy(grid_df['LON'], grid_df['LAT']),
            crs="EPSG:4326"
        )
    else:
        grid_gdf = grid_gdf_path.copy()

    grid_gdf["DEPTH"] = 0.0
    grid_gdf["PLATE"] = "UN"
    grid_gdf["AZI"] = np.nan
    grid_gdf["QUALITY"] = np.nan

    # Quality assignment (unchanged)
    grid_gdf = quality_assignment(
        anchor_gdf, grid_gdf,
        R_km=qual_R_km, cutoff_frac=Q_cut_frac, weight_mode=Q_mode
    )

    # Plate join (unchanged)
    plates_path = "/mnt/e/GNN_paper/IGNNK/data/wsm2016/PB2002/tectonicplates-master/tectonicplates-master/PB2002_plates.shp"
    plates_gdf = gpd.read_file(plates_path)
    joined = gpd.sjoin(grid_gdf, plates_gdf, how="left", predicate="within")
    grid_gdf["PLATE"] = joined["Code"].fillna("UN")

    # Boundary distance 
    segments = parse_pb2002_boundaries_with_type(PB2002_boundary_path)
    grid_gdf = find_distance_to_nearest_boundary(grid_gdf, segments)

    # Filters 
    grid_gdf   = grid_gdf[~(grid_gdf['QUALITY'].isin(['E','D','Xne','Xmi','Xru']))]
    grid_gdf   = grid_gdf[grid_gdf["QUALITY"] != "U"]
    anchor_gdf = anchor_gdf[~(anchor_gdf['QUALITY'].isin(['Xne','Xmi','Xru']))]

  
    gdf = pd.concat([anchor_gdf, grid_gdf], ignore_index=True).reset_index(drop=True)

    # Bounding box 
    ###12/10/2025### update to specify an area of interest instead of using the 
    ################ grid corners for the study area
    lon_min, lon_max = anchor_gdf["LON"].min(), anchor_gdf["LON"].max()
    lat_min, lat_max = anchor_gdf["LAT"].min(), anchor_gdf["LAT"].max()
    gdf = gdf[gdf["LON"].between(lon_min, lon_max) & gdf["LAT"].between(lat_min, lat_max)]

    # -------- Encode orientations (anchors only) --------
    N = len(gdf)
    X = np.zeros((N, 2), dtype=np.float32)
    anchor_mask = gdf["AZI"].notna().to_numpy()
    if anchor_mask.any():
        X[anchor_mask] = np.vstack(gdf.loc[anchor_mask, "AZI"].apply(encode_azimuth))

    # -------- Affinity via kNN + RBF (NEW: parameterized) --------
    sigma_km = auto_sigma_km_from_knn(
        gdf["LON"].values, 
        gdf["LAT"].values, 
        kNN=kNN, 
        percentile=kernel_percentile_r_0, 
        w0=kernel_w_0)
    
    A = build_adjacency_bilateral_knn_rbf(
        lon_deg=gdf["LON"].values,
        lat_deg=gdf["LAT"].values,
        is_anchor=anchor_mask,
        enc_cos=X[:, 0],
        enc_sin=X[:, 1],
        kNN=kNN,                 # keep your kNN
        sigma_km=sigma_km,       # spatial scale you were using
        sigma_theta_deg=theta_bound_deg,    # <-- new knob for boundary sharpness
        mode=th_boundary_mode       # try "hybrid" if grids dominate
)

    # -------- Save --------
    np.save(os.path.join(STRESS_DIR, 'shmax_X_train.npy'), X)
    np.save(os.path.join(STRESS_DIR, 'shmax_A_train.npy'), A)
    with open(os.path.join(STRESS_DIR, 'shmax_node_info_train.pkl'), 'wb') as f:
        pickle.dump(gdf, f)

    return X, A, gdf



def load_shmax_train_data():
    """
    Returns  A, X, node_info (DataFrame).
    Automatically calls 'generate_shmax_data()' if artefacts are missing.
    """
    need = [os.path.join(STRESS_DIR, f0) for f0 in
            ('shmax_X_train.npy', 'shmax_A_train.npy', 'shmax_node_info_train.pkl', 'wsm_train_data.pkl')]
    if not all(os.path.isfile(p) for p in need):
            generate_shmax_train_data()


    X = np.load(os.path.join(STRESS_DIR, 'shmax_X_train.npy')).astype(np.float32)
    A = np.load(os.path.join(STRESS_DIR, 'shmax_A_train.npy')).astype(np.float32)
    with open(os.path.join(STRESS_DIR, 'shmax_node_info_train.pkl'), 'rb') as f1:
        node_info = pickle.load(f1)
    # with open(os.path.join(STRESS_DIR, 'wsm_train_data.pkl'), 'rb') as f2:
    #     train_data_999_gdf = pickle.load(f2)    

    #grid
    grid_gdf = node_info[(node_info['AZI'].isna())]
    #anchor 
    anchor_gdf = node_info[~(node_info['AZI'].isna())]

    return A, X, grid_gdf, anchor_gdf, node_info


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


def angular_mae(pred_cos, pred_sin, true_cos, true_sin, mask=None):
    """
    Mean absolute mis-orientation (degrees) between predicted and true sHmax.

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




