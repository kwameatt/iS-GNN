
import numpy as np
from sklearn.cluster import KMeans
import torch

def _encode_axial_mean(c, s):
    C, S = np.sum(c), np.sum(s)
    n = np.hypot(C, S) + 1e-12
    return C/n, S/n

def _km_xy(lon, lat):
    lat0 = np.radians(np.mean(lat))
    x = (np.radians(lon) * np.cos(lat0)) * 6371.0
    y =  np.radians(lat) * 6371.0
    return x, y

def build_anchor_prototypes(node_info, X_total, train_anchor_idx, K=3, w_space=0.5, seed=42):
    """Cluster **training anchors only** into K groups using space + axial encodings."""
    idxA = np.asarray(sorted(train_anchor_idx))
    cA = X_total[idxA, 0]
    sA = X_total[idxA, 1]
    lonA = node_info['LON'].to_numpy()[idxA]
    latA = node_info['LAT'].to_numpy()[idxA]
    xA, yA = _km_xy(lonA, latA)

    sx = max(np.std(xA), 1e-6); sy = max(np.std(yA), 1e-6)
    XA = np.column_stack([w_space*(xA/sx), w_space*(yA/sy), cA, sA])

    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(XA)

    centers_xy = np.zeros((K, 2), dtype=np.float32)
    proto_cs   = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        m = labels == k
        centers_xy[k] = [np.mean(xA[m]), np.mean(yA[m])]
        proto_cs[k]   = _encode_axial_mean(cA[m], sA[m])
    return centers_xy, proto_cs

def prototype_prior(node_info, proto_xy, proto_cs, sigma_km=80.0, topK=2):
    """Spatially weighted mix of cluster prototypes → (2, N) prior."""
    lon = node_info['LON'].to_numpy(); lat = node_info['LAT'].to_numpy()
    x, y = _km_xy(lon, lat)
    N = len(lon); K = proto_xy.shape[0]

    W = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        dx = x - proto_xy[k,0]; dy = y - proto_xy[k,1]
        dkm = np.hypot(dx, dy)
        W[:, k] = np.exp(-(dkm / float(sigma_km))**2)

    if topK is not None and topK < K:
        idx = np.argpartition(-W, kth=topK-1, axis=1)
        mask = np.zeros_like(W, dtype=bool)
        mask[np.arange(N)[:,None], idx[:, :topK]] = True
        W[~mask] = 0.0

    Z = W.sum(axis=1, keepdims=True) + 1e-12
    Wn = W / Z
    C = (Wn @ proto_cs[:,0]); S = (Wn @ proto_cs[:,1])
    norm = np.hypot(C, S) + 1e-12
    return np.vstack([C/norm, S/norm])  # (2, N)

######################
def anchor_support(lon, lat, anchor_idx, sigma_km=120.0):
    N = len(lon)
    supp = np.zeros(N, dtype=np.float32)
    for i in range(N):
        d = haversine_km(lon[i], lat[i], lon[anchor_idx], lat[anchor_idx])
        supp[i] = np.sum(np.exp(-(d / sigma_km)**2))
    supp /= (supp.max() + 1e-12)
    return supp  # 0..1

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

#######################
def angular_from_cs(c, s):  # returns degrees in [0, 180)
    return (0.5*np.degrees(np.arctan2(s, c)) + 180) % 180

def mc_dropout_uncertainty(STmodel, inp, A_q, A_h, T=20):
    STmodel.train()  # keep dropout on, but no grads
    preds = []
    with torch.no_grad():
        for _ in range(T):
            out = STmodel(inp, A_q, A_h)[0].cpu().numpy()  # (2,N)
            # normalize each sample before angle
            c, s = out[0], out[1]
            mag = np.sqrt(c*c + s*s) + 1e-8
            preds.append(angular_from_cs(c/mag, s/mag))
    STmodel.eval()
    angs = np.stack(preds, axis=0)        # (T, N)
    # circular (axial) std: compute via doubling trick
    ang_rad = np.radians(angs*2.0)
    C = np.mean(np.cos(ang_rad), axis=0)
    S = np.mean(np.sin(ang_rad), axis=0)
    R = np.sqrt(C*C + S*S) + 1e-8         # mean resultant length
    circ_std_deg = np.degrees(np.sqrt(-2*np.log(R)))/2.0  # axial back-conversion
    # convert to confidence 0..1 (lower std -> higher conf)
    conf = np.exp(-(circ_std_deg/20.0)**2)  # 20° scale; tune
    return conf, circ_std_deg               # per-node

############################
def multi_hop_anchor_influence(A, anchors, K=2):
    # A: (N,N) symmetric, row-normalize to P
    D = A.sum(axis=1, keepdims=True) + 1e-12
    P = A / D
    N = A.shape[0]
    v = np.zeros(N, dtype=np.float32); v[anchors] = 1.0
    infl = np.zeros(N, dtype=np.float32)
    Pk = P.copy()
    for _ in range(K):
        infl += Pk @ v
        Pk = Pk @ P
    infl /= infl.max() + 1e-12
    return infl
###########################