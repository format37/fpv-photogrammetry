"""Rasterise a georeferenced cloud (georef.py) into a height grid + orthophoto for the web volume demo.

DSM cell = median height of its points; holes are filled by inpainting and flagged in the mask.
The canyon axis is the least-cost path over the DSM (cost grows with height above the local floor) between
two rough end points, then smoothed; the demo's walls stay perpendicular to it.

usage: build_dsm.py GEO_DIR --roi E0 N0 E1 N1 --axis E N E N [--cell 2] [--ortho-cell 1] [--track TELEMETRY.csv]
writes GEO_DIR/web/{dsm.png, ortho.jpg, meta.json} and GEO_DIR/dsm.tif (if rasterio is installed)
       dsm.png: R,G = uint16 height (big-endian, units of meta.zstep above meta.zmin), B = 255 measured, 128 filled hole, 0 outside the model
"""
import argparse, json, os, heapq, numpy as np, cv2, pandas as pd
from scipy.ndimage import gaussian_filter1d

ap = argparse.ArgumentParser(); ap.add_argument("geo"); ap.add_argument("--roi", nargs=4, type=float, required=True)
ap.add_argument("--axis", nargs=4, type=float, required=True); ap.add_argument("--cell", type=float, default=2.0)
ap.add_argument("--ortho-cell", type=float, default=1.0); ap.add_argument("--track")
a = ap.parse_args(); out = f"{a.geo}/web"; os.makedirs(out, exist_ok=True)
d = np.load(f"{a.geo}/cloud_enu.npz"); P, C = d["xyz"], d["rgb"]; G = json.load(open(f"{a.geo}/georef.json"))
e0, n0, e1, n1 = a.roi; m = (P[:, 0] >= e0) & (P[:, 0] < e1) & (P[:, 1] >= n0) & (P[:, 1] < n1); P, C = P[m], C[m]

def grid(cell):                                  # row 0 = north edge
    W, H = int((e1 - e0) / cell), int((n1 - n0) / cell)
    ix = np.clip(((P[:, 0] - e0) / cell).astype(int), 0, W - 1); iy = np.clip(((n1 - P[:, 1]) / cell).astype(int), 0, H - 1)
    return W, H, iy * W + ix

W, H, key = grid(a.cell)
o = np.lexsort((P[:, 2], key)); ks, zs = key[o], P[o, 2]
st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]); cnt = np.diff(np.r_[st, len(ks)])
Z = np.full(W * H, np.nan, np.float32); Z[ks[st]] = zs[st + cnt // 2]; Z = Z.reshape(H, W)   # median per cell
valid = np.isfinite(Z)
# drop isolated spikes (stray MVS points): cells far from the 5x5 median of their neighbourhood
zr = float(np.nanmax(Z) - np.nanmin(Z)); zn = np.nan_to_num((Z - np.nanmin(Z)) / zr).astype(np.float32)
zf = cv2.inpaint(zn, (~valid).astype(np.uint8), 3, cv2.INPAINT_NS) * zr + np.nanmin(Z); med = cv2.medianBlur(zf.astype(np.float32), 5)
spike = valid & (np.abs(Z - med) > 6); Z[spike] = np.nan; valid &= ~spike
zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
z8 = np.nan_to_num((Z - zmin) / (zmax - zmin), nan=0).astype(np.float32)
fill = cv2.inpaint(z8, (~valid).astype(np.uint8), 5, cv2.INPAINT_NS) * (zmax - zmin) + zmin        # float inpaint
Z = np.where(valid, Z, fill).astype(np.float32); Z = cv2.medianBlur(Z, 3)
big = cv2.blur(cv2.medianBlur(Z, 5), (31, 31)); far_spike = np.abs(Z - big) > 45; Z[far_spike] = big[far_spike]    # leftover towers of noise
dist = cv2.distanceTransform((~valid).astype(np.uint8), cv2.DIST_L2, 3) * a.cell                               # metres to the nearest measured cell
near = cv2.morphologyEx((dist < 40).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)  # model = measured + holes up to ~40 m
sm = cv2.GaussianBlur(Z, (0, 0), 6); Z = np.where(valid, Z, sm).astype(np.float32)                              # filled cells: smooth, they carry no detail
print(f"DSM {W}x{H} @ {a.cell} m, measured {valid.mean() * 100:.1f}% of cells, spikes removed {spike.sum()}, height {zmin:.0f}..{zmax:.0f} m above home")

# orthophoto: mean colour per cell, holes inpainted
Wo, Ho, ko = grid(a.ortho_cell); n = np.bincount(ko, minlength=Wo * Ho)
img = np.stack([np.bincount(ko, C[:, c], Wo * Ho) for c in range(3)], 1) / np.maximum(n, 1)[:, None]
img = img.reshape(Ho, Wo, 3).astype(np.uint8); hole = (n == 0).reshape(Ho, Wo).astype(np.uint8)
img = cv2.inpaint(img, hole, 3, cv2.INPAINT_TELEA); print(f"ortho {Wo}x{Ho} @ {a.ortho_cell} m, filled {hole.mean() * 100:.1f}%")
cv2.imwrite(f"{out}/ortho.jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 82])

# canyon axis: Dijkstra, cost = 1 + (height above the lowest cell within ~60 m)^2 / 4
k = int(60 / a.cell) | 1; floor = cv2.erode(Z, np.ones((k, k), np.uint8)); cost = 1 + (Z - floor) ** 2 / 4
rc = lambda E, N: (int((n1 - N) / a.cell), int((E - e0) / a.cell))
s, t = rc(*a.axis[:2]), rc(*a.axis[2:]); dist = np.full((H, W), np.inf); prev = {}; dist[s] = 0; pq = [(0.0, s)]
while pq:
    dcur, u = heapq.heappop(pq)
    if u == t: break
    if dcur > dist[u]: continue
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        v = (u[0] + dy, u[1] + dx)
        if 0 <= v[0] < H and 0 <= v[1] < W:
            nd = dcur + cost[v] * (1.414 if dy and dx else 1)
            if nd < dist[v]: dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
path = [t]
while path[-1] != s: path.append(prev[path[-1]])
path = np.array(path[::-1], float); raw = np.stack([e0 + (path[:, 1] + .5) * a.cell, n1 - (path[:, 0] + .5) * a.cell], 1)
sm = gaussian_filter1d(raw, 120 / a.cell, axis=0, mode="nearest")                    # walls must not swing: smooth over ~120 m
seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sm, axis=0), axis=1))]; L = seg[-1]
st_ = np.arange(0, L, 10.0); axis = np.stack([np.interp(st_, seg, sm[:, 0]), np.interp(st_, seg, sm[:, 1])], 1)
print(f"axis {L:.0f} m, floor {Z[rc(*axis[0])]:.0f} -> {Z[rc(*axis[-1])]:.0f} m above home")

zstep = (zmax - zmin) / 65535; q = np.round((Z - zmin) / zstep).astype(np.uint16)
png = np.dstack([np.where(valid, 255, np.where(near, 128, 0)), q & 255, q >> 8]).astype(np.uint8)                      # BGR order for OpenCV -> R = high byte
cv2.imwrite(f"{out}/dsm.png", png, [cv2.IMWRITE_PNG_COMPRESSION, 9])
lat0, lon0, h0 = G["home"]; Re = 6378137.0
meta = dict(w=W, h=H, cell=a.cell, e0=e0, n1=n1, zmin=zmin, zstep=zstep, home=dict(lat=lat0, lon=lon0, alt=h0),
            axis=axis.round(1).tolist(), georef=dict(rms=round(G["rms"], 2), scale=round(G["scale"], 3), lag=round(G["lag"], 2), cams=G["n"]),
            measured=round(float(valid.sum() / max(near.sum(), 1)), 4))
cams = pd.read_csv(f"{a.geo}/cams_enu.csv"); cams = cams[cams.used]
meta["cams"] = cams[["e", "n", "u"]].round(1).values.tolist()
if a.track:
    g = pd.read_csv(a.track).iloc[::5]                                                 # 1 Hz
    E = np.radians(g.lon - lon0) * Re * np.cos(np.radians(lat0)); N = np.radians(g.lat - lat0) * Re
    meta["track"] = np.stack([E, N, g.alt_gps_msl - h0], 1).round(1).tolist()
json.dump(meta, open(f"{out}/meta.json", "w"), separators=(",", ":"))
np.save(f"{a.geo}/dsm.npy", Z); np.save(f"{a.geo}/dsm_valid.npy", valid)
try:
    import rasterio; from rasterio.transform import from_origin
    zone = int((lon0 + 180) // 6) + 1; from pyproj import Transformer
    x0, y0 = Transformer.from_crs(4326, 32600 + zone, always_xy=True).transform(lon0, lat0)   # ENU ~ UTM near home (grid convergence ignored)
    with rasterio.open(f"{a.geo}/dsm.tif", "w", driver="GTiff", width=W, height=H, count=1, dtype="float32", crs=f"EPSG:{32600 + zone}",
                       transform=from_origin(x0 + e0, y0 + n1, a.cell, a.cell), compress="deflate") as f: f.write(Z + h0, 1)
    print("wrote dsm.tif (heights MSL)")
except Exception as e: print("no GeoTIFF:", type(e).__name__)
for f in ("dsm.png", "ortho.jpg", "meta.json"): print(f, os.path.getsize(f"{out}/{f}") // 1024, "KB")
