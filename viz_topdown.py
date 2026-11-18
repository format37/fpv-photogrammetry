"""Level a COLMAP model to its ground plane and draw a top-down map: sparse points in their
colours + camera trajectories per pass (+ optional PnP-localized query cameras).

usage: viz_topdown.py MODEL_DIR OUT.png [--pnp pnp_query.csv] [--save-align align.txt]
"""
import argparse, csv, numpy as np, open3d as o3d, pycolmap, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--pnp"); ap.add_argument("--save-align")
ap.add_argument("--gap", type=float, default=10.0, help="split a pass at a time gap longer than this (s)")
a = ap.parse_args()
rec = pycolmap.Reconstruction(a.model)

P = np.array([p.xyz for p in rec.points3D.values()]); Col = np.array([p.color for p in rec.points3D.values()]) / 255
cams = sorted([(im.name, im.projection_center(), im.viewing_direction()) for im in rec.images.values() if im.has_pose],
              key=lambda x: x[0])
C = np.array([c[1] for c in cams])

# drop far points (horizon, clouds): keep within 3x the median distance to the camera centroid
d = np.linalg.norm(P - C.mean(0), axis=1); keep = d < 3 * np.median(d)
P, Col = P[keep], Col[keep]

# ground plane by RANSAC, normal pointing towards the cameras
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
(a_, b_, c_, d_), inl = pc.segment_plane(distance_threshold=0.02 * np.median(d), ransac_n=3, num_iterations=5000)
n = np.array([a_, b_, c_]); n /= np.linalg.norm(n)
if (C.mean(0) @ n + d_ / np.linalg.norm([a_, b_, c_])) < 0: n = -n
z = n; x = np.cross([0, 0, 1.0], z); x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
y = np.cross(z, x); R = np.stack([x, y, z])               # world -> leveled
origin = P[inl].mean(0)
L = (P - origin) @ R.T; LC = (C - origin) @ R.T
alt = LC[:, 2]
print(f"ground plane inliers {len(inl)}/{len(P)}; camera height above plane: "
      f"median {np.median(alt):.3f}, min {alt.min():.3f}, max {alt.max():.3f} (model units)")
if a.save_align:
    np.savetxt(a.save_align, np.vstack([R, origin]), header="rows 0-2: R world->leveled, row 3: origin")

fig, ax = plt.subplots(figsize=(14, 11), dpi=110)
o = np.argsort(L[:, 2])
ax.scatter(L[o, 0], L[o, 1], c=Col[o], s=0.6, linewidths=0)
# passes = runs of frames from the same video prefix with no time gap > --gap s (adaptive keyframes skip hovers) (names: <prefix>t<sec>.jpg)
import re
def key(name):
    m = re.match(r"(.*?)t(\d+\.\d+)\.jpg$", name); return (m.group(1), float(m.group(2))) if m else (name, 0.0)
passes, prev = {}, None
for i, (name, _, _) in enumerate(cams):
    pre, t = key(name)
    if prev is None or pre != prev[0] or t - prev[1] > a.gap:
        cur = f"{pre}{int(t // 60)}:{int(t % 60):02d}"; passes[cur] = []
    passes[cur].append(i); prev = (pre, t)
palette = ["#e4572e", "#2e86de", "#8e44ad", "#f39c12", "#16a085", "#c0392b", "#2c3e50", "#d35400"]
for (label, idx), col in zip(passes.items(), palette * 10):
    label = f"pass from {label}"
    ax.plot(LC[idx, 0], LC[idx, 1], "-o", ms=2.5, lw=1.2, color=col, label=f"{label} ({len(idx)} frames)")
    for i in idx[::6]:
        v = R @ cams[i][2]; ax.arrow(LC[i, 0], LC[i, 1], v[0] * 0.08 * np.median(d), v[1] * 0.08 * np.median(d),
                                     color=col, width=0.0, head_width=0.012 * np.median(d), alpha=0.8)
if a.pnp:
    rows = [r for r in csv.DictReader(open(a.pnp)) if r["mode"] == "query" and r["cx"]]
    runs, prev = [], None                                          # one polyline per run, like the passes
    for r in rows:
        pre, t = key(r["image"])
        if prev is None or pre != prev[0] or t - prev[1] > a.gap: runs.append([])
        runs[-1].append([float(r["cx"]), float(r["cy"]), float(r["cz"])]); prev = (pre, t)
    for j, run in enumerate(runs):
        LQ = (np.array(run) - origin) @ R.T
        ax.plot(LQ[:, 0], LQ[:, 1], "-^", ms=5, lw=1.2, color="#20bf6b",
                label=f"PnP-localized query frames ({len(rows)})" if j == 0 else None)
ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=10)
lim = np.percentile(np.abs(np.vstack([L[:, :2], LC[:, :2]])), 99.5)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_title(f"Sparse SfM - {len(cams)} posed frames, {len(rec.points3D)} points "
             f"(top-down, levelled to the RANSAC ground plane; arbitrary scale)")
fig.tight_layout(); fig.savefig(a.out); print("wrote", a.out)
