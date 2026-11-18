"""Georeference a COLMAP workspace with flight-controller GPS: fit a 7-DoF similarity (Umeyama) from the SfM
camera centres to the GPS track, then write the dense cloud in local ENU metres (origin = home).

Frame time = clip offset + t in the frame name (<clip>_t<sec>.jpg) + LAG (video -> telemetry, seconds).
LAG is searched on a grid; the best one minimises the fit residual. Clips given in --drop (e.g. a mirrored pass)
are left out of the fit and their points are removed from the cloud through fused.ply.vis.

usage: georef.py WS TELEMETRY.csv OUTDIR --clip c39=0 c40=216.04 ... [--drop c43] [--home LAT LON ALT]
       TELEMETRY.csv columns: t_since_arm_s, lat, lon, alt_gps_msl
writes OUTDIR/georef.json (s, R, t, lag, home, residuals), OUTDIR/cloud_enu.npz (xyz float32, rgb uint8), cams_enu.csv
"""
import argparse, json, os, re, struct, numpy as np, pandas as pd, pycolmap, open3d as o3d

ap = argparse.ArgumentParser(); ap.add_argument("ws"); ap.add_argument("tel"); ap.add_argument("out")
ap.add_argument("--clip", nargs="+", required=True); ap.add_argument("--drop", nargs="*", default=[])
ap.add_argument("--home", nargs=3, type=float); ap.add_argument("--lag", nargs=3, type=float, default=[-2, 4, 0.05])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
off = {k: float(v) for k, v in (c.split("=") for c in a.clip)}
rec = pycolmap.Reconstruction(os.path.join(a.ws, open(f"{a.ws}/best_model.txt").read().strip()))
ims = sorted(rec.images.values(), key=lambda i: i.name)
clip = np.array([re.match(r"(.*?)_?t\d", i.name).group(1) for i in ims])
tv = np.array([off[c] + float(re.search(r"t(\d+\.\d+)\.jpg", i.name).group(1)) for c, i in zip(clip, ims)])
C = np.array([i.projection_center() for i in ims]); use = ~np.isin(clip, a.drop)

g = pd.read_csv(a.tel); lat0, lon0, h0 = a.home or (g.lat[0], g.lon[0], g.alt_gps_msl[0]); Re = 6378137.0
G = np.stack([np.radians(g.lon - lon0) * Re * np.cos(np.radians(lat0)), np.radians(g.lat - lat0) * Re, g.alt_gps_msl - h0], 1)
gps_at = lambda t: np.stack([np.interp(t, g.t_since_arm_s, G[:, k]) for k in range(3)], 1)

def umeyama(X, Y):                                           # Y ~ s R X + t
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / Xc.var(0).sum(); return s, R, my - s * R @ mx

best = None
for lag in np.arange(a.lag[0], a.lag[1] + 1e-9, a.lag[2]):
    Y = gps_at(tv[use] + lag); s, R, t = umeyama(C[use], Y); r = np.linalg.norm(s * C[use] @ R.T + t - Y, axis=1)
    if best is None or (r**2).mean() < best[0]: best = ((r**2).mean(), lag, s, R, t, r)
_, lag, s, R, t, r = best
print(f"fit on {use.sum()} cameras: lag {lag:+.2f} s, scale {s:.3f} m/unit, rms {np.sqrt((r**2).mean()):.2f} m, median {np.median(r):.2f}, max {r.max():.2f}")
per = {}
for c in sorted(set(clip[use])):
    m = clip[use] == c; d = (s * C[use][m] @ R.T + t - gps_at(tv[use][m] + lag)); per[c] = [round(float(np.sqrt((d**2).sum(1).mean())), 2)] + d.mean(0).round(2).tolist()
    print(f"  {c}: rms {per[c][0]} m, mean offset ENU {per[c][1:]}")
json.dump(dict(scale=s, R=R.tolist(), t=t.tolist(), lag=lag, home=[lat0, lon0, h0], rms=float(np.sqrt((r**2).mean())), per_clip=per,
               n=int(use.sum()), dropped=a.drop), open(f"{a.out}/georef.json", "w"), indent=1)
pd.DataFrame(dict(name=[i.name for i in ims], t_arm=tv + lag, used=use, **{k: (s * C @ R.T + t)[:, j] for j, k in enumerate("enu")})).to_csv(f"{a.out}/cams_enu.csv", index=False)

pc = o3d.io.read_point_cloud(f"{a.ws}/dense/fused.ply"); P = np.asarray(pc.points); col = (np.asarray(pc.colors) * 255).astype(np.uint8)
keep = np.ones(len(P), bool)
if a.drop:                                                   # fused.ply.vis: per point, indices into the dense model's image list
    drec = pycolmap.Reconstruction(f"{a.ws}/dense/sparse"); order = [drec.images[k].name for k in sorted(drec.images)]
    bad = np.array([re.match(r"(.*?)_?t\d", n).group(1) in a.drop for n in order])
    buf = np.fromfile(f"{a.ws}/dense/fused.ply.vis", np.uint32, offset=8); i = 0
    for p in range(len(P)):
        k = buf[i]; keep[p] = not bad[buf[i + 1:i + 1 + k]].any(); i += 1 + k
    print(f"dropped {(~keep).sum():,} points seen by {a.drop}")
E = (s * P[keep] @ R.T + t).astype(np.float32)
np.savez(f"{a.out}/cloud_enu.npz", xyz=E, rgb=col[keep]); print(f"{len(E):,} points -> {a.out}/cloud_enu.npz; extent ENU {np.ptp(E, 0).round(0)}")
