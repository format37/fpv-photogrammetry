"""Crop the fused dense cloud to an object seen in chosen frames (union of their view frustums), for objects
the drone looked at from a distance instead of flying over (clean_and_mesh.py keeps only the area under the track).

Each --view gives an undistorted frame (dense/images) and a pixel box around the object. A point is kept
if it projects into the box of some view, in front of it, at a depth within the range of the sparse points
seen inside that box (+-10%), which cuts the far background behind the object and the near ground under the drone.
--visible TOL also drops points hidden from that view: deeper than (1+TOL) x its PatchMatch depth map
(min over a 9x9 window). Without it, terrain behind the object (seen by other passes) falls into the frustum.

usage: crop_focus.py WS OUTDIR --view c19_t159.750.jpg 560 300 1920 1080 [--view NAME x0 y0 x1 y1 ...]
                     [--visible 0.05] [--density 0.02 20] [--fused dense/fused.ply]
then:  clean_and_mesh.py OUTDIR/fused_focus.ply MODEL OUTDIR --no-crop
"""
import argparse, os, numpy as np, open3d as o3d, pycolmap, cv2

def read_depth(path):                                   # COLMAP dense .bin: "w&h&c&" + float32, column-major
    with open(path, "rb") as f:
        hdr = b""
        while hdr.count(b"&") < 3: hdr += f.read(1)
        w, h, c = map(int, hdr.decode().split("&")[:3]); d = np.fromfile(f, np.float32)
    return np.transpose(d.reshape((w, h, c), order="F"), (1, 0, 2)).squeeze()

ap = argparse.ArgumentParser(); ap.add_argument("ws"); ap.add_argument("out")
ap.add_argument("--view", nargs=5, action="append", required=True, metavar=("NAME", "X0", "Y0", "X1", "Y1"))
ap.add_argument("--depth-pct", nargs=2, type=float, default=[1, 99], help="depth range from these percentiles of sparse points")
ap.add_argument("--visible", type=float, metavar="TOL", help="keep only points visible from the view (e.g. 0.05)")
ap.add_argument("--density", nargs=2, type=float, metavar=("R", "N"), help="then drop points with < N neighbours within R "
                "(far oblique views smear depth noise along near-horizontal rays; e.g. 0.02 20)")
ap.add_argument("--fused", default="dense/fused.ply", help="cloud to crop, relative to WS (e.g. a stricter re-fusion)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

rec = pycolmap.Reconstruction(f"{a.ws}/dense/sparse")                 # undistorted PINHOLE model = dense/images
pcd = o3d.io.read_point_cloud(f"{a.ws}/{a.fused}"); P = np.asarray(pcd.points)
print(f"fused: {len(P):,} points")
keep = np.zeros(len(P), bool)
for name, *box in a.view:
    x0, y0, x1, y1 = map(float, box)
    im = rec.find_image_with_name(name); assert im is not None, f"{name} not in the dense model"
    cam = rec.cameras[im.camera_id]; K = cam.calibration_matrix(); M = im.cam_from_world().matrix()
    def project(X):
        Xc = X @ M[:, :3].T + M[:, 3]; z = Xc[:, 2]
        uv = (Xc @ K.T)[:, :2] / np.where(z > 0, z, np.inf)[:, None]; return uv, z
    def inbox(uv, z): return (z > 0) & (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    S = np.array([rec.points3D[p.point3D_id].xyz for p in im.points2D if p.has_point3D()])
    uv, z = project(S); zs = z[inbox(uv, z)]
    lo, hi = np.percentile(zs, a.depth_pct) * [0.9, 1.1]
    uv, z = project(P); k = inbox(uv, z) & (z > lo) & (z < hi)
    if a.visible:
        D = read_depth(f"{a.ws}/dense/stereo/depth_maps/{name}.geometric.bin")
        D = cv2.erode(np.where(D > 0, D, np.inf).astype(np.float32), np.ones((9, 9), np.uint8))   # min filter
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, D.shape[1] - 1); v = np.clip(np.round(uv[:, 1]).astype(int), 0, D.shape[0] - 1)
        Dv = D[v, u].astype(np.float64); k &= np.isfinite(Dv) & (z < (1 + a.visible) * Dv)
    keep |= k
    print(f"view {name} box {box}: {len(zs)} sparse points, depth {lo:.3f}..{hi:.3f} -> {k.sum():,} dense points")
roi = pcd.select_by_index(np.flatnonzero(keep))
if a.density:
    n0 = len(roi.points); roi, _ = roi.remove_radius_outlier(int(a.density[1]), a.density[0])
    print(f"density filter r={a.density[0]} n>={int(a.density[1])}: {n0:,} -> {len(roi.points):,}")
o3d.io.write_point_cloud(f"{a.out}/fused_focus.ply", roi)
print(f"kept {len(roi.points):,} points -> {a.out}/fused_focus.ply")
