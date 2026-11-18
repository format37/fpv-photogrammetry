#!/usr/bin/env python3
"""voxview - fast coloured voxel viewer for SfM/MVS point clouds (Open3D native window).

    python3 voxview.py WORKSPACE                      # dense/dense_roi.ply (or fused.ply) + the model in best_model.txt (else the largest in sparse/)
    python3 voxview.py cloud.ply --model sparse/1     # any coloured PLY; the model adds cameras + levelling
    python3 voxview.py sparse/1                       # just the sparse SfM points
    python3 voxview.py WORKSPACE --voxel 0.02 --min-pts 3 --screenshot shot.png

The cloud is levelled (RANSAC ground plane -> z up), cropped to the area under the flight path,
binned into voxels; each voxel gets the mean colour of its points (= the colour from the video).

Keys:  [ / ]  finer / coarser voxels      C  colour: video RGB <-> height ramp <-> point count
       T      camera trajectory on/off    B  background dark/light
       L      lighting on/off             R  reset view        H  help        Q/Esc  quit
Mouse: left-drag rotate, right/ctrl-drag pan, wheel zoom.
"""
import argparse, glob, os, sys, numpy as np, open3d as o3d

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("input", help="workspace dir, .ply file, or COLMAP sparse model dir")
ap.add_argument("--model", help="COLMAP sparse model dir (cameras, levelling, crop)")
ap.add_argument("--voxel", type=float, help="voxel size in model units (default: ~700 voxels across)")
ap.add_argument("--min-pts", type=int, default=2, help="drop voxels with fewer points (noise)")
ap.add_argument("--no-crop", action="store_true"); ap.add_argument("--no-level", action="store_true")
ap.add_argument("--screenshot", help="render one frame to this PNG and exit")
ap.add_argument("--size", default="1600x1000")
ap.add_argument("--color", choices=["rgb", "height", "count"], default="rgb", help="initial colour mode")
ap.add_argument("--view", default="-0.45,-0.75,0.55", help="initial view direction (front vector, z up)")
ap.add_argument("--zoom", type=float, default=0.38)
a = ap.parse_args()

# ---------------------------------------------------------------- load
def is_model(p): return os.path.isdir(p) and (os.path.exists(f"{p}/points3D.bin") or os.path.exists(f"{p}/points3D.txt"))

def largest_model(ws):
    import pycolmap
    best, n = None, -1
    for d in sorted(glob.glob(f"{ws}/sparse/*/")):
        k = pycolmap.Reconstruction(d).num_reg_images()
        if k > n: best, n = d, k
    return best

src, model = a.input, a.model
if os.path.isdir(src) and not is_model(src):                      # workspace
    ws = src
    for cand in ("dense/dense_roi.ply", "dense/fused.ply"):
        if os.path.exists(f"{ws}/{cand}"): src = f"{ws}/{cand}"; break
    else: src = largest_model(ws)
    bm = f"{ws}/best_model.txt"                                    # written by run_dense.sh (honours MODEL=...)
    bm = os.path.join(ws, open(bm).read().strip()) if os.path.exists(bm) else None
    model = model or (bm if bm and is_model(bm) else largest_model(ws))
if is_model(src): model = model or src

cams = None
if model:
    import pycolmap
    rec = pycolmap.Reconstruction(model)
    ims = sorted([im for im in rec.images.values() if im.has_pose], key=lambda im: im.name)
    cams = np.array([im.projection_center() for im in ims])
    cam_R = [im.cam_from_world().rotation.matrix() for im in ims]
    cam_K = rec.cameras[ims[0].camera_id]

if is_model(src):
    P = np.array([p.xyz for p in rec.points3D.values()]); RGB = np.array([p.color for p in rec.points3D.values()]) / 255.0
    print(f"sparse model {src}: {len(P):,} points")
else:
    pc = o3d.io.read_point_cloud(src)
    if len(pc.points) == 0:
        m = o3d.io.read_triangle_mesh(src); pc = o3d.geometry.PointCloud(m.vertices); pc.colors = m.vertex_colors
    P = np.asarray(pc.points); RGB = np.asarray(pc.colors) if pc.has_colors() else np.full((len(P), 3), 0.7)
    print(f"{src}: {len(P):,} points")

# ---------------------------------------------------------------- level + crop
ref = cams.mean(0) if cams is not None else np.median(P, 0)
d = np.linalg.norm(P - ref, axis=1); med = np.median(d)
if not a.no_level:
    near = np.flatnonzero(d < 2 * med)
    sub = near[np.random.default_rng(0).choice(len(near), min(len(near), 400_000), replace=False)]
    (pa, pb, pc_, pd), _ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[sub])).segment_plane(0.02 * med, 3, 3000)
    n = np.array([pa, pb, pc_]); s = np.linalg.norm(n); n, pd = n / s, pd / s
    if cams is not None and (cams.mean(0) @ n + pd) < 0: n, pd = -n, -pd
    x = np.cross(n, [0, 0, 1.0]); x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
    Rl = np.stack([x, np.cross(n, x), n]); origin = -pd * n           # world -> levelled (z = height above ground)
else:
    Rl, origin = np.eye(3), np.zeros(3)
L = (P - origin) @ Rl.T
LC = (cams - origin) @ Rl.T if cams is not None else None

if not a.no_crop:
    if LC is not None:
        from scipy.spatial import cKDTree
        h = np.median(LC[:, 2]); h = h if h > 0 else med
        dist, _ = cKDTree(LC[:, :2]).query(L[:, :2], distance_upper_bound=1.5 * h, workers=-1)
        keep = np.isfinite(dist) & (L[:, 2] > -0.3 * h) & (L[:, 2] < 0.6 * h)
    else:
        lo, hi = np.percentile(L, 1, 0), np.percentile(L, 99, 0); keep = np.all((L >= lo) & (L <= hi), 1)
    L, RGB = L[keep], RGB[keep]
    print(f"cropped to the flown area: {len(L):,} points")

ext_xy = np.percentile(L[:, :2], 98, 0) - np.percentile(L[:, :2], 2, 0)
state = dict(voxel=a.voxel or float(ext_xy.max() / 700), cmode=["rgb", "height", "count"].index(a.color), traj=True, dark=True, light=True)

# ---------------------------------------------------------------- voxelise
def voxelise(s):
    q = np.floor(L / s).astype(np.int64); qmin = q.min(0); q -= qmin; dims = q.max(0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    col = np.stack([np.bincount(inv, RGB[:, c], len(uk)) for c in range(3)], 1) / cnt[:, None]
    ok = cnt >= a.min_pts
    uk, col, cnt = uk[ok], col[ok], cnt[ok]
    qi = np.stack([uk // (dims[1] * dims[2]), (uk // dims[2]) % dims[1], uk % dims[2]], 1) + qmin
    return (qi + 0.5) * s, col, cnt

def ramp(t):                                    # perceptual-ish height ramp (blue-teal-yellow-red)
    stops = np.array([[0.19, 0.25, 0.58], [0.13, 0.57, 0.55], [0.55, 0.78, 0.25], [0.99, 0.86, 0.28], [0.85, 0.27, 0.18]])
    t = np.clip(t, 0, 1) * (len(stops) - 1); i = np.minimum(t.astype(int), len(stops) - 2); f = (t - i)[:, None]
    return stops[i] * (1 - f) + stops[i + 1] * f

def build():
    C, col, cnt = voxelise(state["voxel"])
    if state["cmode"] == 1:
        z = C[:, 2]; lo, hi = np.percentile(z, [2, 98]); col = ramp((z - lo) / (hi - lo + 1e-9))
    elif state["cmode"] == 2:
        col = ramp(np.log(cnt) / np.log(np.percentile(cnt, 99)))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(C)); pc.colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, state["voxel"])
    mode = ["video RGB", "height", "points per voxel"][state["cmode"]]
    print(f"voxel {state['voxel']:.4g} | {len(C):,} voxels | colour: {mode}")
    return vg

def trajectory():
    if LC is None: return None
    s = state["voxel"] * 7; pts, lines, cols = [], [], []
    for i in range(len(LC) - 1):
        if np.linalg.norm(LC[i + 1] - LC[i]) < 10 * np.median(np.linalg.norm(np.diff(LC, axis=0), axis=1)) + 1e-9:
            pts += [LC[i], LC[i + 1]]; lines.append([len(pts) - 2, len(pts) - 1]); cols.append([0.95, 0.35, 0.2])
    for i in range(0, len(LC), max(1, len(LC) // 40)):           # small frusta
        Rw = Rl @ cam_R[i].T                                         # camera axes in levelled frame
        w, h = cam_K.width / 2, cam_K.height / 2; f = cam_K.mean_focal_length()
        corners = [LC[i] + Rw @ np.array([sx * w / f, sy * h / f, 1.0]) * s for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        b = len(pts); pts += [LC[i]] + corners
        lines += [[b, b + k] for k in range(1, 5)] + [[b + k, b + 1 + k % 4] for k in range(1, 5)]
        cols += [[1.0, 0.85, 0.3]] * 8
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(pts)), o3d.utility.Vector2iVector(np.array(lines)))
    ls.colors = o3d.utility.Vector3dVector(np.array(cols)); return ls

# ---------------------------------------------------------------- viewer
W, H = map(int, a.size.split("x"))
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("voxview - " + os.path.basename(os.path.normpath(a.input)), W, H, visible=not a.screenshot)
geo = {"vox": build(), "traj": trajectory()}
for g in geo.values():
    if g is not None: vis.add_geometry(g)

def style():
    ro = vis.get_render_option()
    ro.background_color = np.array([0.11, 0.12, 0.14]) if state["dark"] else np.array([0.94, 0.95, 0.97])
    ro.light_on = state["light"]; ro.line_width = 2.0

def home(v=vis):
    vc = v.get_view_control(); ctr = np.median(L, 0)
    vc.set_lookat(ctr); vc.set_up([0, 0, 1]); vc.set_front([float(x) for x in a.view.split(",")]); vc.set_zoom(a.zoom)
    return False

def rebuild(v):
    v.remove_geometry(geo["vox"], reset_bounding_box=False); geo["vox"] = build()
    v.add_geometry(geo["vox"], reset_bounding_box=False); return True

def finer(v): state["voxel"] /= 1.5; return rebuild(v)
def coarser(v): state["voxel"] *= 1.5; return rebuild(v)
def cmode(v): state["cmode"] = (state["cmode"] + 1) % 3; return rebuild(v)
def traj(v):
    if geo["traj"] is None: return False
    state["traj"] = not state["traj"]
    (v.add_geometry if state["traj"] else v.remove_geometry)(geo["traj"], reset_bounding_box=False); return True
def bg(v): state["dark"] = not state["dark"]; style(); return True
def light(v): state["light"] = not state["light"]; style(); return True
def helpk(v): print(__doc__.split("Keys:")[1]); return False

for k, f in {"[": finer, "]": coarser, "C": cmode, "T": traj, "B": bg, "L": light, "R": home, "H": helpk}.items():
    vis.register_key_callback(ord(k), f)
style(); home()

if a.screenshot:
    for _ in range(3): vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(a.screenshot, do_render=True); vis.destroy_window(); print("wrote", a.screenshot); sys.exit()
print("keys: [ ] voxel size | C colour mode | T trajectory | B background | L light | R reset | H help")
vis.run(); vis.destroy_window()
