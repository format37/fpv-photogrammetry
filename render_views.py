"""Render previews of the mesh / dense cloud with Open3D (offscreen, EGL):
  - top-down and oblique overviews in the ground-levelled frame
  - "reality check": the mesh rendered from the bundle-adjusted pose of a real frame,
    side by side with that (undistorted) frame.

usage: render_views.py DENSE_DIR OUT_DIR [--frames t037.167.jpg t138.833.jpg]
"""
import argparse, os, sys, numpy as np, open3d as o3d, pycolmap, cv2
R_ = o3d.visualization.rendering

ap = argparse.ArgumentParser(); ap.add_argument("dense"); ap.add_argument("out")
ap.add_argument("--frames", nargs="*", default=[]); ap.add_argument("--geom", default="mesh_poisson.ply")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

path = f"{a.dense}/{a.geom}"
is_mesh = "mesh" in a.geom
g = o3d.io.read_triangle_mesh(path) if is_mesh else o3d.io.read_point_cloud(path)
nx, ny, nz, e, height = np.loadtxt(f"{a.dense}/ground_plane.txt"); n = np.array([nx, ny, nz])
mat = R_.MaterialRecord(); mat.shader = "defaultUnlit"; mat.point_size = 2.0

def renderer(w, h):
    r = R_.OffscreenRenderer(w, h); r.scene.set_background([0.93, 0.94, 0.96, 1.0])
    r.scene.add_geometry("g", g, mat); return r

V = np.asarray(g.vertices if is_mesh else g.points); ctr = np.median(V, axis=0)
ctr = ctr - (ctr @ n + e) * n                                     # on the ground
x = np.cross(n, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(n, x)
ext = np.percentile(np.abs((V - ctr) @ np.stack([x, y]).T), 97)

r = renderer(1600, 1200)
views = {"top": (90, 0)} | {f"oblique_{az}": (35, az) for az in (0, 90, 180, 270)}
for name, (el, az) in views.items():
    el_r, az_r = np.radians(el), np.radians(az)
    dirv = np.cos(el_r) * (np.cos(az_r) * x + np.sin(az_r) * y) + np.sin(el_r) * n
    dist = ext / np.tan(np.radians(25)) * (1.0 if el == 90 else 1.15)
    up = y if el == 90 else n
    r.setup_camera(50, ctr, ctr + dirv * dist, up)
    o3d.io.write_image(f"{a.out}/{name}.png", r.render_to_image()); print("wrote", name)

# reality check from real camera poses (undistorted PINHOLE model of the dense workspace)
rec = pycolmap.Reconstruction(f"{a.dense}/sparse")
r.scene.clear_geometry()          # free the overview copy; one renderer per image size (a renderer per frame segfaults on ~20M-tri meshes)
check_r = {}
for fname in a.frames:
    im = rec.find_image_with_name(fname)
    if im is None: print("not in model:", fname); continue
    cam = rec.cameras[im.camera_id]; W, H = cam.width, cam.height
    K = cam.calibration_matrix()
    E = np.eye(4); E[:3] = im.cam_from_world().matrix()
    rr = check_r.get((W, H)) or check_r.setdefault((W, H), renderer(W, H)); rr.setup_camera(K, E, W, H)
    ren = cv2.cvtColor(np.asarray(rr.render_to_image()), cv2.COLOR_RGB2BGR)
    photo = cv2.imread(f"{a.dense}/images/{fname}")
    both = np.hstack([photo, ren])
    for txt, x0 in (("real frame " + fname, 20), ("mesh rendered from its SfM pose", W + 20)):
        cv2.putText(both, txt, (x0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
        cv2.putText(both, txt, (x0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    cv2.imwrite(f"{a.out}/check_{fname[:-4]}.jpg", cv2.resize(both, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote check", fname)

sys.stdout.flush(); os._exit(0)   # skip Open3D renderer teardown (aborts at interpreter exit); _exit does not flush
