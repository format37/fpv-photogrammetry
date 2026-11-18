"""Localize query frames against a COLMAP sparse model by direct 2D-3D matching + PnP.

Pipeline (what an SfM "image registration" step does, written out by hand):
  1. 3D descriptor bank: every 3D point gets the mean of the SIFT descriptors of its
     observations (read from COLMAP's database.db), L2-normalised.
  2. Query: SIFT with pycolmap (same implementation as the DB), FLANN 2-NN against the bank,
     Lowe ratio test -> 2D-3D correspondences.
  3. cv2.solvePnPRansac (P3P minimal solver, OPENCV distortion model) -> pose + inliers,
     then Levenberg-Marquardt refinement on the inliers (solvePnPRefineLM).
  4. Cross-check with pycolmap.estimate_and_refine_absolute_pose (LO-RANSAC + Ceres refine).

Modes:
  --loo N      leave-one-out on every N-th registered image: the image's own observations are
               removed from the descriptor bank, its pose is re-estimated and compared with the
               bundle-adjusted pose (rotation error in deg, position error in model units and
               as % of the mean camera-to-scene distance).
  --query DIR  localize unseen frames (e.g. a pass not used in the reconstruction) and write
               their poses to a CSV.
"""
import argparse, csv, sqlite3, sys, numpy as np, cv2, pycolmap

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--database", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--loo", type=int, default=0); ap.add_argument("--query")
ap.add_argument("--out", default="pnp_results.csv"); ap.add_argument("--ratio", type=float, default=0.8)
ap.add_argument("--reproj", type=float, default=4.0, help="RANSAC inlier threshold, px")
a = ap.parse_args()

rec = pycolmap.Reconstruction(a.model)
cam = next(iter(rec.cameras.values()))
fx, fy, cx, cy, k1, k2, p1, p2 = cam.params           # OPENCV model == cv2 (k1,k2,p1,p2)
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); dist = np.array([k1, k2, p1, p2])
print(f"camera {cam.model.name} {cam.width}x{cam.height} f=({fx:.1f},{fy:.1f}) c=({cx:.1f},{cy:.1f}) "
      f"k1={k1:.4f} k2={k2:.4f} p=({p1:.5f},{p2:.5f})")

# ---- 1. descriptor bank: per-3D-point descriptor sums, so leave-one-out is a subtraction ----
db = sqlite3.connect(a.database)
name2dbid = {n: i for i, n in db.execute("select image_id, name from images")}
pid_list = np.array(sorted(rec.points3D.keys()))
pid_index = {p: i for i, p in enumerate(pid_list)}
xyz = np.array([rec.points3D[p].xyz for p in pid_list])
dsum = np.zeros((len(pid_list), 128)); dcnt = np.zeros(len(pid_list))
obs = {}                                              # image_id -> (bank rows, descriptors)
for iid in rec.reg_image_ids():
    img = rec.images[iid]
    r = db.execute("select rows, cols, data from descriptors where image_id=?", (name2dbid[img.name],)).fetchone()
    D = np.frombuffer(r[2], np.uint8).reshape(r[0], r[1]).astype(np.float64)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    rows, descs = [], []
    for k, p2d in enumerate(img.points2D):
        if p2d.has_point3D():
            j = pid_index[p2d.point3D_id]; rows.append(j); descs.append(D[k])
    rows, descs = np.array(rows), np.array(descs)
    np.add.at(dsum, rows, descs); np.add.at(dcnt, rows, 1)
    obs[iid] = (rows, descs)
print(f"bank: {len(pid_list)} 3D points, {int(dcnt.sum())} observations")

sift = pycolmap.Sift(pycolmap.FeatureExtractionOptions(), device=pycolmap.Device.auto)
flann_params = (dict(algorithm=1, trees=8), dict(checks=128))

def build_index(sums, cnts):
    keep = cnts > 0
    B = (sums[keep] / cnts[keep, None]); B /= np.linalg.norm(B, axis=1, keepdims=True)
    m = cv2.FlannBasedMatcher(*flann_params); m.add([B.astype(np.float32)]); m.train()
    return m, np.flatnonzero(keep)

def query_features(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kp, d = sift.extract(g.astype(np.float32) / 255.0)
    d = d.astype(np.float64); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return kp[:, :2].astype(np.float64), d.astype(np.float32)

def localize(path, matcher, bank_rows):
    kp, d = query_features(path)
    knn = matcher.knnMatch(d, k=2)
    good = [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < a.ratio * n.distance]
    if len(good) < 12: return None
    q = np.array([g[0] for g in good]); t = bank_rows[np.array([g[1] for g in good])]
    p2, p3 = kp[q], xyz[t]
    ok, rvec, tvec, inl = cv2.solvePnPRansac(p3, p2, K, dist, iterationsCount=5000,
                                             reprojectionError=a.reproj, confidence=0.9999,
                                             flags=cv2.SOLVEPNP_AP3P)
    if not ok or inl is None or len(inl) < 12: return None
    inl = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(p3[inl], p2[inl], K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(p3[inl], rvec, tvec, K, dist)
    err = np.linalg.norm(proj[:, 0] - p2[inl], axis=1)
    R = cv2.Rodrigues(rvec)[0]
    # independent estimate with COLMAP's own absolute-pose solver
    res = pycolmap.estimate_and_refine_absolute_pose(p2, p3, cam)
    cc = None
    if res is not None:
        Rc = res["cam_from_world"].rotation.matrix(); tc = res["cam_from_world"].translation
        cc = (rot_err(R, Rc), np.linalg.norm(-R.T @ tvec[:, 0] - (-Rc.T @ tc)), int(res["num_inliers"]))
    return dict(R=R, t=tvec[:, 0], C=-R.T @ tvec[:, 0], n_match=len(good), n_inl=len(inl),
                rmse=float(np.sqrt((err ** 2).mean())), colmap=cc)

def rot_err(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))

# scene scale: mean distance camera -> its observed 3D points (to express errors in %)
depths = []
for iid in list(rec.reg_image_ids())[::5]:
    im = rec.images[iid]; C = im.projection_center()
    depths += [np.linalg.norm(rec.points3D[p.point3D_id].xyz - C) for p in im.points2D if p.has_point3D()][::20]
scene_d = float(np.median(depths)); print(f"median camera->point distance: {scene_d:.3f} model units")

out = open(a.out, "w", newline=""); w = csv.writer(out)
w.writerow(["image", "mode", "n_match", "n_inliers", "rmse_px", "rot_err_deg", "pos_err", "pos_err_pct",
            "vs_colmap_rot_deg", "vs_colmap_pos", "cx", "cy", "cz"])

if a.loo:
    ids = sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name)[::a.loo]
    errs = []
    for iid in ids:
        rows, descs = obs[iid]
        s, c = dsum.copy(), dcnt.copy()
        np.subtract.at(s, rows, descs); np.subtract.at(c, rows, 1)    # forget this image
        m, bank_rows = build_index(s, c)
        im = rec.images[iid]; r = localize(f"{a.images}/{im.name}", m, bank_rows)
        if r is None: print(im.name, "FAILED"); continue
        Rg = im.cam_from_world().rotation.matrix(); Cg = im.projection_center()
        re, pe = rot_err(r["R"], Rg), np.linalg.norm(r["C"] - Cg)
        errs.append((re, pe / scene_d * 100, r["rmse"], r["n_inl"]))
        w.writerow([im.name, "loo", r["n_match"], r["n_inl"], f"{r['rmse']:.3f}", f"{re:.4f}", f"{pe:.5f}",
                    f"{pe / scene_d * 100:.4f}", f"{r['colmap'][0]:.4f}" if r["colmap"] else "",
                    f"{r['colmap'][1]:.5f}" if r["colmap"] else "", *np.round(r["C"], 5)])
        print(f"{im.name}: {r['n_inl']:5d}/{r['n_match']:5d} inl  rmse {r['rmse']:.2f}px  "
              f"rot {re:.3f}deg  pos {pe / scene_d * 100:.3f}% of depth")
    e = np.array(errs)
    print(f"\nLOO {len(e)}/{len(ids)} localized | rot err median {np.median(e[:,0]):.3f} deg (max {e[:,0].max():.3f}) | "
          f"pos err median {np.median(e[:,1]):.3f}% of scene depth (max {e[:,1].max():.3f}%) | "
          f"inlier rmse median {np.median(e[:,2]):.2f}px | inliers median {int(np.median(e[:,3]))}")

if a.query:
    import glob, os
    m, bank_rows = build_index(dsum, dcnt)
    files = sorted(glob.glob(f"{a.query}/*.jpg")); okn = 0
    for f in files:
        r = localize(f, m, bank_rows)
        if r is None: print(os.path.basename(f), "FAILED"); continue
        okn += 1
        w.writerow([os.path.basename(f), "query", r["n_match"], r["n_inl"], f"{r['rmse']:.3f}", "", "", "",
                    f"{r['colmap'][0]:.4f}" if r["colmap"] else "", f"{r['colmap'][1]:.5f}" if r["colmap"] else "",
                    *np.round(r["C"], 5)])
        print(f"{os.path.basename(f)}: {r['n_inl']:5d}/{r['n_match']:5d} inl  rmse {r['rmse']:.2f}px  "
              f"opencv-vs-colmap {r['colmap'][0]:.3f}deg" if r["colmap"] else "")
    print(f"\nquery: {okn}/{len(files)} localized")
out.close()
