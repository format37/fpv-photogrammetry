"""Sanity-check a COLMAP model of drone keyframes (<prefix>t<sec>.jpg) and optionally write a cleaned copy.

Per clip: frames, reprojection error recomputed from the current poses/intrinsics (the stored per-point
errors go stale after bundle_adjuster/model_merger), share of observations > 4 px. Then suspects:
  - weak frames: few triangulated observations (default < 300);
  - pose jumps: frame-to-frame speed > JUMP x the clip's median speed (a drone cannot teleport;
    a jump that comes straight back is one misregistered frame);
  - mirrored passes: for a down-tilted camera the lower half of the frame is nearer than the upper half.
    A pass where most frames have it the other way round was reconstructed in the mirror solution of a
    distant, nearly planar scene (it still reprojects at ~1.5 px because its points are seen only by that pass).
With --drop / --drop-clips the listed frames/clips are deregistered and the model written to --out
(follow with `colmap bundle_adjuster`).

usage: inspect_model.py MODEL [--jump 4] [--min-obs 300] [--drop a.jpg b.jpg] [--drop-clips c09 c11 --out DIR]
"""
import argparse, collections, re, numpy as np, pycolmap

ap = argparse.ArgumentParser(); ap.add_argument("model")
ap.add_argument("--jump", type=float, default=4.0); ap.add_argument("--min-obs", type=int, default=300)
ap.add_argument("--drop", nargs="*", default=[]); ap.add_argument("--drop-clips", nargs="*", default=[])
ap.add_argument("--out")
a = ap.parse_args()
rec = pycolmap.Reconstruction(a.model)
cam = rec.cameras[min(rec.cameras)]
print(f"{rec.num_reg_images()} images, {rec.num_points3D()} points; camera {cam.model.name} "
      f"{[round(float(x), 4) for x in cam.params]}")

def clip_time(name):
    m = re.match(r"(.*?)_?t(\d+\.\d+)\.jpg$", name); return (m.group(1), float(m.group(2))) if m else (name, 0.0)

P = {pid: p.xyz for pid, p in rec.points3D.items()}
per = collections.defaultdict(list); frames = collections.defaultdict(list); suspects = []
order = collections.defaultdict(list)       # lower image half nearer than the upper half = normal for a down-tilted camera
for im in sorted(rec.images.values(), key=lambda i: i.name):
    c = rec.cameras[im.camera_id]
    obs = [(p.xy, P[p.point3D_id]) for p in im.points2D if p.has_point3D()]
    clip, t = clip_time(im.name); frames[clip].append((t, im.projection_center(), im.name))
    if len(obs) < a.min_obs: suspects.append((im.name, f"only {len(obs)} observations"))
    if not obs: continue
    xy = np.array([o[0] for o in obs]); X = np.array([o[1] for o in obs])
    M = im.cam_from_world().matrix(); Xc = X @ M[:, :3].T + M[:, 3]; uv = np.array(c.img_from_cam(Xc))
    per[clip].append(np.nan_to_num(np.linalg.norm(uv - xy, axis=1), nan=1e3))   # NaN = outside the lens domain
    low = xy[:, 1] >= c.height / 2
    if len(obs) >= 50 and low.any() and (~low).any():
        order[clip].append(np.median(Xc[low, 2]) < np.median(Xc[~low, 2]))

allerr = np.concatenate([e for v in per.values() for e in v])
print(f"all: mean {allerr.mean():.2f} px, median {np.median(allerr):.2f}, >4px {np.mean(allerr > 4) * 100:.1f}%")
for clip in sorted(frames):
    e = np.concatenate(per[clip]) if per[clip] else np.zeros(1)
    v = frames[clip]; sp = []
    for (t0, c0, n0), (t1, c1, n1) in zip(v, v[1:]):
        if 0 < t1 - t0 < 3: sp.append((np.linalg.norm(c1 - c0) / (t1 - t0), n1))
    med = np.median([s for s, _ in sp]) if sp else 0
    od = order[clip]
    print(f"  {clip}: {len(v):3d} frames  mean {e.mean():5.2f} px  >4px {np.mean(e > 4) * 100:4.1f}%  "
          f"median speed {med:.3f} u/s  normal depth order {sum(od)}/{len(od)}")
    if od and sum(od) < len(od) / 2:
        suspects.append((clip, f"depth order inverted in {len(od) - sum(od)}/{len(od)} frames: the pass is probably "
                               "reconstructed mirrored (distant near-planar scene); re-register it (see AGENTS.md)"))
    for (s, n), (s2, _) in zip(sp, sp[1:] + [(0, None)]):
        if med and s > a.jump * med and s2 > a.jump * med:        # out and straight back: one bad frame
            suspects.append((n, f"pose jump {s / med:.1f}x the clip's median speed, and back"))
for n, why in suspects: print(f"SUSPECT {n}: {why}")

drop = set(a.drop) | {im.name for im in rec.images.values() if clip_time(im.name)[0] in a.drop_clips}
if drop:
    assert a.out, "--out is required with --drop/--drop-clips"
    for f in {im.frame_id for im in rec.images.values() if im.name in drop}: rec.deregister_frame(f)
    import os; os.makedirs(a.out, exist_ok=True); rec.write(a.out)
    print(f"dropped {len(drop)} frames -> {rec.num_reg_images()} images, written to {a.out} "
          f"(now run: colmap bundle_adjuster --input_path {a.out} --output_path ... --BundleAdjustment.refine_principal_point 0)")
