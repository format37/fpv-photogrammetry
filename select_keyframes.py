#!/usr/bin/env python3
"""Motion-adaptive keyframe selection from a 4K drone clip (GPU decode).

Instead of a fixed frame rate, a new keyframe is taken when the image content has moved enough
since the last keyframe: KLT-track corners from the last keyframe to every candidate frame; once the
median track displacement exceeds --shift (fraction of the image width), or too few tracks survive,
the sharpest of the next --look candidates becomes the new keyframe. Hover / slow segments therefore
give few frames, fast segments many. Frames that are blurred or washed out (haze, flare) relative to
their neighbourhood are never chosen.

usage: select_keyframes.py VIDEO OUTDIR START-END [START-END ...] --prefix c40_ [--shift 0.06]
times as seconds or m:ss. Writes OUTDIR/<prefix>t<sec>.jpg and a CSV log next to OUTDIR.
"""
import argparse, os, subprocess, numpy as np, cv2

def hw_fmt(video):   # CUDA frames are p010 for 10-bit sources, nv12 for 8-bit
    pix = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt",
                          "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip()
    return "p010le" if "10" in pix else "nv12"

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("outdir"); ap.add_argument("segments", nargs="+")
ap.add_argument("--prefix", default=""); ap.add_argument("--cand-fps", type=float, default=12)
ap.add_argument("--shift", type=float, default=0.06, help="median flow since last keyframe, fraction of width")
ap.add_argument("--look", type=int, default=3, help="pick the sharpest of this many candidates")
ap.add_argument("--min-track", type=float, default=0.35, help="force a keyframe when fewer tracks survive")
ap.add_argument("--width", type=int, default=1920); ap.add_argument("--log", default="")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
W, H = a.width, a.width * 9 // 16
AW = 640; AH = AW * 9 // 16; s_ = AW / W                         # analysis resolution
log = open(a.log or os.path.join(os.path.dirname(os.path.abspath(a.outdir)), f"keyframes_{a.prefix or 'clip'}.csv"), "a")

def secs(x): return sum(float(p) * 60 ** i for i, p in enumerate(reversed(x.split(":"))))

def quality(g):
    ground = g[AH // 3:]                                          # lower 2/3: less sky
    return cv2.Laplacian(ground, cv2.CV_64F).var(), float(ground.std())

def corners(g):
    m = np.zeros_like(g); m[AH // 4:] = 255                       # do not track clouds / sky
    p = cv2.goodFeaturesToTrack(g, 400, 0.01, 12, mask=m)
    return p if p is not None else np.zeros((0, 1, 2), np.float32)

total = 0
for seg in a.segments:
    t0, t1 = map(secs, seg.split("-"))
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-ss", str(t0),
           "-t", str(t1 - t0), "-i", a.video, "-map", "0:v:0",
           "-vf", f"fps={a.cand_fps},scale_cuda={W}:{H},hwdownload,format={hw_fmt(a.video)},format=bgr24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    kf_g = kf_p = None; pending = []; k = 0; kept = []; sharp_hist = []
    def commit(cands):
        global kf_g, kf_p
        med = np.median(sharp_hist[-60:]) if sharp_hist else 0
        ok = [c for c in cands if c["sharp"] >= 0.6 * med] or cands
        best = max(ok, key=lambda c: c["sharp"])
        name = f"{a.prefix}t{best['t']:07.3f}.jpg"
        cv2.imwrite(os.path.join(a.outdir, name), best["img"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.write(f"{name},{best['sharp']:.1f},{best['contrast']:.1f},{best['shift']:.4f},{best['tracked']:.2f}\n")
        kf_g = best["g"]; kf_p = corners(kf_g); kept.append(best["t"])
    while True:
        raw = p.stdout.read(W * H * 3)
        if len(raw) < W * H * 3: break
        img = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
        g = cv2.cvtColor(cv2.resize(img, (AW, AH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp, contrast = quality(g); sharp_hist.append(sharp)
        c = dict(t=t0 + k / a.cand_fps, img=img.copy(), g=g, sharp=sharp, contrast=contrast, shift=0.0, tracked=1.0); k += 1
        if kf_g is None:                                           # first keyframe: sharpest of the first few
            pending.append(c)
            if len(pending) == a.look: commit(pending); pending = []
            continue
        if pending:                                                # look-ahead window after a trigger
            pending.append(c)
            if len(pending) == a.look: commit(pending); pending = []
            continue
        if len(kf_p):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(kf_g, g, kf_p, None, winSize=(21, 21), maxLevel=4)
            good = st[:, 0] == 1
            c["tracked"] = good.mean()
            c["shift"] = float(np.median(np.linalg.norm((nxt - kf_p)[good, 0], axis=1))) / AW if good.any() else 1.0
        else:
            c["tracked"], c["shift"] = 0.0, 1.0
        if c["shift"] >= a.shift or c["tracked"] < a.min_track:
            pending = [c]
    if pending: commit(pending)
    p.wait(); total += len(kept)
    gaps = np.diff(kept) if len(kept) > 1 else np.array([0])
    print(f"{a.prefix}{seg}: {k} candidates -> {len(kept)} keyframes "
          f"(gap median {np.median(gaps):.2f}s, min {gaps.min():.2f}s, max {gaps.max():.2f}s)")
print(f"total {total}")
