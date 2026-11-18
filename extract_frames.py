"""Decode segments of a DJI 4K120 clip on the GPU at a candidate rate, keep the sharpest
frame of every window (variance of Laplacian), write JPEGs named by source time.

usage: extract_frames.py VIDEO OUTDIR START-END [START-END ...] [--keep-fps 4 --cand-fps 12 --width 1920 --prefix c08_]
"""
import argparse, os, subprocess, numpy as np, cv2

def hw_fmt(video):   # CUDA frames are p010 for 10-bit sources, nv12 for 8-bit
    pix = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt",
                          "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip()
    return "p010le" if "10" in pix else "nv12"

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("outdir"); ap.add_argument("segments", nargs="+")
ap.add_argument("--keep-fps", type=float, default=4); ap.add_argument("--cand-fps", type=float, default=12)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--log", default="", help="append name,sharpness,min_sharpness rows to this CSV")
ap.add_argument("--prefix", default="", help="frame name prefix, e.g. the clip id, when mixing several videos")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
W, H = a.width, a.width * 9 // 16
group = int(round(a.cand_fps / a.keep_fps))
log = open(a.log, "a") if a.log else None

for seg in a.segments:
    t0, t1 = map(float, seg.split("-"))
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-ss", str(t0), "-t", str(t1 - t0), "-i", a.video, "-map", "0:v:0",
           "-vf", f"fps={a.cand_fps},scale_cuda={W}:{H},hwdownload,format={hw_fmt(a.video)},format=bgr24",
           "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    buf, k, kept = [], 0, 0
    while True:
        raw = p.stdout.read(W * H * 3)
        if len(raw) < W * H * 3: break
        img = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(g[H // 3:], cv2.CV_64F).var()   # ignore the sky band
        buf.append((sharp, t0 + k / a.cand_fps, img)); k += 1
        if len(buf) == group:
            s, t, best = max(buf, key=lambda x: x[0])
            name = f"{a.prefix}t{t:07.3f}.jpg"
            cv2.imwrite(os.path.join(a.outdir, name), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept += 1
            if log: log.write(f"{name},{s:.1f},{min(b[0] for b in buf):.1f}\n")
            buf = []
    p.wait()
    print(f"segment {seg}: {k} candidates -> {kept} kept")
