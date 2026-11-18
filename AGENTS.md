# Instructions for AI agents

You are helping someone turn drone video into a 3D model with the tools in this repo. Read README.md
first. Rules that prevent the expensive mistakes:

## Workflow

1. **Let the human choose the segments.** Make proxies (`make_proxies.sh`), ask which time ranges of
   which clips to use, and say plainly if the footage looks unsuitable (see `docs/capture.md`).
2. **One prefix per clip** in keyframe names (`c40_t047.250.jpg`): the tools parse clip and time from it.
3. **Inspect before dense.** Run `inspect_model.py` on the sparse model. Dense takes 1.5 h; a bad pose
   wastes all of it. Drop or re-register anything it flags.
4. **Long jobs run detached** (`setsid nohup ... &`), log to a file, and are resumable where possible.
   Never edit a shell script while it is running: bash reads it incrementally.
5. **Report measured numbers**: frames registered, reprojection error recomputed from poses, GPS
   residual, share of the grid that is interpolated. Do not call a result good because it rendered.

## Verification

- Reprojection error measures self-consistency only. Always add an independent check: GPS residuals,
  a render compared with a frame from a *different* pass, or `localize_pnp.py` on unseen frames.
- A mirrored pass reprojects fine. `inspect_model.py` reports "normal depth order N/M" per clip; a
  clip with most frames inverted is mirrored. Fix: drop it, or re-register it against the rest
  (`subset_db.py` → `colmap image_registrator` → `point_triangulator` → `bundle_adjuster`).
- After `bundle_adjuster` or `model_merger`, stored point errors are stale. Recompute.
- In the GPS fit, per-clip residuals of ~1 m with a larger global residual mean drift between passes;
  a clip that is off by hundreds of metres is misregistered.

## Telemetry

- The log is ArduPilot DataFlash (`.BIN`). Use `extract_telemetry.py`; do not parse times by hand.
- Check the sync with the video: the launch should be visible at the logged launch time, and the
  video length should match the armed time.
- `georef.py --clip NAME=OFFSET`: offset of each clip file from the start of the recording, from
  `ffprobe` durations. A recording may be split into several files.

## Environment

- COLMAP 3.11 CLI plus pycolmap 3.13. In pycolmap 3.13 `image.cam_from_world()` is a method and frames
  are deregistered with `rec.deregister_frame(image.frame_id)`.
- Run Python tools that read downloaded data with `python3 -I`.
- Software decode for 10-bit HEVC when exact frame counts matter; some CUDA decode paths duplicate frames.
- The demo is static: rebuild `demo/canyon-lab.js` from `demo/lab.ts` with
  `esbuild demo/lab.ts --bundle --format=esm --minify --loader:.css=text --outfile=demo/canyon-lab.js`
  (needs `three` installed).

## Privacy

Flight logs contain precise home coordinates. Ask before publishing a log, a track, or a model.
