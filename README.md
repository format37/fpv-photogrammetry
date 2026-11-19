# fpv-photogrammetry

Turn ordinary FPV flight video into a metric 3D terrain model, scale it with the flight controller's
GPS log, and measure things on it. Built around [COLMAP](https://colmap.github.io/) (SfM + MVS).

- Write-up: [Measuring a Canyon from FPV Video](https://latentstate.dev/posts/canyon-volume-from-fpv-video/)
- Source footage: <https://youtu.be/s8RxP2BK_d0>
- Example data: [format37/kojori-canyon-fpv](https://huggingface.co/datasets/format37/kojori-canyon-fpv) on
  Hugging Face — the keyframes, the GPS track and the reference outputs of the flight in the write-up.
- Demo: `demo/` — flood the reconstructed Kojori canyon between two dams and read the volume.

```
video ──► keyframes ──► sparse SfM ──► check ──► dense MVS ──► georeference ──► height grid ──► demo
          select_       run_sparse    inspect_   run_dense     georef.py        build_dsm.py    demo/
          keyframes                   model
flight log (.BIN) ──► extract_telemetry.py ──► GPS track ─────────┘
```

## Run the demo

```bash
cd demo && python3 -m http.server 8000      # open http://localhost:8000
```

It needs WebGL. Data for the Kojori canyon (2.8 MB) is in `demo/data/`.

## Try it on the example data

The Kojori flight is a public dataset (320 MB, CC BY 4.0): 241 keyframes, the flight controller's GPS
track, and the reference outputs of every stage. `examples/kojori.sh` runs the tools on it.

```bash
pip install -U huggingface_hub
examples/kojori.sh fetch     # download into ./kojori
examples/kojori.sh geo       # no GPU, under a minute: GPS fit + height grid from the reference outputs
examples/kojori.sh mini      # GPU, about 3 min: sparse SfM on 59 frames (two passes), model check, GPS fit
examples/kojori.sh full      # GPU, about 2.5 h: steps 2-7 below on all 241 frames
```

| run | what to expect |
|---|---|
| `geo` | `fit on 223 cameras: lag +1.15 s, scale 162.548 m/unit, rms 3.03 m`, then `height grid: identical to the reference` |
| `mini` | 59 of 59 frames registered, focal length near 800 px, mean reprojection error about 0.7 px, `normal depth order` for both clips, GPS fit about 3.4 m rms |
| `full` | numbers close to the Kojori result below; `inspect_model.py` should flag clip `c43` as mirrored |

`geo` is exact, so it doubles as a regression test for `georef.py` and `build_dsm.py`. COLMAP runs are
not bit-reproducible: `mini` and `full` give close numbers, not equal ones. The model's unit is arbitrary,
so the scale in metres per unit differs from run to run.

`mini` fixes the video-to-log lag at 1.15 s instead of searching for it. Its two passes fly the same
course, and a time lag then looks like an along-track shift that the fit absorbs. The lag is only
observable when the passes go in different directions, as in the full set.

## Requirements

- Linux, an NVIDIA GPU (dense stereo needs CUDA; 24 GB was used), `ffmpeg` with NVENC/NVDEC.
- COLMAP 3.11 built with CUDA, on `PATH`.
- Python 3.10+: `pip install -r requirements.txt` (pycolmap 3.13, Open3D 0.19 were used).

## Reproduce

`T` is this repo, `WS` a new workspace folder. Times are for 241 frames on an RTX 4090.

```bash
# 0. Light 720p copies for choosing segments (4K120 HEVC is heavy to scrub)
$T/make_proxies.sh /path/to/videos

# 1. Keyframes: one per ~2.5 % image shift, sharpest of each window. One prefix per clip.
python3 $T/select_keyframes.py CLIP_0040.MP4 $WS/images 0:47-2:28 --prefix c40_ --shift 0.025

# 2. Sparse SfM (SIFT, exhaustive matching, incremental mapper)                          ~35 min
$T/run_sparse.sh $WS 806                    # optional focal prior in px at frame width
#    wide/fisheye modes: CAMERA_MODEL=OPENCV_FISHEYE $T/run_sparse.sh $WS 810

# 3. Check the model BEFORE spending 1.5 h on dense
python3 -I $T/inspect_model.py $WS/sparse/0

# 4. Dense MVS, fusion, mesh, previews                                                   ~95 min
$T/run_dense.sh $WS                         # MODEL=path forces a specific/cleaned model

# 5. Telemetry: ArduPilot DataFlash log -> CSV (see below)
python3 $T/extract_telemetry.py 00000065.BIN $TEL

# 6. Georeference: SfM camera centres -> GPS track (7-DoF similarity + time lag search)
python3 -I $T/georef.py $WS $TEL/gps_5hz.csv $GEO --clip c39=0 c40=216.04 c41=432.12 c42=648.21

# 7. Height grid, orthophoto and canyon axis for the demo
python3 -I $T/build_dsm.py $GEO --roi -900 -2000 1500 100 --axis -420 -400 1250 -1620 --track $TEL/gps_5hz.csv
cp $GEO/web/* $T/demo/data/
```

`--clip NAME=OFFSET`: seconds from the start of the recording (= arm) to the start of that clip file.
For a recording split into files, the offset is the sum of the previous files' durations.

Viewing: `python3 $T/voxview.py $WS` (coloured voxels, camera track; keys `[ ]` voxel size, `C` colour mode).

## What each tool does

| tool | purpose |
|---|---|
| `make_proxies.sh` | 720p viewing copies, GPU transcode |
| `select_keyframes.py` | motion-adaptive keyframes (KLT track displacement), rejects blurred frames |
| `extract_frames.py` | fixed-rate alternative: sharpest frame per window |
| `dji_meta.py` | dump DJI O4 telemetry packets (focal length prior; there is no GPS in them) |
| `run_sparse.sh` | COLMAP features, matching, mapper |
| `inspect_model.py` | per-clip reprojection error, weak frames, pose jumps, **mirrored passes**; can drop frames/clips |
| `subset_db.py` | copy a COLMAP database keeping only some images (to re-register one pass) |
| `run_dense.sh` | undistort, PatchMatch stereo, fusion, crop, Poisson mesh, renders |
| `clean_and_mesh.py` | crop to the flown area, outlier removal, Poisson mesh |
| `crop_focus.py` | crop to an object seen from afar (view frusta + visibility + density filter) |
| `render_views.py`, `viz_topdown.py`, `voxview.py` | previews, top-down map, interactive voxel viewer |
| `localize_pnp.py` | localise unseen frames against the model (2D–3D matching + PnP), a model sanity check |
| `extract_telemetry.py` | ArduPilot `.BIN` → GPS/pose CSVs with arm events and clock correction |
| `georef.py` | similarity fit to GPS, writes the cloud in local east/north/up metres |
| `build_dsm.py` | height grid (DSM), orthophoto, least-cost canyon axis, web data |
| `examples/kojori.sh` | download the example dataset and run the tools on it |

## Telemetry

ArduPilot writes `APM/LOGS/NNNNNNNN.BIN` on the flight controller's SD card; copy it off.
`extract_telemetry.py` reads it with `pymavlink` and handles the parts that are easy to get wrong:

- picks the GPS the autopilot actually used;
- finds arm/disarm events (video recording that starts on arm shares its zero with the log);
- corrects the flight controller clock against GPS fix epochs (ours ran 0.1 % fast: 1.7 s per flight);
- writes time since arm, so `georef.py` only has to solve for the camera's start-up lag (about 1 s).

At 20 m/s, 0.1 s of timing error is 2 m of position error.

## Checks worth keeping

- **Low reprojection error is not correctness.** A pass over distant, nearly flat terrain can be
  reconstructed as a mirror image at 1.3 px. `inspect_model.py` flags it (for a down-tilted camera the
  bottom of the frame must be nearer than the top); GPS residuals confirm it.
- **A render from the frame that built the depth always matches.** Validate from a different pass.
- **`model_merger` can misalign silently.** Check merged models visually and against GPS.
- Stored per-point errors go stale after `bundle_adjuster`; `inspect_model.py` recomputes them.

Kojori result: 241 frames, 137 k sparse points, 0.75 px median reprojection error, 8.0 M dense points,
GPS fit 3.0 m rms over 223 cameras (0.7–1.5 m per pass), 2 m height grid.

## Shooting footage that reconstructs well

See [docs/capture.md](docs/capture.md). Short version: fly lower and slower than feels natural, make
several offset passes with the camera tilted 30–45° down, avoid haze and the sun in frame, lock the
camera's field of view, and log GPS.

## For AI agents

See [AGENTS.md](AGENTS.md).

## License

MIT. The demo bundles [three.js](https://threejs.org/) (MIT).
