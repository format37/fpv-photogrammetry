#!/usr/bin/env bash
# Dense MVS + mesh + previews on the largest sparse model of WS.
#   undistort -> PatchMatch (photometric + geometric) -> fusion -> crop/clean -> Poisson mesh -> renders
# usage: run_dense.sh WS            env: MAXSIZE=1920 (1280 ~2x faster)  SOURCES=10  DELAUNAY=0  SKIP_UNDISTORT=0  MODEL=sparse_x/0
#   MAX_SCALE=1.0: undistorted frame keeps the original focal and size, the stretched periphery is cropped.
#   COLMAP's default 2.0 widens the frame to keep the whole FOV, and MAXSIZE then shrinks it: the centre loses
#   ~20% resolution on a 100 deg lens and 50% on DJI FOV:MAX fisheye (1920x565 at half focal).
set -euo pipefail
WS=${1:?usage: run_dense.sh WS}; T=$(cd "$(dirname "$0")" && pwd)
MAXSIZE=${MAXSIZE:-1920}; SOURCES=${SOURCES:-10}; MAX_SCALE=${MAX_SCALE:-1.0}
cd "$WS"; ts() { date +%H:%M:%S; }

best=""; bestn=0
for d in ${MODEL:-sparse/*/}; do        # MODEL=path/to/model forces a model (e.g. a cleaned/merged one)
  n=$(python3 -c "import pycolmap,sys; print(pycolmap.Reconstruction(sys.argv[1]).num_reg_images())" "$d")
  echo "$(ts) model $d: $n images"; if [ "$n" -gt "$bestn" ]; then best=$d; bestn=$n; fi
done
echo "$(ts) using $best ($bestn images)"; echo "$best" > best_model.txt

if [ "${SKIP_UNDISTORT:-0}" != 1 ]; then
  echo "$(ts) undistort"
  colmap image_undistorter --image_path images --input_path "$best" --output_path dense \
    --output_type COLMAP --max_image_size "$MAXSIZE" --max_scale "$MAX_SCALE" > logs_undistort.txt 2>&1
fi
sed -i "s/^__auto__, [0-9]*$/__auto__, $SOURCES/" dense/stereo/patch-match.cfg   # source views per depth map

echo "$(ts) patch-match stereo"
colmap patch_match_stereo --workspace_path dense --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size "$MAXSIZE" > logs_patchmatch.txt 2>&1

echo "$(ts) fusion"
colmap stereo_fusion --workspace_path dense --workspace_format COLMAP --input_type geometric \
  --output_path dense/fused.ply --StereoFusion.min_num_pixels 4 > logs_fusion.txt 2>&1

echo "$(ts) crop + clean + poisson mesh"
python3 -I "$T/clean_and_mesh.py" dense/fused.ply "$best" dense

if [ "${DELAUNAY:-0}" = 1 ]; then     # slow on millions of points (hours); runs on the full fused cloud
  echo "$(ts) delaunay mesh"
  colmap delaunay_mesher --input_path dense --input_type dense --output_path dense/meshed-delaunay.ply > logs_delaunay.txt 2>&1 || true
fi

echo "$(ts) renders"
mkdir -p renders
mid=$(python3 -c "
import pycolmap,sys; r=pycolmap.Reconstruction(sys.argv[1]); n=sorted(i.name for i in r.images.values() if i.has_pose)
print(n[len(n)//4], n[3*len(n)//4])" "$best")
# previews are optional: a crash here must not hide the finished reconstruction
python3 -I "$T/render_views.py" dense renders --frames $mid 2>&1 | grep -v "^\[Open3D" || echo "render_views failed"
python3 -I "$T/viz_topdown.py" "$best" renders/sparse_topdown.png 2>&1 | grep -v "^\[Open3D" || echo "viz_topdown failed"
python3 "$T/voxview.py" . --screenshot renders/voxels_rgb.png 2>&1 | grep -v "^\[Open3D" || echo "voxview rgb failed"
python3 "$T/voxview.py" . --color height --screenshot renders/voxels_height.png 2>&1 | grep -v "^\[Open3D" || echo "voxview height failed"
echo "$(ts) DONE"
