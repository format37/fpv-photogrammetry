#!/usr/bin/env bash
# Sparse SfM on WS/images (one sub-folder per clip = one camera per clip).
#   features (GPU SIFT) -> matching -> incremental mapper -> stats
# usage: run_sparse.sh WS [FOCAL_PX_AT_FRAME_WIDTH]
#   focal prior: optional; for DJI O4 Pro use dji_meta.py --focal (1611.56 @3840 -> 805.8 @1920)
#   env CAMERA_MODEL=OPENCV (default) | OPENCV_FISHEYE (DJI FOV:MAX / strong barrel) - both take f,f,cx,cy + 4 coeffs
set -euo pipefail
WS=${1:?usage: run_sparse.sh WS [FOCAL]}; FOCAL=${2:-}
cd "$WS"; ts() { date +%H:%M:%S; }
N=$(find images -type f -iname '*.jpg' | wc -l); echo "$(ts) $N images"
W=$(python3 -c "import cv2,glob; print(cv2.imread(sorted(glob.glob('images/**/*.jpg', recursive=True))[0]).shape[1])")
H=$(python3 -c "import cv2,glob; print(cv2.imread(sorted(glob.glob('images/**/*.jpg', recursive=True))[0]).shape[0])")
CAM=(--ImageReader.camera_model "${CAMERA_MODEL:-OPENCV}" --ImageReader.single_camera_per_folder 1)
[ -n "$FOCAL" ] && CAM+=(--ImageReader.camera_params "$FOCAL,$FOCAL,$((W/2)),$((H/2)),0,0,0,0")

rm -f database.db
echo "$(ts) features"
colmap feature_extractor --database_path database.db --image_path images "${CAM[@]}" \
  --SiftExtraction.use_gpu 1 --SiftExtraction.max_num_features 8192 > logs_extract.txt 2>&1

if [ "$N" -le 400 ]; then
  echo "$(ts) exhaustive matching (finds loop closures between passes)"
  colmap exhaustive_matcher --database_path database.db --SiftMatching.use_gpu 1 --SiftMatching.guided_matching 1 > logs_match.txt 2>&1
else
  echo "$(ts) sequential matching ($N > 400 images; loop closures only within overlap)"
  colmap sequential_matcher --database_path database.db --SiftMatching.use_gpu 1 --SiftMatching.guided_matching 1 \
    --SequentialMatching.overlap 30 --SequentialMatching.quadratic_overlap 1 > logs_match.txt 2>&1
fi

echo "$(ts) mapper"
mkdir -p sparse
colmap mapper --database_path database.db --image_path images --output_path sparse \
  --Mapper.ba_refine_principal_point 0 --Mapper.ba_global_images_ratio 1.3 --Mapper.ba_global_points_ratio 1.3 \
  > logs_mapper.txt 2>&1

for d in sparse/*/; do
  echo "== $d"; colmap model_analyzer --path "$d" 2>&1 | sed -n 's/^.*\] //p' | grep -E "Registered|Points|track|reprojection"
  python3 -c "import pycolmap,sys; r=pycolmap.Reconstruction(sys.argv[1]); [print('  camera', c.camera_id, c.model.name, [round(float(x),4) for x in c.params]) for c in r.cameras.values()]" "$d"
done
echo "$(ts) sparse done"
