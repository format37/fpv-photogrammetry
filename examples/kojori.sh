#!/usr/bin/env bash
# Worked example on the public Kojori canyon dataset: https://huggingface.co/datasets/format37/kojori-canyon-fpv
#
#   examples/kojori.sh fetch [DIR]   download the dataset (320 MB) into DIR (default: ./kojori)
#   examples/kojori.sh geo   [DIR]   no GPU, under a minute: GPS fit of the reference SfM model, height grid from the
#                                    reference cloud; both must reproduce the reference outputs exactly
#   examples/kojori.sh mini  [DIR]   GPU, about 3 min: sparse SfM on 59 frames (two passes), model check, GPS fit
#   examples/kojori.sh full  [DIR]   GPU, about 2.5 h: everything from the 241 frames, as in the write-up
#
# Results go to DIR/work/. A fresh SfM run is not bit-reproducible: expect numbers close to the reference, not equal.
set -euo pipefail
T=$(cd "$(dirname "$0")/.." && pwd); CMD=${1:?usage: kojori.sh fetch|geo|mini|full [DIR]}; D=${2:-kojori}
CLIPS="c39=0 c40=216.040825 c41=432.123358 c42=648.205891 c43=864.255058"   # clip file start, seconds after arm (clips.csv)
ORIGIN="41.6510441 44.7295113 1116.98"       # lat lon alt of the local east/north/up frame used by the reference outputs
ROI="-900 -2000 1500 100"; AXIS="-420 -400 1250 -1620"                      # demo area and rough canyon end points, metres
FOCAL=805.78                                 # px at 1920 px width, from the DJI metadata (dji_meta.py --focal)
GPS=$D/telemetry/gps_5hz.csv

case $CMD in
fetch)
  hf download format37/kojori-canyon-fpv --repo-type dataset --local-dir "$D" ;;
geo)
  W=$D/work/geo; mkdir -p "$W"
  python3 -I "$T/georef.py" "$D/reference" "$GPS" "$W" --clip $CLIPS --drop c43 --home $ORIGIN
  cp "$D/reference/geo/cloud_enu.npz" "$W/"
  python3 -I "$T/build_dsm.py" "$W" --roi $ROI --axis $AXIS --track "$GPS"
  cmp "$W/web/dsm.png" "$D/reference/geo/web/dsm.png" && echo "height grid: identical to the reference"
  echo "view it: cp $W/web/* $T/demo/data/ && (cd $T/demo && python3 -m http.server 8000)" ;;
mini)
  W=$D/work/mini; mkdir -p "$W"; [ -e "$W/images" ] || cp -r "$D/mini/images" "$W/images"
  "$T/run_sparse.sh" "$W" $FOCAL
  python3 -I "$T/inspect_model.py" "$W/sparse/0"
  # Both passes of the subset fly the same course, so a time lag is absorbed by the fit as an along-track shift and
  # cannot be found from them. The lag is fixed to the value found on the full set (passes in several directions).
  python3 -I "$T/georef.py" "$W" "$GPS" "$W/geo" --clip $CLIPS --home $ORIGIN --lag 1.15 1.15 1 ;;
full)
  W=$D/work/full; mkdir -p "$W"; [ -e "$W/images" ] || cp -r "$D/images" "$W/images"
  "$T/run_sparse.sh" "$W" $FOCAL
  python3 -I "$T/inspect_model.py" "$W/sparse/0"      # in the reference run it flags clip c43 as a mirror image
  "$T/run_dense.sh" "$W"
  # c43 is left out of the GPS fit and its points are removed: correct if it is mirrored, harmless if it is not
  python3 -I "$T/georef.py" "$W" "$GPS" "$W/geo" --clip $CLIPS --drop c43 --home $ORIGIN
  python3 -I "$T/build_dsm.py" "$W/geo" --roi $ROI --axis $AXIS --track "$GPS"
  echo "view it: cp $W/geo/web/* $T/demo/data/ && (cd $T/demo && python3 -m http.server 8000)" ;;
*) echo "usage: kojori.sh fetch|geo|mini|full [DIR]"; exit 2 ;;
esac
