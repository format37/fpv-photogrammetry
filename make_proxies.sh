#!/usr/bin/env bash
# Light 720p30 H.264 viewing copies of every video in DIR -> DIR/proxy/<name>_proxy.mp4
# GPU decode + scale (NVDEC/scale_cuda) + NVENC; timing is 1:1 with the original.
# usage: make_proxies.sh DIR [PARALLEL=2]
set -euo pipefail
DIR=${1:?usage: make_proxies.sh DIR [PARALLEL]}; PAR=${2:-2}
mkdir -p "$DIR/proxy"
one() {
  f=$1; out="$(dirname "$f")/proxy/$(basename "${f%.*}")_proxy.mp4"
  [ -s "$out" ] && { echo "skip $out"; return; }
  pix=$(ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 "$f")
  case "$pix" in *10le|*10be) dl="p010le";; *) dl="nv12";; esac
  audio="-an"; ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$f" | grep -q . && audio="-map 0:a:0 -c:a aac -b:a 96k"
  ffmpeg -hide_banner -loglevel error -y -hwaccel cuda -hwaccel_output_format cuda -i "$f" -map 0:v:0 $audio \
    -vf "fps=30,scale_cuda=1280:720,hwdownload,format=$dl,format=yuv420p" \
    -c:v h264_nvenc -preset p4 -rc vbr -cq 26 -b:v 4M -maxrate 6M -g 30 -movflags +faststart "$out" \
  || ffmpeg -hide_banner -loglevel error -y -i "$f" -map 0:v:0 $audio -vf "fps=30,scale=1280:-2,format=yuv420p" \
       -c:v libx264 -preset veryfast -crf 24 -g 30 -movflags +faststart "$out"     # CPU fallback
  echo "done $out"
}
export -f one
find "$DIR" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) ! -name '*_proxy.mp4' -print0 \
  | xargs -0 -P "$PAR" -I{} bash -c 'one "$1"' _ {}
