"""Dump one DJI djmd telemetry packet as protobuf fields (no .proto needed).

usage: dji_meta.py VIDEO [PACKET_INDEX]     (packet 0 = header with model/firmware)
DJI O4 Pro (dvtm_O4P): field 1.8.1 / 3.2.15.1 = focal length in px at recording width (1611.56 @3840,
confirmed by self-calibration: 1599.7). No GPS in O4 Air Unit telemetry.
"""
import struct, sys, subprocess

def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7f) << s; s += 7
        if c < 0x80: return r, i

def parse(b, path="", out=None, depth=0):
    out = [] if out is None else out
    i = 0
    try:
        while i < len(b):
            key, i = varint(b, i)
            f, wt = key >> 3, key & 7
            p = f"{path}.{f}" if path else str(f)
            if wt == 0:
                v, i = varint(b, i); out.append((p, "varint", v))
            elif wt == 1:
                v = struct.unpack("<d", b[i:i+8])[0]; i += 8; out.append((p, "double", v))
            elif wt == 5:
                v = struct.unpack("<f", b[i:i+4])[0]; i += 4; out.append((p, "float", v))
            elif wt == 2:
                n, i = varint(b, i); sub = b[i:i+n]; i += n
                before = len(out)
                ok = depth < 8 and n > 0 and parse(sub, p, out, depth+1) is not None
                if not ok:
                    del out[before:]
                    try: out.append((p, "str", sub.decode()))
                    except UnicodeDecodeError: out.append((p, "bytes", sub.hex()[:40]))
            else:
                return None
        return out
    except (IndexError, struct.error):
        return None

if __name__ == "__main__":
    if "--focal" in sys.argv:            # print focal (px) scaled to --width, if the clip header carries it
        video = sys.argv[1]; width = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 1920
        data = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-map", "0:1", "-c", "copy", "-f", "data", "-"],
                              capture_output=True).stdout
        n0 = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "1", "-show_entries", "packet=size",
                                 "-of", "csv=p=0", "-read_intervals", "%+#1", video], capture_output=True, text=True).stdout.split()[0])
        f = dict((p, v) for p, t, v in parse(data[:n0]) or [])
        rec_w = f.get("2.3.1"); foc = f.get("1.8.1")
        if f.get("1.1.1") == "dvtm_O4P.proto" and foc and rec_w: print(f"{foc * width / rec_w:.2f}")
        sys.exit()
    video, idx = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
    data = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-map", "0:1", "-c", "copy",
                           "-f", "data", "-"], capture_output=True).stdout
    sizes = [int(x.split(",")[1]) for x in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "1", "-show_entries", "packet=pts_time,size",
         "-of", "csv=p=0", video], capture_output=True, text=True).stdout.split()]
    off = sum(sizes[:idx]); pkt = data[off:off+sizes[idx]]
    for p, t, v in parse(pkt) or []:
        print(f"{p:20s} {t:7s} {v}")
