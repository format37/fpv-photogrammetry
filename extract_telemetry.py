#!/usr/bin/env python3
"""Extract a time-aligned flight track from an ArduPilot DataFlash log (.BIN).

Usage:
    extract_telemetry.py LOG.BIN OUTDIR [--arm N] [--gps I] [--lag S] [--prefix P] [--no-terrain] [--list]

Outputs written to OUTDIR (each name optionally prefixed with --prefix):
    gps_5hz.csv      one row per fix of the selected GPS receiver. Attitude and heights are
                     interpolated to the fix epoch. The name assumes the usual 5 Hz GPS rate;
                     the actual rate is reported in meta.json.
    pose_25hz.csv    EKF position (POS) with attitude (ATT) interpolated to the POS timestamps,
                     at the logging rate of POS (25 Hz on a default ArduPlane setup).
    terrain_1hz.csv  terrain height under the vehicle from the TERR message, if the log has it.
    meta.json        armed windows, arm/disarm/mode events, launch/touchdown estimates, home,
                     GPS receiver selection, clock fit and a short flight summary.

What the script does:
1. Armed windows. Arm and disarm are taken from ARM messages and from the "Throttle armed" /
   "Throttle disarmed" status texts. The text matters because the first arm of a log is often
   missing as an ARM message: with LOG_DISARMED=0 the log file is opened by the arm itself.
   The longest armed window is used unless --arm N picks another one (1-based, see --list).
2. GPS receiver. When the log carries the GPS "U" (in use) flag, the receiver that ArduPilot
   used most during the window is selected. Otherwise the one with the lower median horizontal
   accuracy (GPA.HAcc) is selected. --gps overrides the choice.
3. Clock. The flight controller's microsecond clock (TimeUS) is not exactly true time; drift of
   about 0.1 % has been seen on H7 boards. A straight line is fitted from TimeUS to the GPS
   fix epochs (GPS week * 604800 + time of week) of the selected receiver over the whole log.
   Outliers over 0.5 s are rejected and the line is refitted.
4. Lag. The fit maps TimeUS to the epoch of the fix that was being logged at that moment.
   Samples stamped by the flight controller itself (ATT, POS, BARO, events) describe "now",
   which is later than that fix epoch by the receiver-to-log latency. That latency (--lag) is
   added. The default 0.12 s is the lag ArduPilot itself assumes for u-blox M8/M9/M10/F9
   receivers (AP_GPS_UBLOX::get_lag). GPS rows keep their exact fix epoch.
5. Time columns. t_since_arm_s is true seconds since the selected arm event. utc is GPS time
   minus the GPS-UTC leap seconds.

Columns and conventions:
    lat, lon         WGS84 degrees.
    alt_gps_msl      GPS altitude above mean sea level (geoid height, not ellipsoidal height).
    alt_msl          EKF altitude above mean sea level.
    alt_rel          EKF height above home.
    baro_alt         barometric altitude relative to the ground-level calibration at boot.
    gps_speed        ground speed in m/s.
    course           degrees from true north.
    vz_down          vertical speed in m/s, positive down (NED).
    roll, pitch, yaw degrees, ArduPilot body frame in NED, Euler ZYX. Roll is positive right
                     wing down, pitch positive nose up, yaw 0-360 from true north.
    hacc, vacc       receiver accuracy estimates in metres.
    fix              GPS status: 3 = 3D, 4 = 3D + SBAS/DGPS, 5/6 = RTK float/fixed.
    mode             flight mode name.
    terrain_msl      raw terrain database height (TERR.TerrH minus the reference offset ROfs).
                     ArduPilot logs TerrH with an offset added so that the terrain matches the
                     vehicle altitude at arm. The logged value is kept in terrain_logged_msl.

Launch and touchdown in meta.json are heuristics:
    launch    = peak IMU acceleration within +-3 s of the first time GPS ground speed
                exceeds 3 m/s.
    touchdown = first IMU acceleration above 30 m/s^2 from 5 s before the last time ground
                speed exceeds 3 m/s, up to disarm. If there is none, it falls back to that
                last fast sample.

Dependencies: pymavlink, numpy, pandas.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from pymavlink import mavutil

# Fields read from each DataFlash message type (missing fields become NaN).
WANTED = {
    'GPS': ['TimeUS', 'I', 'Status', 'GMS', 'GWk', 'NSats', 'HDop', 'Lat', 'Lng', 'Alt',
            'Spd', 'GCrs', 'VZ', 'U'],
    'GPA': ['TimeUS', 'I', 'HAcc', 'VAcc'],
    'ATT': ['TimeUS', 'Roll', 'Pitch', 'Yaw'],
    'POS': ['TimeUS', 'Lat', 'Lng', 'Alt', 'RelHomeAlt'],
    'BARO': ['TimeUS', 'I', 'Alt'],
    'MODE': ['TimeUS', 'Mode', 'ModeNum'],
    'ARM': ['TimeUS', 'ArmState'],
    'MSG': ['TimeUS', 'Message'],
    'IMU': ['TimeUS', 'I', 'AccX', 'AccY', 'AccZ'],
    'TERR': ['TimeUS', 'Status', 'Lat', 'Lng', 'TerrH', 'CHeight', 'ROfs'],
    'ORGN': ['TimeUS', 'Type', 'Lat', 'Lng', 'Alt'],
}

GPS_EPOCH_UNIX = 315964800          # 1980-01-06T00:00:00Z as a Unix timestamp
SECONDS_PER_WEEK = 604800
# GPS-UTC offset: (first UTC day it applies, leap seconds). Newest first.
LEAP_SECONDS = [(dt.date(2017, 1, 1), 18), (dt.date(2015, 7, 1), 17), (dt.date(2012, 7, 1), 16),
                (dt.date(2009, 1, 1), 15), (dt.date(2006, 1, 1), 14)]
MOVING_SPEED = 3.0                  # m/s, ground speed taken as "flying"
TOUCHDOWN_ACC = 30.0                # m/s^2 (~3 g), first impact after the last fast sample
FIX_3D = 3

ARM_TEXT = re.compile(r'^(throttle armed|arming motors)', re.I)
DISARM_TEXT = re.compile(r'^(throttle disarmed|disarming motors)', re.I)
FIRMWARE_TEXT = re.compile(r'^(Ardu\w+|Blimp|AntennaTracker) V\d')


def load_log(path):
    """Read the needed message types into one DataFrame per type."""
    rows = defaultdict(list)
    log = mavutil.mavlink_connection(path)
    types = list(WANTED)
    while True:
        msg = log.recv_match(type=types)
        if msg is None:
            break
        t = msg.get_type()
        rows[t].append(tuple(getattr(msg, f, None) for f in WANTED[t]))
    d = {t: pd.DataFrame(rows[t], columns=WANTED[t]) for t in WANTED}
    for t in ('GPS', 'GPA', 'BARO', 'IMU'):     # single-instance logs may lack the I field
        d[t]['I'] = d[t]['I'].fillna(0).astype(int)
    return d


def firmware_info(msgs):
    """Return (firmware string, board name) from the boot banner texts."""
    fw = board = None
    for i, m in enumerate(msgs.Message.astype(str)):
        if fw is None and FIRMWARE_TEXT.match(m):
            fw = m
            # The banner continues with "ChibiOS: ..." and then "<board> <MCU unique id>".
            for nxt in msgs.Message.astype(str).iloc[i + 1:i + 4]:
                if not nxt.startswith(('ChibiOS', 'NuttX', 'Param')) and ' ' in nxt:
                    board = nxt.split()[0]     # drop the MCU unique id
                    break
    return fw, board


def mode_names(firmware):
    """Mode number -> name table for the vehicle type."""
    fw = firmware or ''
    if fw.startswith('ArduPlane'):
        return mavutil.mode_mapping_apm
    if fw.startswith('ArduCopter'):
        return mavutil.mode_mapping_acm
    if fw.startswith('ArduRover'):
        return mavutil.mode_mapping_rover
    if fw.startswith('ArduSub'):
        return mavutil.mode_mapping_sub
    return {}


def arm_events(d):
    """Arm/disarm events as a time-sorted list of (TimeUS, armed), duplicates merged."""
    raw = [(int(r.TimeUS), bool(r.ArmState)) for r in d['ARM'].itertuples()]
    for r in d['MSG'].itertuples():
        text = str(r.Message)
        if ARM_TEXT.match(text):
            raw.append((int(r.TimeUS), True))
        elif DISARM_TEXT.match(text):
            raw.append((int(r.TimeUS), False))
    raw.sort()
    events = []
    for t, armed in raw:                 # ARM message and status text describe the same event
        if not events or events[-1][1] != armed:
            events.append((t, armed))
    return events


def armed_windows(events, log_end_us):
    """List of (arm TimeUS, disarm TimeUS) pairs; an unterminated arm ends at the log end."""
    windows = []
    for i, (t, armed) in enumerate(events):
        if not armed:
            continue
        end = next((t2 for t2, a2 in events[i + 1:] if not a2), log_end_us)
        windows.append((t, end))
    return windows


def select_gps(gps, gpa, t0, t1, force=None):
    """Pick the GPS instance to use; returns (instance, per-instance statistics)."""
    stats = []
    for inst, g in gps.groupby('I'):
        w = g[(g.TimeUS >= t0) & (g.TimeUS <= t1)]
        if w.empty:
            continue
        a = gpa[(gpa.I == inst) & (gpa.TimeUS >= t0) & (gpa.TimeUS <= t1)]
        u = pd.to_numeric(w.U, errors='coerce')
        stats.append(dict(
            instance=int(inst), samples=int(len(w)),
            fix3d_share=round(float((w.Status >= FIX_3D).mean()), 3),
            in_use_share=round(float(u.mean()), 3) if u.notna().all() else None,
            sats_median=float(w.NSats.median()),
            hdop_median=round(float(w.HDop.median()), 2),
            hacc_median=round(float(a.HAcc.median()), 3) if len(a) else None,
            vacc_median=round(float(a.VAcc.median()), 3) if len(a) else None))
    if not stats:
        sys.exit('no GPS data inside the armed window')
    if force is not None:
        if force not in [s['instance'] for s in stats]:
            sys.exit(f'GPS instance {force} has no data inside the armed window')
        return force, stats, 'forced with --gps'
    if all(s['in_use_share'] is not None for s in stats):
        best = max(stats, key=lambda s: (s['in_use_share'], -(s['hacc_median'] or np.inf)))
        return best['instance'], stats, 'receiver ArduPilot used most (GPS.U flag)'
    best = min(stats, key=lambda s: (s['hacc_median'] if s['hacc_median'] is not None else np.inf,
                                     -s['sats_median']))
    return best['instance'], stats, 'lowest median horizontal accuracy (GPA.HAcc)'


def fit_clock(fixes):
    """Fit GPS seconds = slope * log seconds + offset; returns slope, offset, diagnostics."""
    x = fixes.TimeUS.values / 1e6
    tg = fixes.GWk.values * float(SECONDS_PER_WEEK) + fixes.GMS.values / 1000.0
    ref = tg[0]                          # fit relative to the first epoch for conditioning
    keep = np.ones(len(x), bool)
    for _ in range(2):
        slope, off = np.polyfit(x[keep], tg[keep] - ref, 1)
        res = tg - ref - (slope * x + off)
        keep = np.abs(res) < 0.5
    res = res[keep]
    diag = dict(gps_s_per_log_s=float(slope),
                fc_clock_fast_pct=round((1 / slope - 1) * 100, 4),
                resid_ms_rms=round(float(np.sqrt(np.mean(res ** 2)) * 1e3), 1),
                resid_ms_max=round(float(np.abs(res).max() * 1e3), 1),
                fixes_used=int(keep.sum()), fixes_rejected=int((~keep).sum()))
    return slope, off + ref, diag


def leap_seconds(gps_seconds):
    day = (dt.datetime(1980, 1, 6) + dt.timedelta(seconds=float(gps_seconds))).date()
    return next((n for start, n in LEAP_SECONDS if day >= start), 13)


def utc_strings(gps_seconds, leap):
    """ISO 8601 UTC strings (millisecond precision) for GPS-time seconds."""
    unix_us = np.round((np.asarray(gps_seconds, float) - leap + GPS_EPOCH_UNIX) * 1e6).astype('int64')
    ts = pd.to_datetime(unix_us, unit='us', utc=True)
    return (ts.strftime('%Y-%m-%dT%H:%M:%S.%f').str[:-3] + 'Z').tolist()


def interp_angle_deg(tq, ts, deg):
    """Interpolate a 0-360 degree heading without wrapping through 180."""
    unwrapped = np.degrees(np.unwrap(np.radians(np.asarray(deg, float))))
    return np.mod(np.interp(tq, ts, unwrapped), 360)


def detect_launch_touchdown(gps_w, imu, arm_us, disarm_us):
    """Heuristic launch and touchdown TimeUS (see module docstring); None when not found."""
    fast = gps_w[gps_w.Spd > MOVING_SPEED]
    if fast.empty or imu.empty:
        return None, None
    acc = np.sqrt(imu.AccX ** 2 + imu.AccY ** 2 + imu.AccZ ** 2)
    first_fast, last_fast = int(fast.TimeUS.min()), int(fast.TimeUS.max())
    lo, hi = max(arm_us, first_fast - 3_000_000), min(disarm_us, first_fast + 3_000_000)
    win = (imu.TimeUS >= lo) & (imu.TimeUS <= hi)
    launch = None
    if win.any():
        i = acc[win].idxmax()
        launch = (int(imu.TimeUS[i]), float(acc[i]))
    win = (imu.TimeUS >= last_fast - 5_000_000) & (imu.TimeUS <= disarm_us) & (acc > TOUCHDOWN_ACC)
    if win.any():
        i = acc[win].index[0]
        touchdown = (int(imu.TimeUS[i]), float(acc[i]), f'first IMU accel > {TOUCHDOWN_ACC:g} m/s^2')
    else:
        touchdown = (last_fast, None, f'last GPS ground speed > {MOVING_SPEED:g} m/s')
    return launch, touchdown


def path_length_m(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    r = 6371008.8
    return float(np.hypot(np.diff(la) * r, np.diff(lo) * r * np.cos(la[:-1])).sum())


def main():
    ap = argparse.ArgumentParser(description='Extract a time-aligned flight track from an '
                                 'ArduPilot DataFlash log. See the module docstring for details.')
    ap.add_argument('log', help='ArduPilot DataFlash log (.BIN)')
    ap.add_argument('outdir', help='output directory (created if missing)')
    ap.add_argument('--arm', type=int, help='1-based armed window to export (default: longest)')
    ap.add_argument('--gps', type=int, help='force GPS instance (0-based, as in the GPS.I field)')
    ap.add_argument('--lag', type=float, default=0.12,
                    help='GPS fix-to-log latency in seconds (default 0.12, u-blox M8/M9/M10/F9)')
    ap.add_argument('--prefix', default='', help='prefix for the output file names')
    ap.add_argument('--no-terrain', action='store_true', help='do not write terrain_1hz.csv')
    ap.add_argument('--list', action='store_true', help='list armed windows and exit')
    args = ap.parse_args()

    d = load_log(args.log)
    fw, board = firmware_info(d['MSG'])
    names = mode_names(fw)

    log_end = int(max(d['ATT'].TimeUS.max(), d['POS'].TimeUS.max()))
    events = arm_events(d)
    windows = armed_windows(events, log_end)
    if not windows:
        sys.exit('no arm event found in the log')
    print(f'{os.path.basename(args.log)}: {fw or "unknown firmware"} on {board or "unknown board"}')
    for i, (a, b) in enumerate(windows, 1):
        print(f'  armed window {i}: log {a / 1e6:.3f}-{b / 1e6:.3f} s ({(b - a) / 1e6:.1f} s)')
    if args.list:
        return
    if args.arm is not None:
        if not 1 <= args.arm <= len(windows):
            sys.exit(f'--arm must be between 1 and {len(windows)}')
        win_idx = args.arm - 1
    else:
        win_idx = max(range(len(windows)), key=lambda i: windows[i][1] - windows[i][0])
    arm_us, disarm_us = windows[win_idx]

    # GPS receiver and clock fit
    gps, gpa = d['GPS'], d['GPA']
    inst, gps_stats, gps_reason = select_gps(gps, gpa, arm_us, disarm_us, args.gps)
    fixes = gps[(gps.I == inst) & (gps.Status >= FIX_3D) & (gps.GWk > 0)].copy()
    if len(fixes) < 10:
        sys.exit(f'GPS instance {inst} has too few 3D fixes for a clock fit')
    slope, offset, clock = fit_clock(fixes)
    clock['lag_s'] = args.lag

    def true_time(time_us):
        """GPS seconds of samples stamped by the flight controller."""
        return slope * np.asarray(time_us, float) / 1e6 + offset + args.lag

    t_arm, t_disarm = float(true_time(arm_us)), float(true_time(disarm_us))
    leap = leap_seconds(t_arm)
    fixes['tg'] = fixes.GWk * float(SECONDS_PER_WEEK) + fixes.GMS / 1000.0

    att, pos = d['ATT'], d['POS']
    baro = d['BARO'][d['BARO'].I == 0]
    att_t, pos_t, baro_t = true_time(att.TimeUS), true_time(pos.TimeUS), true_time(baro.TimeUS)
    modes = d['MODE']
    mode_num = modes.ModeNum.where(modes.ModeNum.notna(), modes.Mode)
    mode_t = true_time(modes.TimeUS)
    mode_label = [names.get(int(n), str(int(n))) for n in mode_num]

    def mode_at(tq):
        idx = np.searchsorted(mode_t, tq, side='right') - 1
        return [mode_label[j] if j >= 0 else '' for j in idx]

    os.makedirs(args.outdir, exist_ok=True)
    out = lambda name: os.path.join(args.outdir, args.prefix + name)

    # GPS rows of the selected receiver inside the window, at their exact fix epochs
    w = fixes[(fixes.tg >= t_arm) & (fixes.tg <= t_disarm)]
    acc = gpa[gpa.I == inst][['TimeUS', 'HAcc', 'VAcc']].drop_duplicates('TimeUS')
    w = w.merge(acc, on='TimeUS', how='left')
    tq = w.tg.values
    gps_csv = pd.DataFrame({
        't_since_arm_s': np.round(tq - t_arm, 3), 'utc': utc_strings(tq, leap),
        'lat': w.Lat.values, 'lon': w.Lng.values, 'alt_gps_msl': w.Alt.values,
        'alt_rel': np.round(np.interp(tq, pos_t, pos.RelHomeAlt), 2),
        'baro_alt': np.round(np.interp(tq, baro_t, baro.Alt), 2),
        'gps_speed': w.Spd.values, 'course': w.GCrs.values, 'vz_down': w.VZ.values,
        'roll': np.round(np.interp(tq, att_t, att.Roll), 2),
        'pitch': np.round(np.interp(tq, att_t, att.Pitch), 2),
        'yaw': np.round(interp_angle_deg(tq, att_t, att.Yaw), 2),
        'sats': w.NSats.values, 'hdop': w.HDop.values, 'hacc': w.HAcc.values,
        'vacc': w.VAcc.values, 'fix': w.Status.values, 'mode': mode_at(tq)})
    gps_csv.to_csv(out('gps_5hz.csv'), index=False)

    # EKF pose at the POS rate
    m = (pos_t >= t_arm) & (pos_t <= t_disarm)
    pt, pw = pos_t[m], pos[m]
    pose_csv = pd.DataFrame({
        't_since_arm_s': np.round(pt - t_arm, 3), 'utc': utc_strings(pt, leap),
        'lat': pw.Lat.values, 'lon': pw.Lng.values,
        'alt_msl': np.round(pw.Alt.values, 2), 'alt_rel': np.round(pw.RelHomeAlt.values, 2),
        'roll': np.round(np.interp(pt, att_t, att.Roll), 2),
        'pitch': np.round(np.interp(pt, att_t, att.Pitch), 2),
        'yaw': np.round(interp_angle_deg(pt, att_t, att.Yaw), 2), 'mode': mode_at(pt)})
    pose_csv.to_csv(out('pose_25hz.csv'), index=False)

    # Terrain under the vehicle
    terr = d['TERR']
    terrain_info = None
    if not args.no_terrain and not terr.empty:
        tt = true_time(terr.TimeUS)
        m = (tt >= t_arm) & (tt <= t_disarm)
        tw, ttw = terr[m], tt[m]
        rofs = tw.ROfs.fillna(0).values
        ter_csv = pd.DataFrame({
            't_since_arm_s': np.round(ttw - t_arm, 3), 'utc': utc_strings(ttw, leap),
            'lat': tw.Lat.values, 'lon': tw.Lng.values,
            'terrain_msl': np.round(tw.TerrH.values - rofs, 2),
            'terrain_logged_msl': np.round(tw.TerrH.values, 2), 'ref_offset': np.round(rofs, 2),
            'height_above_terrain': np.round(tw.CHeight.values, 2), 'status': tw.Status.values})
        ter_csv.to_csv(out('terrain_1hz.csv'), index=False)
        terrain_info = dict(file=os.path.basename(out('terrain_1hz.csv')), rows=len(ter_csv),
                            ref_offset_m=sorted(set(np.round(rofs, 2).tolist())))

    # Home: ArduPilot resets home at arm, so take the EKF position at the arm event.
    p0 = pos[pos.TimeUS >= arm_us].iloc[0]
    home = dict(lat=float(p0.Lat), lon=float(p0.Lng), alt_msl=round(float(p0.Alt - p0.RelHomeAlt), 2),
                source='EKF position at arm (home is reset at arm); alt = POS.Alt - POS.RelHomeAlt')
    orgn = d['ORGN']
    origin = orgn[(orgn.Type == 0) & (orgn.TimeUS <= disarm_us)]
    ekf_origin = None
    if not origin.empty:
        o = origin.iloc[-1]
        ekf_origin = dict(lat=float(o.Lat), lon=float(o.Lng), alt_msl=float(o.Alt))

    # Events
    imu = d['IMU'][d['IMU'].I == 0].reset_index(drop=True)
    gps_win = gps[(gps.I == inst) & (gps.TimeUS >= arm_us) & (gps.TimeUS <= disarm_us)]
    launch, touchdown = detect_launch_touchdown(gps_win, imu, arm_us, disarm_us)
    ev = []
    for n, (t, armed) in enumerate(events):
        selected = ' (selected window)' if t in (arm_us, disarm_us) else ''
        ev.append((t, ('ARM' if armed else 'DISARM') + selected))
    for t, label in zip(modes.TimeUS, mode_label):
        ev.append((int(t), f'mode {label}'))
    if launch:
        ev.append((launch[0], f'launch: peak accel {launch[1] / 9.80665:.1f} g (heuristic)'))
    if touchdown:
        ev.append((touchdown[0], f'touchdown: {touchdown[2]} (heuristic)'))
    ev.sort()
    events_json = [dict(event=label, log_time_s=round(t / 1e6, 3),
                        utc=utc_strings([true_time(t)], leap)[0],
                        t_since_arm_s=round(float(true_time(t)) - t_arm, 3)) for t, label in ev]

    # Short summary of the selected window
    summary = dict(armed_s=round(t_disarm - t_arm, 1),
                   max_height_above_home_m=round(float(pw.RelHomeAlt.max()), 1))
    if launch and touchdown:
        t_l, t_t = float(true_time(launch[0])), float(true_time(touchdown[0]))
        fl = fixes[(fixes.tg >= t_l) & (fixes.tg <= t_t)]
        summary.update(airborne_s=round(t_t - t_l, 1),
                       distance_flown_km=round(path_length_m(fl.Lat.values, fl.Lng.values) / 1000, 2),
                       ground_speed_median_ms=round(float(fl.Spd.median()), 1),
                       ground_speed_max_ms=round(float(fl.Spd.max()), 1))
    span = tq[-1] - tq[0] if len(tq) > 1 else 0
    meta = dict(
        log=os.path.basename(args.log), firmware=fw, board=board,
        armed_windows=[dict(index=i, log_start_s=round(a / 1e6, 3), log_end_s=round(b / 1e6, 3),
                            duration_log_s=round((b - a) / 1e6, 1))
                       for i, (a, b) in enumerate(windows, 1)],
        selected_window=win_idx + 1,
        arm_utc=utc_strings([t_arm], leap)[0], disarm_utc=utc_strings([t_disarm], leap)[0],
        gps=dict(selected_instance=int(inst), reason=gps_reason,
                 rate_hz=round((len(tq) - 1) / span, 2) if span else None, instances=gps_stats),
        clock_fit=clock, utc_leap_seconds=leap, home=home, ekf_origin=ekf_origin,
        terrain=terrain_info, summary=summary, events=events_json,
        files=[os.path.basename(out(f)) for f in ('gps_5hz.csv', 'pose_25hz.csv')])
    with open(out('meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)

    print(f'selected window {win_idx + 1}: ARM {meta["arm_utc"]} -> DISARM {meta["disarm_utc"]} '
          f'({summary["armed_s"]} s)')
    print(f'GPS instance {inst} ({gps_reason}); FC clock fast by {clock["fc_clock_fast_pct"]} %, '
          f'fit residual {clock["resid_ms_rms"]} ms rms')
    print(f'wrote {len(gps_csv)} GPS rows, {len(pose_csv)} pose rows'
          + (f', {terrain_info["rows"]} terrain rows' if terrain_info else '') + f' to {args.outdir}')


if __name__ == '__main__':
    main()
