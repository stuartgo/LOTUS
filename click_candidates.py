"""Button-press extraction and bed/wake selection (polars).

One entry point for the pipeline:

    get_day_clicks(file_path, data) -> (morning, evening)

`file_path` is used only to locate the recording's .bin file (the source of the
button presses); `data` is the day's polars DataFrame. Returns clock hours,
-1.0 for anything that could not be identified.

Frames are polars throughout. The numeric core (`minute_activity`,
`select_clicks`) stays numpy on purpose: it is elementwise work on a contiguous
(T, C) block, which is what numpy is for, and keeping it array-based means it
never has to care whether the caller's frame is polars, pandas, or a bare array.

Selection method, for a methods section:

    For each recording day we considered every pair of button presses (b, w)
    with b in the bedtime window (18:00-06:00), w in the wake window
    (03:00-14:00), and an enclosed duration of MIN_TIB_H-MAX_TIB_H h
    (2-14 h by default); we kept the pair maximising the movement contrast
    between the enclosed interval and the 60-min periods flanking it
    (excluding 5 min either side of each press), requiring both flanks to be
    at least 1.5x more active than the interval; if no pair satisfied this,
    the night was recorded as missing.

Scoring pairs rather than individual presses is the fix for spurious night-time
presses. Scoring a press alone on post/pre activity diverges during sleep
(pre ~ 0), so a 03:00 press followed by any movement outscores the true wake
press. Scoring the INTERVAL makes the two boundaries constrain each other, and
enforces b < w and a plausible time in bed by construction.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Parameters (all physiologically interpretable -> all defensible in text)
# ---------------------------------------------------------------------------

DAY_BOUNDARY_HOUR = 15      # shifted day starts at 15:00; matches start_hour downstream
DEBOUNCE_MIN = 2.0          # presses closer than this are one physical press
BED_WINDOW_CLOCK = (18.0, 6.0)     # 18:00 -> 06:00  : bedtime candidates
WAKE_WINDOW_CLOCK = (3.0, 14.0)    # 03:00 -> 14:00  : wake candidates
                                   # (overlapping on purpose: 03-06 can be either)
MIN_TIB_H = 2.0             # shortest accepted time in bed
MAX_TIB_H = 14.0            # longest accepted time in bed
GUARD_MIN = 5.0             # excluded either side of a press
FLANK_MIN = 60.0            # length of the comparison periods outside the interval
MIN_CONTRAST = 1.5          # each flank must be >= this x the interval activity
MIN_WINDOW_MIN = 20         # a window needs this many valid minutes to be usable
FLOOR_FRAC = 0.05           # activity floor, as a fraction of the day's median

# Paths are read from the environment so the module is importable anywhere.
BIN_ROOT = Path(os.environ.get("SLEEPCLICK_BIN_ROOT", "./data/raw"))
CLICKS_DIR = Path(os.environ.get("SLEEPCLICK_CLICKS_DIR", "./data/clicks"))


def _to_shifted(clock_hour: float) -> float:
    """Clock hour -> hours since the start of the shifted day, in [0, 24)."""
    return (clock_hour - DAY_BOUNDARY_HOUR) % 24


BED_WINDOW = tuple(_to_shifted(h) for h in BED_WINDOW_CLOCK)    # (3, 15)
WAKE_WINDOW = tuple(_to_shifted(h) for h in WAKE_WINDOW_CLOCK)  # (12, 23)
assert BED_WINDOW[0] < BED_WINDOW[1] and WAKE_WINDOW[0] < WAKE_WINDOW[1], (
    "candidate windows must not wrap the shifted-day boundary"
)


# ---------------------------------------------------------------------------
# Raw press extraction
# ---------------------------------------------------------------------------
# Hex digits with bit 1 set: (int(c, 16) >> 1) & 1 == 1.
_BIT1_HEX = frozenset("2367abefABEF")
_BIT1_TABLE = bytes(1 if chr(b) in _BIT1_HEX else 0 for b in range(256))


def get_button_press_times(filepath) -> pl.DataFrame:
    """Parse button-press timestamps out of a GENEActiv .bin. -> frame ['time']."""
    HEADER_LINES = 59
    PAGE_STRIDE = 10
    TIMESTAMP_OFFSET = 3
    FREQ_OFFSET = 8
    DATA_OFFSET = 9
    CHUNK = 12

    with open(filepath, "rb") as f:
        lines = f.read().split(b"\n")

    button_times = []
    for i in range(HEADER_LINES, len(lines), PAGE_STRIDE):
        if i + DATA_OFFSET >= len(lines):
            break
        data = lines[i + DATA_OFFSET]
        flags = data[CHUNK - 1 :: CHUNK].translate(_BIT1_TABLE)
        if 1 not in flags:
            continue
        pressed = np.frombuffer(flags, dtype=np.uint8).nonzero()[0]
        freq = float(lines[i + FREQ_OFFSET].split(b":")[1])
        time_string = lines[i + TIMESTAMP_OFFSET].split(b":", maxsplit=1)[1][:23].decode()
        time = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S:%f")
        button_times.extend(time + timedelta(seconds=int(j) / freq) for j in pressed)

    if not button_times:
        return pl.DataFrame(schema={"time": pl.Datetime("ns")})
    return pl.DataFrame({"time": button_times}).with_columns(
        pl.col("time").cast(pl.Datetime("ns"))
    )


# ---------------------------------------------------------------------------
# Candidate preservation
# ---------------------------------------------------------------------------

_CAND_SCHEMA = {"day": pl.Date, "time": pl.Datetime("ns"), "kind": pl.Utf8}


def debounce_presses(presses: pl.DataFrame, gap_min: float = DEBOUNCE_MIN) -> pl.DataFrame:
    """Collapse runs of presses less than `gap_min` apart to their first time.

    The extractor flags every SAMPLE with bit 1 set, so one physical press of a
    fraction of a second yields several rows at the sampling rate. Without this,
    a single press is counted many times.
    """
    if presses.is_empty():
        return presses
    return (
        presses.sort("time")
        .with_columns(
            (pl.col("time").diff().dt.total_nanoseconds() / 6e10).alias("_gap_min")
        )
        .filter(pl.col("_gap_min").is_null() | (pl.col("_gap_min") >= gap_min))
        .drop("_gap_min")
    )


def clean_presses(presses: pl.DataFrame) -> pl.DataFrame:
    """Debounce, group into shifted days, tag candidate role(s). Drops nothing.

    Long format, one row per (press, role): columns [day, time, kind] with kind
    in {'bed', 'wake'}. A press in the 03:00-06:00 overlap appears twice; the
    pairing in `select_clicks` decides which role, if any, it takes.
    """
    if presses.is_empty():
        return pl.DataFrame(schema=_CAND_SCHEMA)

    shifted = pl.col("time") - pl.duration(hours=DAY_BOUNDARY_HOUR)
    base = debounce_presses(presses).with_columns(
        shifted.dt.date().alias("day"),
        (shifted.dt.hour()
         + shifted.dt.minute() / 60
         + shifted.dt.second() / 3600).alias("_h"),
    )

    parts = [
        base.filter((pl.col("_h") >= lo) & (pl.col("_h") < hi))
            .with_columns(pl.lit(kind, dtype=pl.Utf8).alias("kind"))
        for kind, (lo, hi) in (("bed", BED_WINDOW), ("wake", WAKE_WINDOW))
    ]
    parts = [p for p in parts if not p.is_empty()]
    if not parts:
        return pl.DataFrame(schema=_CAND_SCHEMA)

    return (
        pl.concat(parts)
        .select("day", "time", "kind")
        .sort("day", "time", "kind")
    )


def split_candidates(candidates: pl.DataFrame, start=None, end=None):
    """Candidate frame -> (bed_times, wake_times) as numpy datetime64[ns].

    Optionally restricted to [start, end]. Accepts either vocabulary
    ('bed'/'wake' or the original 'evening'/'morning'). Kept separate from
    `select_clicks` so the selector itself never touches a frame.
    """
    if candidates is None or candidates.is_empty():
        return np.empty(0, "datetime64[ns]"), np.empty(0, "datetime64[ns]")

    c = candidates.with_columns(
        pl.col("time").cast(pl.Datetime("ns")),
        pl.col("kind").replace({"evening": "bed", "morning": "wake"}),
    )
    if start is not None and end is not None:
        c = c.filter(pl.col("time").is_between(_as_dt(start), _as_dt(end)))

    return (
        c.filter(pl.col("kind") == "bed")["time"].to_numpy().astype("datetime64[ns]"),
        c.filter(pl.col("kind") == "wake")["time"].to_numpy().astype("datetime64[ns]"),
    )


def _as_dt(t) -> datetime:
    """numpy datetime64 / str / datetime -> python datetime, for polars compares."""
    if isinstance(t, datetime):
        return t
    return np.datetime64(t, "us").astype(datetime)


# ---------------------------------------------------------------------------
# Activity, computed on whatever the caller passes in
# ---------------------------------------------------------------------------

def minute_activity(signal, times):
    """Reduce a day of signal to a regular per-minute activity series.

    Returns (minutes, activity): `minutes` is a contiguous datetime64[m] axis,
    `activity` the mean |first difference| (across channels and samples) within
    each minute, nan where a minute has no data.

    |dsignal| ignores the gravity DC offset, so it measures motion rather than
    posture -- the same proxy the original filter_bad_clicks used. Upcast to
    float32 because day caches may hold float16.
    """
    signal = np.asarray(signal)
    times = np.asarray(times, dtype="datetime64[ns]")
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.shape[0] < 2:
        return np.empty(0, "datetime64[m]"), np.empty(0, np.float32)

    a = np.abs(np.diff(signal.astype(np.float32), axis=0)).mean(axis=1)
    minute = times[: len(a)].astype("datetime64[m]")
    idx = (minute - minute[0]).astype(np.int64)
    n = int(idx[-1]) + 1

    tot = np.bincount(idx, weights=a.astype(np.float64), minlength=n)
    cnt = np.bincount(idx, minlength=n)
    act = np.full(n, np.nan)
    np.divide(tot, cnt, out=act, where=cnt > 0)
    return minute[0] + np.arange(n, dtype="timedelta64[m]"), act.astype(np.float32)


def _level(act, i0, i1, min_minutes) -> float:
    """Median activity over minutes [i0, i1); nan if too little valid data.

    The median (not the mean) keeps a single burst -- a bathroom trip inside the
    sleep interval -- from disqualifying a real night.
    """
    i0, i1 = max(0, int(i0)), min(len(act), int(i1))
    if i1 <= i0:
        return float("nan")
    seg = act[i0:i1]
    if np.count_nonzero(np.isfinite(seg)) < min_minutes:
        return float("nan")
    return float(np.nanmedian(seg))


# ---------------------------------------------------------------------------
# Selection: pure function, arrays in, tuple out
# ---------------------------------------------------------------------------

def select_clicks(
    signal,
    times,
    bed_times,
    wake_times,
    guard_min: float = GUARD_MIN,
    flank_min: float = FLANK_MIN,
    min_tib_h: float = MIN_TIB_H,
    max_tib_h: float = MAX_TIB_H,
    min_contrast: float = MIN_CONTRAST,
    min_window_min: int = MIN_WINDOW_MIN,
    floor_frac: float = FLOOR_FRAC,
):
    """Choose the (bedtime, wake time) press pair best supported by movement.

    Reads nothing; the caller supplies the day's signal and its candidates.

    Args:
        signal      : (T, C) array for one day, e.g. columns x, y, z.
        times       : (T,) sorted datetime64 array aligned with `signal` rows.
        bed_times   : candidate bedtime press timestamps.
        wake_times  : candidate wake press timestamps.
        guard_min   : minutes excluded either side of a press, so the movement of
                      putting the device down / picking it up is in neither the
                      interval nor the flank.
        flank_min   : length of the comparison periods outside the interval.
        min_tib_h   : shortest accepted enclosed duration.
        max_tib_h   : longest accepted enclosed duration.
        min_contrast: each flank must be >= this x the interval activity.
        min_window_min : a window needs this many valid minutes to be usable.
        floor_frac  : activity floor as a fraction of the day's median activity;
                      regularises the contrasts so they stay finite and
                      comparable across participants and devices.

    Returns:
        (bedtime, waketime, reason, diagnostics), where bedtime/waketime are
        numpy datetime64[ns] or None. reason in {'ok', 'no_candidates',
        'no_signal', 'none_passed'}. diagnostics is a list of dicts, one per
        pair considered, cheap to log for auditing rejections.
    """
    bed_times = sorted(np.asarray(bed_times, dtype="datetime64[ns]").ravel().tolist())
    wake_times = sorted(np.asarray(wake_times, dtype="datetime64[ns]").ravel().tolist())
    bed_times = [np.datetime64(t, "ns") for t in bed_times]
    wake_times = [np.datetime64(t, "ns") for t in wake_times]
    if not bed_times or not wake_times:
        return None, None, "no_candidates", []

    minutes, act = minute_activity(signal, times)
    if minutes.size == 0 or not np.isfinite(act).any():
        return None, None, "no_signal", []

    day_med = float(np.nanmedian(act))
    floor = floor_frac * day_med if day_med > 0 else 1e-12
    g, f = int(round(guard_min)), int(round(flank_min))

    def to_idx(t):
        return int((np.datetime64(t, "m") - minutes[0]) / np.timedelta64(1, "m"))

    diagnostics, best = [], None
    for b in bed_times:
        ib = to_idx(b)
        for w in wake_times:
            iw = to_idx(w)
            tib_h = (iw - ib) / 60.0
            rec = {"bedtime": b, "waketime": w, "tib_h": tib_h}

            # b < w and a plausible time in bed come for free here.
            if not (min_tib_h <= tib_h <= max_tib_h):
                diagnostics.append({**rec, "valid": False, "why": "duration"})
                continue

            inside = _level(act, ib + g, iw - g, min_window_min)
            pre = _level(act, ib - g - f, ib - g, min_window_min)
            post = _level(act, iw + g, iw + g + f, min_window_min)
            c_bed = (pre + floor) / (inside + floor)
            c_wake = (post + floor) / (inside + floor)
            rec.update(inside=inside, pre=pre, post=post,
                       contrast_bed=c_bed, contrast_wake=c_wake)

            if not np.isfinite([inside, pre, post]).all():
                diagnostics.append({**rec, "valid": False, "why": "no_data"})
                continue
            if c_bed < min_contrast or c_wake < min_contrast:
                diagnostics.append({**rec, "valid": False, "why": "contrast"})
                continue

            score = float(np.log(c_bed) + np.log(c_wake))
            diagnostics.append({**rec, "valid": True, "why": "ok", "score": score})
            if best is None or score > best["score"]:
                best = {**rec, "score": score}

    if best is None:
        return None, None, "none_passed", diagnostics
    return best["bedtime"], best["waketime"], "ok", diagnostics


# ---------------------------------------------------------------------------
# One-call entry point: file path + the day's data -> (morning, evening)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _bin_index(root_str: str) -> dict:
    """stem -> .bin path for the whole recording tree, walked once."""
    return {p.stem: p for p in Path(root_str).rglob("*.bin")}


def _find_bin_file(file_path, bin_root):
    """Resolve whatever identifies a recording to its .bin.

    Accepts the .bin path itself, a day-parquet path
    ('{pid}_{stem}_day_{n}.parquet'), or a bare stem.
    """
    p = Path(file_path)
    if p.suffix.lower() == ".bin" and p.exists():
        return p

    stem = p.stem
    stems = [stem]
    if "_day" in stem:                     # '{pid}_{stem}_day_{n}' -> '{pid}_{stem}'
        stems.append(stem.split("_day")[0])
    stems += [s.split("_", 1)[1] for s in list(stems) if "_" in s]  # drop pid prefix
    stems = list(dict.fromkeys(stems))

    for s in stems:                        # sibling of the given path first
        cand = p.parent / f"{s}.bin"
        if cand.exists():
            return cand
    index = _bin_index(str(bin_root))      # then the recording tree
    for s in stems:
        if s in index:
            return index[s]
    return None


@lru_cache(maxsize=64)
def _candidates_from_bin(path_str: str, mtime: float) -> pl.DataFrame:
    """Extract, debounce and tag one recording's presses. -> [day, time, kind].

    Every day of a recording needs the same press list and days are processed
    independently, so without this the .bin is re-parsed once per day. `mtime`
    is in the key so a replaced file is not served stale.
    """
    return clean_presses(get_button_press_times(path_str))


def _day_times_and_signal(data, signal_cols, time_col):
    """Pull (times, signal) out of a polars frame as numpy arrays."""
    times = data[time_col].to_numpy().astype("datetime64[ns]")
    signal = data.select(list(signal_cols)).to_numpy()   # one contiguous (T, C) block
    return times, signal


def _clock_hours(t) -> float:
    """datetime64 / datetime -> clock hour as a float."""
    d = _as_dt(t)
    return d.hour + d.minute / 60 + d.second / 3600


def get_day_clicks(
    file_path,
    data,
    bin_root=BIN_ROOT,
    signal_cols=("x", "y", "z"),
    time_col="time",
    missing=-1.0,
    details=False,
    **select_kw,
):
    """Return (morning, evening) click times for one day, as clock hours.

    The only function you need to call.

    Args:
        file_path   : anything identifying the recording -- the .bin path, the
                      day-parquet path, or a bare stem. Used only to find the
                      .bin the presses are read from.
        data        : the day's polars DataFrame. Needs the signal columns and
                      a time column.
        bin_root    : directory tree searched for the recording's .bin.
        signal_cols : columns used for the movement signal.
        time_col    : name of the timestamp column.
        missing     : value returned for a click that could not be identified.
        details     : if True, return (morning, evening, info) with reason,
                      bedtime, waketime, tib_h, contrasts, n_candidates and the
                      per-pair diagnostics.
        **select_kw : passed to `select_clicks` (guard_min, flank_min,
                      min_tib_h, max_tib_h, min_contrast, ...).

    Returns:
        (morning, evening) as floats: clock hour of the wake press and of the
        bed press, `missing` where none was identified. Same [morning, evening]
        order and same -1 convention as the original pipeline.
    """
    def _out(reason, bedtime=None, waketime=None, diag=(), n_cand=0, **extra):
        morning = _clock_hours(waketime) if waketime is not None else missing
        evening = _clock_hours(bedtime) if bedtime is not None else missing
        if not details:
            return morning, evening
        info = {"reason": reason, "bedtime": bedtime, "waketime": waketime,
                "n_candidates": n_cand, "diagnostics": list(diag)}
        info.update(extra)
        return morning, evening, info

    times, signal = _day_times_and_signal(data, signal_cols, time_col)

    bin_path = _find_bin_file(file_path, bin_root)
    if bin_path is None:
        print(f"[warn] no .bin found for {file_path}")
        return _out("no_candidates")
    try:
        cands = _candidates_from_bin(str(bin_path), bin_path.stat().st_mtime)
    except Exception as e:
        print(f"[warn] could not read presses from {bin_path}: {e}")
        return _out("no_candidates")

    bed, wake = split_candidates(cands, times[0], times[-1])
    if not len(bed) or not len(wake):
        return _out("no_candidates")

    bedtime, waketime, reason, diag = select_clicks(
        signal, times, bed, wake, **select_kw
    )
    # Diagnostics of the pair that was actually selected (not merely the first valid one).
    best = next((d for d in diag if d.get("valid") and d["bedtime"] == bedtime
                 and d["waketime"] == waketime), {}) if reason == "ok" else {}
    return _out(reason, bedtime, waketime, diag, len(bed) + len(wake),
                tib_h=best.get("tib_h", float("nan")),
                contrast_bed=best.get("contrast_bed", float("nan")),
                contrast_wake=best.get("contrast_wake", float("nan")))


# ---------------------------------------------------------------------------
# Per-file driver: extract raw presses -> candidate parquet (no selection here)
# ---------------------------------------------------------------------------

def presses_from_bin(file_path) -> pl.DataFrame:
    """One recording's candidate presses straight from its .bin -> [day, time, kind].

    The parquet-free equivalent of reading a candidate parquet: identical
    contents (debounced, tagged with the bed/wake windows), just recomputed.
    Module level and exception-free so it can be handed to a Pool.
    """
    try:
        return clean_presses(get_button_press_times(file_path))
    except Exception as e:
        print(f"[warn] could not read presses from {file_path}: {e}")
        return pl.DataFrame(schema=_CAND_SCHEMA)


def process_file(file):
    # New filename so stale single-pick caches cannot be read by accident.
    out_path = CLICKS_DIR / f"{file.stem}_button_press_candidates_v2.parquet"
    if out_path.exists():
        return
    presses_from_bin(file).write_parquet(out_path)


if __name__ == "__main__":
    CLICKS_DIR.mkdir(parents=True, exist_ok=True)
    all_files = list(BIN_ROOT.rglob("*.bin"))
    with Pool(processes=min(48, os.cpu_count())) as pool:
        list(tqdm(pool.imap_unordered(process_file, all_files), total=len(all_files)))