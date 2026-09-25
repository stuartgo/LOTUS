"""Data pipeline: raw GENEActiv .bin recordings -> per-day tensors -> CV folds.

Steps
    1. ``process_single_file``      : read a .bin with wristpy, run non-wear
                                       detection, cut the recording into
                                       15:00-to-15:00 days and keep only days
                                       with full wear and no missing samples.
    2. ``do_feature_extraction_single``: per day, stack the x/y/z signal and
                                       derive the (wake, bed) button-press
                                       labels with ``click_candidates``.
    3. ``build_xy``                  : concatenate everything, cache to disk,
                                       and write down-sampled copies
                                       (1 Hz, 1 min, 5 min).
    4. ``create_folds``              : participant-level K-fold split with
                                       train-only normalisation.

Paths are configured with environment variables (see README):
    SLEEPCLICK_DATA_DIR   working directory for intermediate files
    SLEEPCLICK_BIN_ROOT   directory tree containing the raw .bin files
"""

import os

# Must be set before numpy / torch / polars are imported to take effect. Each
# worker process in the Pool should be single-threaded.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import hashlib
import logging
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from click_candidates import BIN_ROOT, get_day_clicks

logging.getLogger("wristpy").setLevel(logging.CRITICAL)

DATA_DIR = Path(os.environ.get("SLEEPCLICK_DATA_DIR", "./data"))
CACHE_DIR = DATA_DIR / "cache"

NATIVE_HZ = 10
DAY_START_HOUR = 15
SAMPLES_PER_DAY = NATIVE_HZ * 60 * 60 * 24

# Down-sampling factors relative to the native 10 Hz signal.
FREQ_FACTORS = {"10Hz": 1, "1Hz": 10, "1min": 10 * 60, "5min": 5 * 10 * 60}
# Column of the cached label tensor y.pt, which stores (morning, evening).
TARGET_COLS = {"morning": 0, "evening": 1}


# ---------------------------------------------------------------------------
# Step 1: raw recording -> cleaned per-day parquet
# ---------------------------------------------------------------------------

def watchdata_to_dataframe(
    file_path: Path,
    allow_duplicates: bool = False,
    nonwear: Literal["ggir", "cta", "detach"] | None = "ggir",
) -> pl.DataFrame:
    """Read a wrist-worn recording into a polars frame [time, x, y, z, non_wear].

    Args:
        file_path: GENEActiv .bin or Actigraph .gt3x file.
        allow_duplicates: passed through to wristpy for duplicate timestamps.
        nonwear: non-wear algorithm, run on the native-rate signal.
            "ggir"   - GGIR-style, acceleration only.
            "cta"    - Zhou 2015, temperature + acceleration.
            "detach" - Vert 2022 DETACH, temperature + acceleration.
            "cta"/"detach" fall back to "ggir" if the file has no temperature.
            None skips detection and leaves ``non_wear`` null.

    ``non_wear`` is an Int8 flag (1 = non-wear) forward-filled from the coarse
    detection windows onto the accelerometer timeline. Results are cached as
    parquet under ``DATA_DIR/full_file``.
    """
    # Imported here so the rest of the module (and the model) works without wristpy.
    from wristpy.io.readers import readers
    from wristpy.processing import metrics

    file_path = Path(file_path)
    cache_path = DATA_DIR / "full_file" / f"{file_path.stem}_wristpy_data.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    watch = readers.read_watch_data(file_path, allow_duplicates=allow_duplicates)
    acc = watch.acceleration

    nonwear_measurement = None
    if nonwear is not None:
        temp = getattr(watch, "temperature", None)
        has_temp = temp is not None and temp.measurements is not None
        algo = nonwear
        if algo in ("cta", "detach") and not has_temp:
            print(f"[warn] non-wear algorithm '{algo}' needs temperature, which is "
                  f"absent in this file; falling back to 'ggir'.")
            algo = "ggir"

        if algo == "ggir":
            nonwear_measurement = metrics.detect_nonwear(acc)
        elif algo == "cta":
            nonwear_measurement = metrics.combined_temp_accel_detect_nonwear(acc, temp)
        elif algo == "detach":
            nonwear_measurement = metrics.detach_nonwear(acc, temp)
        else:
            raise ValueError(f"Unknown non-wear algorithm: {nonwear!r}")

    df = pl.DataFrame(
        {
            "time": acc.time,
            "x": acc.measurements[:, 0],
            "y": acc.measurements[:, 1],
            "z": acc.measurements[:, 2],
        }
    ).set_sorted("time")

    if nonwear_measurement is not None:
        nonwear_df = pl.DataFrame(
            {"time": nonwear_measurement.time, "non_wear": nonwear_measurement.measurements}
        ).set_sorted("time").with_columns(pl.col("non_wear").cast(pl.Int8))
        df = df.join_asof(nonwear_df, on="time", strategy="backward")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Int8).alias("non_wear"))

    df.write_parquet(cache_path)
    return df


def process_single_file(file_path: Path):
    """Cut one recording into complete 15:00-15:00 days with full wear.

    Writes ``DATA_DIR/cleaned_file/{stem}.parquet`` with a ``day_idx`` column.
    Files whose stem does not start with a numeric participant id are skipped.
    """
    store_path = DATA_DIR / "cleaned_file" / f"{file_path.stem}.parquet"
    if not file_path.stem.split("_")[0].isdigit() or store_path.exists():
        return
    store_path.parent.mkdir(parents=True, exist_ok=True)

    data = watchdata_to_dataframe(file_path)

    # First sample at or after the first 15:00 boundary.
    first_t = data["time"].item(0)
    boundary = first_t.replace(hour=DAY_START_HOUR, minute=0, second=0, microsecond=0)
    if first_t > boundary:
        boundary += timedelta(days=1)
    start = (
        data.with_row_index()
        .filter(pl.col("time") >= boundary)
        .select("index")
        .item(0, 0)
    )

    n_days = (len(data) - start) // SAMPLES_PER_DAY
    window = data.slice(start, n_days * SAMPLES_PER_DAY)

    non_wear = window["non_wear"].to_numpy().reshape(n_days, SAMPLES_PER_DAY)
    is_null = (
        window["x"].is_null() | window["y"].is_null() | window["z"].is_null()
    ).to_numpy().reshape(n_days, SAMPLES_PER_DAY)
    bad = (non_wear == 1) | is_null
    valid_day_idx = np.where(bad.sum(axis=1) == 0)[0]

    days = [
        window.slice(d * SAMPLES_PER_DAY, SAMPLES_PER_DAY)
        .drop("non_wear")
        .with_columns(pl.lit(d).alias("day_idx"))
        for d in valid_day_idx
    ]
    if days:
        pl.concat(days).write_parquet(store_path)


def process_single_file_wrapper(file_path: Path):
    try:
        return process_single_file(file_path)
    except Exception as e:
        print(f"Error processing {file_path.stem}: {e}")
        with open(DATA_DIR / "error_log.txt", "a") as f:
            f.write(f"Error processing {file_path.stem}: {e}\n")
        return None


# ---------------------------------------------------------------------------
# Step 2: cleaned days -> (signal, labels) tensors
# ---------------------------------------------------------------------------

def do_feature_extraction_single(file_path: Path):
    """One cleaned recording -> (X, labels, pids, day_start_timestamps).

    X      : (D, SAMPLES_PER_DAY, 3) float16
    labels : (D, 2) (morning, evening) press time in clock hours, -1 if missing
    """
    pid = int(file_path.stem.split("_")[0])
    try:
        data = pl.read_parquet(file_path)
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return None

    all_data, all_labels, days = [], [], []
    for _, day in data.group_by("day_idx", maintain_order=True):
        if len(day) != SAMPLES_PER_DAY:
            continue
        morning, evening = get_day_clicks(file_path, day, bin_root=BIN_ROOT)
        arr = day.select(["x", "y", "z"]).cast(pl.Float32).to_numpy().astype(np.float16)
        all_data.append(torch.from_numpy(arr))
        all_labels.append((morning, evening))
        days.append(day["time"][0].timestamp())

    if not all_data:
        return None
    return (
        torch.stack(all_data),
        torch.tensor(all_labels),
        torch.full((len(days),), pid),
        torch.tensor(days).float(),
    )


# ---------------------------------------------------------------------------
# Step 3: concatenate + cache
# ---------------------------------------------------------------------------

def xy_cache_paths(cache_dir: Path, freq: str):
    """(X, y, pids, days) cache paths for a given sampling frequency."""
    cache_dir = Path(cache_dir)
    return (
        cache_dir / f"X_{freq}.pt",
        cache_dir / "y.pt",
        cache_dir / "pids.pt",
        cache_dir / "days.pt",
    )


def xy_cache_exists(cache_dir: Path, freq: str) -> bool:
    return all(p.exists() for p in xy_cache_paths(cache_dir, freq))


def build_xy(all_data, cache_dir: Path, freq_name: str = "10Hz",
             target: Literal["morning", "evening"] = "evening"):
    """Build X, y, pids, days once, cache them, and reload if present.

    On a cache miss, the native 10 Hz tensor is saved together with every
    down-sampled version in ``FREQ_FACTORS`` (non-overlapping means), and the
    one requested by ``freq_name`` is returned.

    Returns:
        X    : (N, T, 3) float16 at ``freq_name``
        y    : (N,) press time in clock hours for ``target`` (-1 = missing)
        pids : (N,) participant ids
        days : (N,) POSIX timestamp of each day's first sample
    """
    if freq_name not in FREQ_FACTORS:
        raise ValueError(f"freq_name must be one of {list(FREQ_FACTORS)}, got {freq_name!r}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path, pids_path, days_path = xy_cache_paths(cache_dir, freq_name)
    col = TARGET_COLS[target]

    if xy_cache_exists(cache_dir, freq_name):
        X = torch.load(x_path, map_location="cpu", mmap=True)
        y = torch.load(y_path, map_location="cpu")[:, col]
        pids = torch.load(pids_path, map_location="cpu")
        days = torch.load(days_path, map_location="cpu")
        return X, y, pids, days

    if all_data is None:
        raise RuntimeError(f"No cache in {cache_dir} and no data given to build one.")

    print("Building X/y cache ...")
    all_data = [r for r in all_data if r is not None]
    X = torch.cat([d[0] for d in all_data]).half()
    y = torch.cat([d[1] for d in all_data])
    pids = torch.cat([d[2] for d in all_data])
    days = torch.cat([d[3] for d in all_data])

    out = None
    for name, factor in FREQ_FACTORS.items():
        Xf = X if factor == 1 else X.unfold(1, factor, factor).mean(-1)
        torch.save(Xf, xy_cache_paths(cache_dir, name)[0])
        if name == freq_name:
            out = Xf
    torch.save(y, y_path)
    torch.save(pids, pids_path)
    torch.save(days, days_path)
    print("Done saving.")
    return out, y[:, col], pids, days


# ---------------------------------------------------------------------------
# Step 4: normalisation + participant-level folds
# ---------------------------------------------------------------------------

def compute_mean_std(X, idx, chunk_size=2048, cache_dir=None):
    """Per-channel mean/std over rows ``idx`` of X, streamed in chunks.

    X may be a (large) memory-mapped tensor; the result is cached on a hash of
    the selected rows and X's shape when ``cache_dir`` is given, since reading
    the full tensor is I/O bound.
    """
    idx = np.asarray(idx)
    cache_path = None
    if cache_dir is not None:
        key = hashlib.md5(np.ascontiguousarray(np.sort(idx)).tobytes()
                          + repr(tuple(X.shape)).encode()).hexdigest()
        cache_path = Path(cache_dir) / f"meanstd_{key}.pt"
        if cache_path.exists():
            mean_std = torch.load(cache_path, map_location="cpu")
            return mean_std[0], mean_std[1]

    n, s, ss = 0, None, None
    for i in range(0, len(idx), chunk_size):
        chunk = X[idx[i:i + chunk_size]].float()
        cs = chunk.sum(dim=[0, 1], keepdim=True)
        css = (chunk ** 2).sum(dim=[0, 1], keepdim=True)
        s = cs if s is None else s + cs
        ss = css if ss is None else ss + css
        n += chunk.shape[0] * chunk.shape[1]
    mean = s / n
    std = (ss / n - mean ** 2).clamp_min(0).sqrt()
    if cache_path is not None:
        torch.save(torch.stack([mean, std]), cache_path)
    return mean, std


def participant_folds(pids, n_folds: int = 5, seed: int = 0):
    """Assign participants (not days) to folds. -> list of row-index arrays."""
    pids = np.asarray(pids)
    unique_pids = sorted(set(pids.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)
    pid_fold = {p: f for f, fp in enumerate(np.array_split(unique_pids, n_folds))
                for p in fp.tolist()}
    row_fold = np.fromiter((pid_fold[p] for p in pids.tolist()), dtype=np.int64, count=len(pids))
    return [np.where(row_fold == f)[0] for f in range(n_folds)]


class FoldFactory:
    """Builds (train, val, test) loaders per fold on demand.

    Fold ``i`` is the test set, fold ``i+1`` the validation set and the rest
    training. Normalisation statistics come from the training rows only.
    Batches are ``(x, y, pid, day)`` with x of shape (B, T, C).
    """

    def __init__(self, X, y, pids, dates, fold_idx, n_folds, batch_size,
                 num_workers=0, cache_dir=None):
        self.X, self.y = X, y
        self.base = TensorDataset(X, y, pids, dates)   # one shared, un-normalised copy
        self.fold_idx = fold_idx
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir

    def _loader(self, idx, mean, std, shuffle):
        std = std + 1e-8

        def collate(batch):
            xs, ys, pids, dates = default_collate(batch)
            return (xs - mean) / std, ys, pids, dates

        return DataLoader(
            Subset(self.base, idx.tolist()),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
            persistent_workers=self.num_workers > 0,
        )

    def __len__(self):
        return self.n_folds

    def get_fold(self, i):
        test_idx = self.fold_idx[i]
        val_idx = self.fold_idx[(i + 1) % self.n_folds]
        train_idx = np.concatenate(
            [self.fold_idx[j] for j in range(self.n_folds)
             if j not in (i, (i + 1) % self.n_folds)]
        )
        mean, std = compute_mean_std(self.X, train_idx, cache_dir=self.cache_dir)
        train_loader = self._loader(torch.as_tensor(train_idx), mean, std, shuffle=True)
        val_loader = self._loader(torch.as_tensor(val_idx), mean, std, shuffle=False)
        test_loader = self._loader(torch.as_tensor(test_idx), mean, std, shuffle=False)
        print(f"[fold {i}] days train/val/test = "
              f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        return train_loader, val_loader, test_loader


def create_folds(all_data, n_folds=5, batch_size=16, num_workers=16,
                 cache_dir=CACHE_DIR, freq_name="10Hz", target="evening"):
    """Load/build the tensors, drop days without a label, and split by participant."""
    X, y, pids, days = build_xy(all_data, cache_dir, freq_name, target=target)

    keep = y != -1
    X, y, pids, days = X[keep], y[keep], pids[keep], days[keep]

    fold_idx = participant_folds(pids, n_folds)
    return FoldFactory(X, y, torch.as_tensor(pids), days, fold_idx, n_folds,
                       batch_size, num_workers, cache_dir=cache_dir)


def preprocess_data(freq="10Hz", target="evening", split_raw=False,
                    batch_size=32, n_workers=8):
    """End-to-end: (optionally) split raw files, extract labels, return folds."""
    if split_raw:
        all_files = list(BIN_ROOT.rglob("*.bin"))
        with Pool(processes=n_workers) as pool:
            list(tqdm(pool.imap_unordered(process_single_file_wrapper, all_files, chunksize=4),
                      total=len(all_files)))

    all_results = None
    if xy_cache_exists(CACHE_DIR, freq):
        print("X/y cache found, skipping feature extraction.")
    else:
        files = list((DATA_DIR / "cleaned_file").rglob("*.parquet"))
        print(f"Found {len(files)} cleaned recordings for feature extraction.")
        with Pool(processes=n_workers) as pool:
            all_results = list(tqdm(
                pool.imap_unordered(do_feature_extraction_single, files, chunksize=4),
                total=len(files),
            ))
    return create_folds(all_results, cache_dir=CACHE_DIR, batch_size=batch_size,
                        freq_name=freq, target=target)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--freq", default="10Hz", choices=list(FREQ_FACTORS))
    parser.add_argument("--target", default="evening", choices=list(TARGET_COLS))
    parser.add_argument("--split-raw", action="store_true",
                        help="run step 1 (raw .bin -> cleaned day parquet) first")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    preprocess_data(args.freq, args.target, args.split_raw, n_workers=args.workers)
