from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyBigWig
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
from torch.utils.data import Dataset, Subset


DNA_TO_ID = np.full(256, 4, dtype=np.int64)
for base, idx in {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}.items():
    DNA_TO_ID[ord(base)] = idx
    DNA_TO_ID[ord(base.lower())] = idx


def parse_ab_label(value) -> int:
    text = str(value).strip().upper()
    if text in {"A", "1", "+", "TRUE"}:
        return 1
    if text in {"B", "0", "-1", "-", "FALSE"}:
        return 0
    raise ValueError(f"Cannot parse AB label {value!r}; expected A/B or 1/0.")


@dataclass
class IntervalTable:
    frame: pd.DataFrame
    label_col: str
    split_col: Optional[str]

    @classmethod
    def load(cls, path: str, label_col: str = "label", split_col: Optional[str] = "split") -> "IntervalTable":
        try:
            frame = pd.read_csv(path, sep=None, engine="python")
            needed = {"chrom", "start", "end"}
            if not needed.issubset(frame.columns):
                raise ValueError
        except Exception:
            frame = pd.read_csv(path, sep="\t", header=None, comment="#")
            if frame.shape[1] < 4:
                raise ValueError("Interval table must contain at least chrom, start, end, label columns.")
            cols = ["chrom", "start", "end", "label"] + [f"extra_{i}" for i in range(frame.shape[1] - 4)]
            frame.columns = cols
            label_col = "label"
            if split_col not in frame.columns:
                split_col = None

        for col in ("chrom", "start", "end"):
            if col not in frame.columns:
                raise ValueError(f"Missing required column {col!r} in {path}.")
        if label_col not in frame.columns:
            raise ValueError(f"Missing label column {label_col!r} in {path}.")
        if split_col is not None and split_col not in frame.columns:
            split_col = None

        frame = frame.copy()
        frame["start"] = frame["start"].astype(int)
        frame["end"] = frame["end"].astype(int)
        frame["_ab_label"] = frame[label_col].map(parse_ab_label).astype(int)
        return cls(frame=frame, label_col=label_col, split_col=split_col)

    def split_indices(self, split_name: str) -> list[int]:
        if self.split_col is None:
            return []
        split = self.frame[self.split_col].astype(str).str.lower()
        return self.frame.index[split == split_name.lower()].tolist()


def centered_window(start: int, end: int, length: int) -> tuple[int, int]:
    center = (int(start) + int(end)) // 2
    half = length // 2
    new_start = center - half
    return new_start, new_start + length


def range_cache_path(cache_dir: Optional[str], prefix: str, payload: dict) -> Optional[Path]:
    if not cache_dir:
        return None
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{prefix}_{digest}.json"


def load_cached_ranges(path: Optional[Path]) -> Optional[list[tuple[str, int, int]]]:
    if path is None or not path.exists():
        return None
    with open(path) as handle:
        data = json.load(handle)
    return [(str(chrom), int(start), int(end)) for chrom, start, end in data["ranges"]]


def write_cached_ranges(path: Optional[Path], ranges: list[tuple[str, int, int]], payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump({"payload": payload, "ranges": ranges}, handle)
    tmp_path.replace(path)


class DNAABDataset(Dataset):
    def __init__(self, intervals: IntervalTable, fasta_path: str, length: int):
        self.table = intervals.frame.reset_index(drop=True)
        self.fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
        self.length = int(length)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        start, end = centered_window(row.start, row.end, self.length)
        pad_left = max(0, -start)
        start = max(0, start)
        try:
            seq = self.fasta[row.chrom][start:end]
        except KeyError:
            chrom = str(row.chrom)
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            seq = self.fasta[alt][start:end]
        seq = ("N" * pad_left + seq)[: self.length].ljust(self.length, "N")
        ids = torch.from_numpy(DNA_TO_ID[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]).long()
        return ids, torch.tensor(row._ab_label, dtype=torch.long)


class RnaStrandABDataset(Dataset):
    def __init__(
        self,
        intervals: IntervalTable,
        plus_bigwig: str,
        minus_bigwig: str,
        length: int,
        log1p: bool = True,
        per_window_zscore: bool = True,
    ):
        self.table = intervals.frame.reset_index(drop=True)
        self.plus_bigwig = plus_bigwig
        self.minus_bigwig = minus_bigwig
        self.plus_bw = None
        self.minus_bw = None
        self.length = int(length)
        self.log1p = log1p
        self.per_window_zscore = per_window_zscore

    def __len__(self) -> int:
        return len(self.table)

    def _open_bigwigs(self):
        if self.plus_bw is None:
            self.plus_bw = pyBigWig.open(self.plus_bigwig)
        if self.minus_bw is None:
            self.minus_bw = pyBigWig.open(self.minus_bigwig)

    def _values(self, bw, chrom: str, start: int, end: int) -> np.ndarray:
        try:
            values = bw.values(chrom, start, end)
        except RuntimeError:
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            values = bw.values(alt, start, end)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.log1p:
            values = np.sign(values) * np.log1p(np.abs(values))
        return values

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        self._open_bigwigs()
        start, end = centered_window(row.start, row.end, self.length)
        if start < 0:
            pad = -start
            start = 0
        else:
            pad = 0
        plus = self._values(self.plus_bw, row.chrom, start, end)
        minus = self._values(self.minus_bw, row.chrom, start, end)
        if pad:
            plus = np.pad(plus, (pad, 0))[: self.length]
            minus = np.pad(minus, (pad, 0))[: self.length]
        plus = plus[: self.length]
        minus = minus[: self.length]
        if len(plus) < self.length:
            plus = np.pad(plus, (0, self.length - len(plus)))
            minus = np.pad(minus, (0, self.length - len(minus)))
        x = torch.from_numpy(np.stack([plus, minus], axis=-1)).float()
        if self.per_window_zscore:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True).clamp_min(1e-4)
            x = (x - mean) / std
        return x, torch.tensor(row._ab_label, dtype=torch.long)


class CompartmentBigWigFusionDataset(Dataset):
    def __init__(
        self,
        fasta_path: str,
        plus_bigwig: str,
        minus_bigwig: str,
        target_bigwig: str,
        chromosomes: list[str],
        bin_size: int = 50000,
        input_length: int = 4096,
        max_bins: Optional[int] = None,
        target_threshold: float = 0.5,
        log1p_tracks: bool = True,
        per_window_zscore: bool = True,
        track_window: str = "center",
        range_cache_dir: Optional[str] = None,
        validate_ranges: bool = True,
    ):
        if track_window not in {"center", "bin"}:
            raise ValueError("track_window must be 'center' or 'bin'.")
        self.fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
        self.plus_bigwig = plus_bigwig
        self.minus_bigwig = minus_bigwig
        self.target_bigwig = target_bigwig
        self.plus_bw = None
        self.minus_bw = None
        self.target_bw = None
        self._bw_pid = None
        self._target_bw_pid = None
        self.bin_size = int(bin_size)
        self.input_length = int(input_length)
        self.target_threshold = float(target_threshold)
        self.log1p_tracks = log1p_tracks
        self.per_window_zscore = per_window_zscore
        self.track_window = track_window
        self.validate_ranges = bool(validate_ranges)
        self.bins = self._make_bins(chromosomes, max_bins=max_bins)

    def _make_bins(self, chromosomes: list[str], max_bins: Optional[int]) -> list[tuple[str, int, int, float]]:
        target_bw = self._open_target_bigwig()
        chrom_sizes = target_bw.chroms()
        bins = []
        for chrom in chromosomes:
            if chrom not in chrom_sizes:
                continue
            chrom_len = chrom_sizes[chrom]
            for start in range(0, chrom_len, self.bin_size):
                end = min(start + self.bin_size, chrom_len)
                if end - start < self.bin_size:
                    continue
                target = self._target_stat(chrom, start, end)
                if target is None or not np.isfinite(target):
                    continue
                bins.append((chrom, start, end, float(target)))
                if max_bins is not None and len(bins) >= max_bins:
                    return bins
        return bins

    def __len__(self) -> int:
        return len(self.bins)

    def _open_bigwigs(self):
        pid = os.getpid()
        if self._bw_pid != pid:
            self.plus_bw = None
            self.minus_bw = None
            self._bw_pid = pid
        if self.plus_bw is None:
            self.plus_bw = pyBigWig.open(self.plus_bigwig)
        if self.minus_bw is None:
            self.minus_bw = pyBigWig.open(self.minus_bigwig)

    def _open_target_bigwig(self):
        pid = os.getpid()
        if self._target_bw_pid != pid:
            self.target_bw = None
            self._target_bw_pid = pid
        if self.target_bw is None:
            self.target_bw = pyBigWig.open(self.target_bigwig)
        return self.target_bw

    def _target_stat(self, chrom: str, start: int, end: int):
        bw = self._open_target_bigwig()
        try:
            return bw.stats(chrom, start, end, type="mean")[0]
        except RuntimeError:
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            return bw.stats(alt, start, end, type="mean")[0]

    def _fetch_sequence(self, chrom: str, start: int, end: int) -> torch.Tensor:
        win_start, win_end = centered_window(start, end, self.input_length)
        pad_left = max(0, -win_start)
        win_start = max(0, win_start)
        try:
            seq = self.fasta[chrom][win_start:win_end]
        except KeyError:
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            seq = self.fasta[alt][win_start:win_end]
        seq = ("N" * pad_left + seq)[: self.input_length].ljust(self.input_length, "N")
        return torch.from_numpy(DNA_TO_ID[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]).long()

    def _values(self, bw, chrom: str, start: int, end: int) -> np.ndarray:
        try:
            values = bw.values(chrom, start, end)
        except RuntimeError:
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            values = bw.values(alt, start, end)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.log1p_tracks:
            values = np.sign(values) * np.log1p(np.abs(values))
        return values

    def _binned_values(self, bw, chrom: str, start: int, end: int, n_bins: int) -> np.ndarray:
        try:
            values = bw.stats(chrom, start, end, type="mean", nBins=n_bins)
        except RuntimeError:
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            values = bw.stats(alt, start, end, type="mean", nBins=n_bins)
        values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.log1p_tracks:
            values = np.sign(values) * np.log1p(np.abs(values))
        return values

    def _fetch_tracks(self, chrom: str, start: int, end: int) -> torch.Tensor:
        self._open_bigwigs()
        if self.track_window == "bin":
            plus = self._binned_values(self.plus_bw, chrom, start, end, self.input_length)
            minus = self._binned_values(self.minus_bw, chrom, start, end, self.input_length)
        else:
            win_start, win_end = centered_window(start, end, self.input_length)
            pad_left = max(0, -win_start)
            win_start = max(0, win_start)
            plus = self._values(self.plus_bw, chrom, win_start, win_end)
            minus = self._values(self.minus_bw, chrom, win_start, win_end)
            if pad_left:
                plus = np.pad(plus, (pad_left, 0))[: self.input_length]
                minus = np.pad(minus, (pad_left, 0))[: self.input_length]
            plus = plus[: self.input_length]
            minus = minus[: self.input_length]
            if len(plus) < self.input_length:
                plus = np.pad(plus, (0, self.input_length - len(plus)))
                minus = np.pad(minus, (0, self.input_length - len(minus)))
        tracks = torch.from_numpy(np.stack([plus, minus], axis=-1)).float()
        if self.per_window_zscore:
            mean = tracks.mean(dim=0, keepdim=True)
            std = tracks.std(dim=0, keepdim=True).clamp_min(1e-4)
            tracks = (tracks - mean) / std
        return tracks

    def __getitem__(self, idx: int):
        chrom, start, end, target = self.bins[idx]
        seq = self._fetch_sequence(chrom, start, end)
        tracks = self._fetch_tracks(chrom, start, end)
        score = torch.tensor(target, dtype=torch.float32)
        cls = torch.tensor(1 if target >= self.target_threshold else 0, dtype=torch.long)
        return seq, tracks, score, cls


class RangeCompartmentBigWigFusionDataset(Dataset):
    def __init__(
        self,
        fasta_path: str,
        plus_bigwig: str,
        minus_bigwig: str,
        target_bigwig: str,
        chromosomes: list[str],
        range_size: int = 2_000_000,
        bin_size: int = 50_000,
        input_length: int = 1024,
        stride: Optional[int] = None,
        max_ranges: Optional[int] = None,
        target_threshold: float = 0.5,
        log1p_tracks: bool = True,
        per_window_zscore: bool = True,
        track_window: str = "center",
        range_cache_dir: Optional[str] = None,
        validate_ranges: bool = True,
    ):
        if range_size % bin_size != 0:
            raise ValueError("range_size must be divisible by bin_size.")
        self.bin_dataset = CompartmentBigWigFusionDataset(
            fasta_path=fasta_path,
            plus_bigwig=plus_bigwig,
            minus_bigwig=minus_bigwig,
            target_bigwig=target_bigwig,
            chromosomes=[],
            bin_size=bin_size,
            input_length=input_length,
            target_threshold=target_threshold,
            log1p_tracks=log1p_tracks,
            per_window_zscore=per_window_zscore,
            track_window=track_window,
        )
        self.range_size = int(range_size)
        self.bin_size = int(bin_size)
        self.input_length = int(input_length)
        self.n_bins = self.range_size // self.bin_size
        self.stride = int(stride) if stride is not None else self.range_size
        self.target_threshold = float(target_threshold)
        self.validate_ranges = bool(validate_ranges)
        self.range_cache_dir = range_cache_dir
        self.ranges = self._load_or_make_ranges(chromosomes, max_ranges=max_ranges)

    def _range_cache_payload(self, chromosomes: list[str], max_ranges: Optional[int]) -> dict:
        return {
            "dataset": self.__class__.__name__,
            "target_bigwig": os.path.abspath(self.bin_dataset.target_bigwig),
            "chromosomes": list(chromosomes),
            "range_size": self.range_size,
            "bin_size": self.bin_size,
            "stride": self.stride,
            "max_ranges": max_ranges,
            "validate_ranges": self.validate_ranges,
        }

    def _load_or_make_ranges(self, chromosomes: list[str], max_ranges: Optional[int]) -> list[tuple[str, int, int]]:
        payload = self._range_cache_payload(chromosomes, max_ranges)
        path = range_cache_path(self.range_cache_dir, "ranges", payload)
        cached = load_cached_ranges(path)
        if cached is not None:
            print(json.dumps({"event": "range_cache_hit", "path": str(path), "n_ranges": len(cached)}), flush=True)
            return cached
        print(json.dumps({"event": "range_cache_build", "path": str(path) if path else None}), flush=True)
        ranges = self._make_ranges(chromosomes, max_ranges=max_ranges)
        write_cached_ranges(path, ranges, payload)
        return ranges

    def _make_ranges(self, chromosomes: list[str], max_ranges: Optional[int]) -> list[tuple[str, int, int]]:
        target_bw = self.bin_dataset._open_target_bigwig()
        chrom_sizes = target_bw.chroms()
        ranges = []
        for chrom in chromosomes:
            if chrom not in chrom_sizes:
                continue
            chrom_len = chrom_sizes[chrom]
            for start in range(0, max(0, chrom_len - self.range_size + 1), self.stride):
                end = start + self.range_size
                if self.validate_ranges:
                    valid = True
                    for bin_start in range(start, end, self.bin_size):
                        target = self.bin_dataset._target_stat(chrom, bin_start, bin_start + self.bin_size)
                        if target is None or not np.isfinite(target):
                            valid = False
                            break
                else:
                    valid = True
                if valid:
                    ranges.append((chrom, start, end))
                    if max_ranges is not None and len(ranges) >= max_ranges:
                        return ranges
        return ranges

    def __len__(self) -> int:
        return len(self.ranges)

    def __getitem__(self, idx: int):
        chrom, start, end = self.ranges[idx]
        seqs, tracks, scores, classes = [], [], [], []
        for bin_start in range(start, end, self.bin_size):
            bin_end = bin_start + self.bin_size
            target = self.bin_dataset._target_stat(chrom, bin_start, bin_end)
            if target is None or not np.isfinite(target):
                raise RuntimeError(
                    f"Missing target after range validation: {self.bin_dataset.target_bigwig} "
                    f"{chrom}:{bin_start}-{bin_end}"
                )
            seqs.append(self.bin_dataset._fetch_sequence(chrom, bin_start, bin_end))
            tracks.append(self.bin_dataset._fetch_tracks(chrom, bin_start, bin_end))
            scores.append(float(target))
            classes.append(1 if float(target) >= self.target_threshold else 0)
        return (
            torch.stack(seqs),
            torch.stack(tracks),
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(classes, dtype=torch.long),
            (chrom, start, end),
        )


class BPLevelRangeCompartmentBigWigFusionDataset(Dataset):
    def __init__(
        self,
        fasta_path: str,
        plus_bigwig: str,
        minus_bigwig: str,
        target_bigwig: str,
        chromosomes: list[str],
        range_size: int = 2_000_000,
        bin_size: int = 50_000,
        stride: Optional[int] = None,
        max_ranges: Optional[int] = None,
        target_threshold: float = 0.0,
        log1p_tracks: bool = True,
        per_range_zscore: bool = False,
        range_cache_dir: Optional[str] = None,
        validate_ranges: bool = True,
    ):
        if range_size % bin_size != 0:
            raise ValueError("range_size must be divisible by bin_size.")
        self.fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
        self.plus_bigwig = plus_bigwig
        self.minus_bigwig = minus_bigwig
        self.target_bigwig = target_bigwig
        self.plus_bw = None
        self.minus_bw = None
        self.target_bw = None
        self._bw_pid = None
        self._target_bw_pid = None
        self.range_size = int(range_size)
        self.bin_size = int(bin_size)
        self.n_bins = self.range_size // self.bin_size
        self.stride = int(stride) if stride is not None else self.range_size
        self.target_threshold = float(target_threshold)
        self.log1p_tracks = log1p_tracks
        self.per_range_zscore = per_range_zscore
        self.validate_ranges = bool(validate_ranges)
        self.range_cache_dir = range_cache_dir
        self.ranges = self._load_or_make_ranges(chromosomes, max_ranges=max_ranges)

    def _range_cache_payload(self, chromosomes: list[str], max_ranges: Optional[int]) -> dict:
        return {
            "dataset": self.__class__.__name__,
            "target_bigwig": os.path.abspath(self.target_bigwig),
            "chromosomes": list(chromosomes),
            "range_size": self.range_size,
            "bin_size": self.bin_size,
            "stride": self.stride,
            "max_ranges": max_ranges,
            "validate_ranges": self.validate_ranges,
        }

    def _load_or_make_ranges(self, chromosomes: list[str], max_ranges: Optional[int]) -> list[tuple[str, int, int]]:
        payload = self._range_cache_payload(chromosomes, max_ranges)
        path = range_cache_path(self.range_cache_dir, "bp_ranges", payload)
        cached = load_cached_ranges(path)
        if cached is not None:
            print(json.dumps({"event": "range_cache_hit", "path": str(path), "n_ranges": len(cached)}), flush=True)
            return cached
        print(json.dumps({"event": "range_cache_build", "path": str(path) if path else None}), flush=True)
        ranges = self._make_ranges(chromosomes, max_ranges=max_ranges)
        write_cached_ranges(path, ranges, payload)
        return ranges

    def _open_bigwigs(self):
        pid = os.getpid()
        if self._bw_pid != pid:
            self.plus_bw = None
            self.minus_bw = None
            self._bw_pid = pid
        if self.plus_bw is None:
            self.plus_bw = pyBigWig.open(self.plus_bigwig)
        if self.minus_bw is None:
            self.minus_bw = pyBigWig.open(self.minus_bigwig)

    def _open_target_bigwig(self):
        pid = os.getpid()
        if self._target_bw_pid != pid:
            self.target_bw = None
            self._target_bw_pid = pid
        if self.target_bw is None:
            self.target_bw = pyBigWig.open(self.target_bigwig)
        return self.target_bw

    def _target_stat(self, chrom: str, start: int, end: int):
        bw = self._open_target_bigwig()
        try:
            return bw.stats(chrom, start, end, type="mean")[0]
        except RuntimeError:
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            return bw.stats(alt, start, end, type="mean")[0]

    def _make_ranges(self, chromosomes: list[str], max_ranges: Optional[int]) -> list[tuple[str, int, int]]:
        target_bw = self._open_target_bigwig()
        chrom_sizes = target_bw.chroms()
        ranges = []
        for chrom in chromosomes:
            if chrom not in chrom_sizes:
                continue
            chrom_len = chrom_sizes[chrom]
            for start in range(0, max(0, chrom_len - self.range_size + 1), self.stride):
                end = start + self.range_size
                if self.validate_ranges:
                    valid = True
                    for bin_start in range(start, end, self.bin_size):
                        target = self._target_stat(chrom, bin_start, bin_start + self.bin_size)
                        if target is None or not np.isfinite(target):
                            valid = False
                            break
                else:
                    valid = True
                if valid:
                    ranges.append((chrom, start, end))
                    if max_ranges is not None and len(ranges) >= max_ranges:
                        return ranges
        return ranges

    def __len__(self) -> int:
        return len(self.ranges)

    def _fetch_sequence(self, chrom: str, start: int, end: int) -> torch.Tensor:
        try:
            seq = self.fasta[chrom][start:end]
        except KeyError:
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            seq = self.fasta[alt][start:end]
        seq = seq[: self.range_size].ljust(self.range_size, "N")
        return torch.from_numpy(DNA_TO_ID[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]).long()

    def _values(self, bw, chrom: str, start: int, end: int) -> np.ndarray:
        try:
            values = bw.values(chrom, start, end)
        except RuntimeError:
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            values = bw.values(alt, start, end)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.log1p_tracks:
            values = np.sign(values) * np.log1p(np.abs(values))
        return values

    def _fetch_tracks(self, chrom: str, start: int, end: int) -> torch.Tensor:
        self._open_bigwigs()
        plus = self._values(self.plus_bw, chrom, start, end)[: self.range_size]
        minus = self._values(self.minus_bw, chrom, start, end)[: self.range_size]
        if len(plus) < self.range_size:
            plus = np.pad(plus, (0, self.range_size - len(plus)))
            minus = np.pad(minus, (0, self.range_size - len(minus)))
        tracks = torch.from_numpy(np.stack([plus, minus], axis=-1)).float()
        if self.per_range_zscore:
            mean = tracks.mean(dim=0, keepdim=True)
            std = tracks.std(dim=0, keepdim=True).clamp_min(1e-4)
            tracks = (tracks - mean) / std
        return tracks

    def __getitem__(self, idx: int):
        chrom, start, end = self.ranges[idx]
        scores, classes = [], []
        for bin_start in range(start, end, self.bin_size):
            target = self._target_stat(chrom, bin_start, bin_start + self.bin_size)
            if target is None or not np.isfinite(target):
                raise RuntimeError(f"Missing target: {self.target_bigwig} {chrom}:{bin_start}-{bin_start + self.bin_size}")
            scores.append(float(target))
            classes.append(1 if float(target) >= self.target_threshold else 0)
        return (
            self._fetch_sequence(chrom, start, end),
            self._fetch_tracks(chrom, start, end),
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(classes, dtype=torch.long),
            (chrom, start, end),
        )


@dataclass
class CellBigWigConfig:
    cell: str
    fasta_path: str
    plus_bigwig: str
    minus_bigwig: str
    target_bigwig: str


class MultiCellRangeCompartmentBigWigFusionDataset(Dataset):
    def __init__(
        self,
        configs: list[CellBigWigConfig],
        chromosomes: list[str],
        range_size: int = 2_000_000,
        bin_size: int = 50_000,
        input_length: int = 1024,
        stride: Optional[int] = None,
        max_ranges_per_cell: Optional[int] = None,
        target_threshold: float = 0.5,
        log1p_tracks: bool = True,
        per_window_zscore: bool = True,
        track_window: str = "center",
        range_cache_dir: Optional[str] = None,
        validate_ranges: bool = True,
    ):
        self.datasets: list[RangeCompartmentBigWigFusionDataset] = []
        self.cells: list[str] = []
        self.offsets = [0]
        for cfg in configs:
            ds = RangeCompartmentBigWigFusionDataset(
                fasta_path=cfg.fasta_path,
                plus_bigwig=cfg.plus_bigwig,
                minus_bigwig=cfg.minus_bigwig,
                target_bigwig=cfg.target_bigwig,
                chromosomes=chromosomes,
                range_size=range_size,
                bin_size=bin_size,
                input_length=input_length,
                stride=stride,
                max_ranges=max_ranges_per_cell,
                target_threshold=target_threshold,
                log1p_tracks=log1p_tracks,
                per_window_zscore=per_window_zscore,
                track_window=track_window,
                range_cache_dir=range_cache_dir,
                validate_ranges=validate_ranges,
            )
            if len(ds) == 0:
                continue
            self.datasets.append(ds)
            self.cells.append(cfg.cell)
            self.offsets.append(self.offsets[-1] + len(ds))
        self.n_bins = range_size // bin_size

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, idx: int):
        ds_idx = bisect_right(self.offsets, idx) - 1
        local_idx = idx - self.offsets[ds_idx]
        seq, tracks, score, cls, region = self.datasets[ds_idx][local_idx]
        chrom, start, end = region
        return seq, tracks, score, cls, (self.cells[ds_idx], chrom, start, end)

    def summary(self) -> list[dict]:
        return [
            {"cell": cell, "n_ranges": len(ds)}
            for cell, ds in zip(self.cells, self.datasets)
        ]


class MultiCellBPLevelRangeCompartmentBigWigFusionDataset(Dataset):
    def __init__(
        self,
        configs: list[CellBigWigConfig],
        chromosomes: list[str],
        range_size: int = 2_000_000,
        bin_size: int = 50_000,
        stride: Optional[int] = None,
        max_ranges_per_cell: Optional[int] = None,
        target_threshold: float = 0.0,
        log1p_tracks: bool = True,
        per_range_zscore: bool = False,
        range_cache_dir: Optional[str] = None,
        validate_ranges: bool = True,
    ):
        self.datasets: list[BPLevelRangeCompartmentBigWigFusionDataset] = []
        self.cells: list[str] = []
        self.offsets = [0]
        for cfg in configs:
            ds = BPLevelRangeCompartmentBigWigFusionDataset(
                fasta_path=cfg.fasta_path,
                plus_bigwig=cfg.plus_bigwig,
                minus_bigwig=cfg.minus_bigwig,
                target_bigwig=cfg.target_bigwig,
                chromosomes=chromosomes,
                range_size=range_size,
                bin_size=bin_size,
                stride=stride,
                max_ranges=max_ranges_per_cell,
                target_threshold=target_threshold,
                log1p_tracks=log1p_tracks,
                per_range_zscore=per_range_zscore,
                range_cache_dir=range_cache_dir,
                validate_ranges=validate_ranges,
            )
            if len(ds) == 0:
                continue
            self.datasets.append(ds)
            self.cells.append(cfg.cell)
            self.offsets.append(self.offsets[-1] + len(ds))
        self.n_bins = range_size // bin_size

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, idx: int):
        ds_idx = bisect_right(self.offsets, idx) - 1
        local_idx = idx - self.offsets[ds_idx]
        seq, tracks, score, cls, region = self.datasets[ds_idx][local_idx]
        chrom, start, end = region
        return seq, tracks, score, cls, (self.cells[ds_idx], chrom, start, end)

    def summary(self) -> list[dict]:
        return [
            {"cell": cell, "n_ranges": len(ds)}
            for cell, ds in zip(self.cells, self.datasets)
        ]


def subset_by_split(dataset: Dataset, intervals: IntervalTable, split_name: str) -> Optional[Subset]:
    indices = intervals.split_indices(split_name)
    if not indices:
        return None
    return Subset(dataset, indices)


def random_train_val_split(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    n_val = max(1, int(round(n * val_fraction)))
    return Subset(dataset, perm[n_val:]), Subset(dataset, perm[:n_val])


def resample_tracks(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[1] == length:
        return x
    return F.interpolate(x.transpose(1, 2), size=length, mode="linear", align_corners=False).transpose(1, 2)
