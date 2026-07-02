from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


AUTOSOMES = {
    "human": [f"chr{i}" for i in range(1, 23)],
    "mouse": [f"chr{i}" for i in range(1, 20)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate autosome CrossCompartment predictions to 100kb bins using only the center region of each range."
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", default=["Sample"])
    parser.add_argument(
        "--chroms",
        default=None,
        help="Comma-separated chromosomes to aggregate. Defaults to all autosomes for each cell genome.",
    )
    parser.add_argument(
        "--prediction-set",
        default="autosomes",
        help="Middle token in prediction filenames: {cell}_{prediction_set}_test_predictions.tsv.",
    )
    parser.add_argument("--bin-size", type=int, default=100_000)
    parser.add_argument("--center-size", type=int, default=1_100_000)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--label-bw",
        action="append",
        default=[],
        metavar="CELL=PATH",
        help="Target label bigWig path for a cell. May be provided multiple times.",
    )
    parser.add_argument(
        "--genome",
        choices=sorted(AUTOSOMES),
        default=None,
        help="Genome used to infer autosomes when --chroms is omitted.",
    )
    return parser.parse_args()


def parse_chroms_arg(value: str | None, genome: str | None) -> list[str]:
    if value is None:
        if genome is None:
            raise ValueError("Pass --chroms or pass --genome to infer autosomes")
        return AUTOSOMES[genome]
    chroms = [item.strip() for item in value.split(",") if item.strip()]
    if not chroms:
        raise ValueError("--chroms was provided but no chromosome names were parsed")
    return chroms


def parse_label_bw_overrides(values: list[str]) -> dict[str, str]:
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--label-bw must have CELL=PATH format, got {item!r}")
        cell, path = item.split("=", 1)
        cell = cell.strip()
        path = path.strip()
        if not cell or not path:
            raise ValueError(f"--label-bw must have CELL=PATH format, got {item!r}")
        out[cell] = path
    return out


def center_filter(df: pd.DataFrame, center_size: int) -> pd.DataFrame:
    mid = (df["range_start"].to_numpy(float) + df["range_end"].to_numpy(float)) / 2.0
    half = center_size / 2.0
    keep = (df["start"].to_numpy(float) >= mid - half) & (df["end"].to_numpy(float) <= mid + half)
    return df.loc[keep].copy()


def dedupe_source_bins(path: Path, chroms: set[str], center_size: int, chunksize: int) -> tuple[pd.DataFrame, int, int]:
    grouped_chunks = []
    n_raw = 0
    n_center = 0
    usecols = ["chrom", "start", "end", "range_start", "range_end", "target_score", "pred_score"]
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["chrom"].isin(chroms)]
        chunk = chunk[np.isfinite(chunk["pred_score"]) & np.isfinite(chunk["target_score"])]
        n_raw += len(chunk)
        chunk = center_filter(chunk, center_size)
        n_center += len(chunk)
        if chunk.empty:
            continue
        grouped_chunks.append(
            chunk.groupby(["chrom", "start", "end"], as_index=False).agg(
                pred_score=("pred_score", "mean"),
                source_target_score=("target_score", "mean"),
                n_overlaps=("pred_score", "size"),
            )
        )
    if not grouped_chunks:
        return pd.DataFrame(columns=["chrom", "start", "end", "pred_score", "source_target_score", "n_overlaps"]), n_raw, n_center
    grouped = pd.concat(grouped_chunks, ignore_index=True)
    grouped = (
        grouped.groupby(["chrom", "start", "end"], as_index=False)
        .agg(
            pred_score=("pred_score", "mean"),
            source_target_score=("source_target_score", "mean"),
            n_overlaps=("n_overlaps", "sum"),
        )
        .sort_values(["chrom", "start", "end"])
        .reset_index(drop=True)
    )
    return grouped, n_raw, n_center


def label_bw_100kb(cell: str, label_bws: dict[str, str], chroms: list[str], bin_size: int, threshold: float) -> pd.DataFrame:
    rows = []
    with pyBigWig.open(label_bws[cell]) as bw:
        bw_chroms = bw.chroms()
        for chrom in chroms:
            if chrom not in bw_chroms:
                continue
            chrom_len = int(bw_chroms[chrom])
            starts = np.arange(0, chrom_len - bin_size + 1, bin_size, dtype=np.int64)
            for i, start in enumerate(starts, start=1):
                end = int(start + bin_size)
                value = bw.stats(chrom, int(start), end, type="mean")[0]
                value = 0.0 if value is None or not np.isfinite(float(value)) else float(value)
                rows.append(
                    {
                        "bead": i,
                        "chrom": chrom,
                        "start": int(start),
                        "end": end,
                        "target_score_100kb": value,
                        "target_label": 1 if value > threshold else 0,
                        "target_ab": "A" if value > threshold else "B",
                    }
                )
    return pd.DataFrame(rows)


def overlap_weighted_aggregate(source: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for chrom, chrom_bins in bins.groupby("chrom", sort=False):
        chrom_source = source[source["chrom"].eq(chrom)].sort_values(["start", "end"])
        if chrom_source.empty:
            continue
        starts = chrom_source["start"].to_numpy(np.int64)
        ends = chrom_source["end"].to_numpy(np.int64)
        pred = chrom_source["pred_score"].to_numpy(float)
        source_target = chrom_source["source_target_score"].to_numpy(float)
        left = 0
        for b in chrom_bins.itertuples(index=False):
            while left < len(ends) and ends[left] <= b.start:
                left += 1
            idx = left
            weight = 0.0
            pred_sum = 0.0
            source_target_sum = 0.0
            while idx < len(starts) and starts[idx] < b.end:
                ov = min(int(ends[idx]), int(b.end)) - max(int(starts[idx]), int(b.start))
                if ov > 0:
                    weight += ov
                    pred_sum += pred[idx] * ov
                    source_target_sum += source_target[idx] * ov
                idx += 1
            if weight > 0:
                row = b._asdict()
                row["pred_score_100kb"] = pred_sum / weight
                row["source_target_score_100kb"] = source_target_sum / weight
                row["covered_bp"] = weight
                rows.append(row)
    return pd.DataFrame(rows)


def metrics_for(
    cell: str,
    chroms: list[str],
    agg: pd.DataFrame,
    args: argparse.Namespace,
    n_raw: int,
    n_center: int,
    source: pd.DataFrame,
    pred_path: Path,
) -> dict:
    labels = agg["target_label"].to_numpy(int)
    scores = agg["pred_score_100kb"].to_numpy(float)
    target = agg["target_score_100kb"].to_numpy(float)
    return {
        "cell": cell,
        "chroms": ",".join(chroms),
        "bin_size": args.bin_size,
        "center_size": args.center_size,
        "n_raw_rows": n_raw,
        "n_center_rows": n_center,
        "n_unique_source_bins": len(source),
        "n_100kb_bins": len(agg),
        "positives": int(labels.sum()),
        "negatives": int((1 - labels).sum()),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan"),
        "aupr": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan"),
        "pearson": float(pearsonr(target, scores).statistic) if len(agg) > 1 else float("nan"),
        "spearman": float(spearmanr(target, scores).statistic) if len(agg) > 1 else float("nan"),
        "prediction_file": str(pred_path),
        "label_bw": args.label_bws[cell],
    }


def main() -> None:
    args = parse_args()
    args.label_bws = parse_label_bw_overrides(args.label_bw)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_prefix.with_suffix(".per_100kb.tsv")
    metrics_path = args.output_prefix.with_suffix(".metrics.tsv")
    if detail_path.exists():
        detail_path.unlink()

    metric_rows = []
    wrote_header = False
    for cell in args.cells:
        if cell not in args.label_bws:
            raise ValueError(f"No label bigWig configured for cell {cell!r}; pass --label-bw {cell}=PATH")
        chroms = parse_chroms_arg(args.chroms, args.genome)
        pred_path = args.prediction_dir / f"{cell}_{args.prediction_set}_test_predictions.tsv"
        print(f"[aggregate] {cell}: reading {pred_path}", flush=True)
        source, n_raw, n_center = dedupe_source_bins(pred_path, set(chroms), args.center_size, args.chunksize)
        print(f"[aggregate] {cell}: {n_center} center rows -> {len(source)} unique 2200bp bins", flush=True)
        bins = label_bw_100kb(cell, args.label_bws, chroms, args.bin_size, args.threshold)
        agg = overlap_weighted_aggregate(source, bins)
        agg.insert(0, "cell", cell)
        agg["pred_ab"] = np.where(agg["pred_score_100kb"] > args.threshold, "A", "B")
        agg.to_csv(detail_path, sep="\t", index=False, mode="a", header=not wrote_header)
        wrote_header = True
        metric_rows.append(metrics_for(cell, chroms, agg, args, n_raw, n_center, source, pred_path))
        del source, bins, agg
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(metrics_path, sep="\t", index=False)
    print(metrics[["cell", "n_100kb_bins", "auroc", "aupr", "pearson", "spearman"]].to_string(index=False))
    print(f"wrote {detail_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
