from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pyBigWig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ab_compartment.datasets import BPLevelRangeCompartmentBigWigFusionDataset, RangeCompartmentBigWigFusionDataset
from src.ab_compartment.models import BPLevelFusionRangePredictor, CrossCompartmentFusionRangePredictor
from train_fusion_compartment import parse_chroms
from train_fusion_compartment_range import collate


def parse_args():
    parser = argparse.ArgumentParser(description="Stream range-level DNA+RO-seq compartment predictions to TSV.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--plus-bigwig", required=True)
    parser.add_argument("--minus-bigwig", required=True)
    parser.add_argument("--target-bigwig", required=True)
    parser.add_argument("--chroms", default="chr8")
    parser.add_argument("--output", required=True)
    parser.add_argument("--range-size", type=int, default=2_000_000)
    parser.add_argument("--bin-size", type=int, default=50_000)
    parser.add_argument("--stride", type=int, default=2_000_000)
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--track-window", choices=["center", "bin"], default=None)
    parser.add_argument("--per-window-zscore", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--per-range-zscore", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-ranges", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every-ranges", type=int, default=200)
    parser.add_argument("--append", action="store_true", help="Append rows to an existing TSV and skip the header if it exists.")
    return parser.parse_args()


def build_model(checkpoint: str, args) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    input_mode = train_args.get("input_mode", "per-bin")
    if input_mode == "bp":
        train_range_size = int(train_args.get("range_size", args.range_size))
        train_bin_size = int(train_args.get("bin_size", args.bin_size))
        encoder_chunk_size = train_args.get("encoder_chunk_size")
        encoder_chunk_bins = None if encoder_chunk_size is None else int(encoder_chunk_size) // train_bin_size
        bp_downsample_strides = tuple(int(x) for x in train_args.get("bp_downsample_strides", (10, 10, 10, 10, 5)))
        bp_stem_kernels = train_args.get("bp_stem_kernels")
        if bp_stem_kernels is not None:
            bp_stem_kernels = tuple(int(x) for x in bp_stem_kernels)
        model = BPLevelFusionRangePredictor(
            d_model=int(train_args.get("d_model", 128)),
            n_bins=train_range_size // train_bin_size,
            depth=int(train_args.get("depth", 2)),
            num_heads=int(train_args.get("num_heads", 8)),
            dropout=float(train_args.get("dropout", 0.1)),
            ablation=train_args.get("ablation", "full"),
            fusion_method=train_args.get("fusion_method", "gate"),
            decoder_layers=int(train_args.get("decoder_layers", 0)),
            decoder_kernel_size=int(train_args.get("decoder_kernel_size", 5)),
            decoder_type=train_args.get("decoder_type", "conv"),
            encoder_chunk_bins=encoder_chunk_bins,
            encoder_checkpoint=bool(train_args.get("encoder_checkpoint", True)),
            bp_downsample_strides=bp_downsample_strides,
            bp_stem_type=train_args.get("bp_stem_type", "strided_conv"),
            bp_stem_kernels=bp_stem_kernels,
        )
    else:
        model = CrossCompartmentFusionRangePredictor(
            d_model=int(train_args.get("d_model", 128)),
            block_size=int(train_args.get("input_length", args.input_length)),
            depth=int(train_args.get("depth", 2)),
            num_heads=int(train_args.get("num_heads", 8)),
            dropout=float(train_args.get("dropout", 0.1)),
        )
    model.load_state_dict(ckpt["model"])
    return model, train_args


def _chrom_len(path: str, chrom: str) -> int | None:
    with pyBigWig.open(path) as bw:
        chroms = bw.chroms()
    if chrom in chroms:
        return int(chroms[chrom])
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in chroms:
        return int(chroms[alt])
    return None


def filter_safe_bigwig_ranges(dataset, plus_bigwig: str, minus_bigwig: str, target_bigwig: str) -> None:
    kept = []
    dropped = 0
    cache: dict[str, int] = {}
    for chrom, start, end in dataset.ranges:
        if chrom not in cache:
            lengths = [
                _chrom_len(plus_bigwig, chrom),
                _chrom_len(minus_bigwig, chrom),
                _chrom_len(target_bigwig, chrom),
            ]
            lengths = [x for x in lengths if x is not None]
            cache[chrom] = min(lengths) if lengths else 0
        if end <= cache[chrom]:
            kept.append((chrom, start, end))
        else:
            dropped += 1
    if dropped:
        print(f"dropped {dropped} ranges beyond shared bigWig chromosome length", flush=True)
    dataset.ranges = kept


def main():
    args = parse_args()
    model, train_args = build_model(args.checkpoint, args)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    input_mode = train_args.get("input_mode", "per-bin")
    if input_mode == "bp":
        per_range_zscore = args.per_range_zscore
        if per_range_zscore is None:
            per_range_zscore = bool(train_args.get("per_range_zscore", False))
        dataset = BPLevelRangeCompartmentBigWigFusionDataset(
            fasta_path=args.fasta,
            plus_bigwig=args.plus_bigwig,
            minus_bigwig=args.minus_bigwig,
            target_bigwig=args.target_bigwig,
            chromosomes=parse_chroms(args.chroms),
            range_size=args.range_size,
            bin_size=args.bin_size,
            stride=args.stride,
            max_ranges=args.max_ranges,
            target_threshold=args.target_threshold,
            per_range_zscore=per_range_zscore,
        )
    else:
        track_window = args.track_window if args.track_window is not None else train_args.get("track_window", "center")
        per_window_zscore = args.per_window_zscore
        if per_window_zscore is None:
            per_window_zscore = bool(train_args.get("per_window_zscore", True))
        dataset = RangeCompartmentBigWigFusionDataset(
            fasta_path=args.fasta,
            plus_bigwig=args.plus_bigwig,
            minus_bigwig=args.minus_bigwig,
            target_bigwig=args.target_bigwig,
            chromosomes=parse_chroms(args.chroms),
            range_size=args.range_size,
            bin_size=args.bin_size,
            input_length=args.input_length,
            stride=args.stride,
            max_ranges=args.max_ranges,
            target_threshold=args.target_threshold,
            per_window_zscore=per_window_zscore,
            track_window=track_window,
        )

    filter_safe_bigwig_ranges(dataset, args.plus_bigwig, args.minus_bigwig, args.target_bigwig)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chrom",
        "start",
        "end",
        "range_start",
        "range_end",
        "target_score",
        "target_ab",
        "pred_score",
        "pred_ab",
    ]
    n_ranges = 0
    n_rows = 0
    mode = "a" if args.append else "w"
    write_header = not args.append or not out.exists() or out.stat().st_size == 0
    with out.open(mode, newline="", encoding="utf-8") as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        for seq, tracks, score, _cls, regions in loader:
            pred = model(seq.to(device), tracks.to(device)).detach().cpu()
            for batch_idx, region in enumerate(regions):
                chrom, range_start, range_end = region
                n_ranges += 1
                for bin_idx in range(pred.shape[1]):
                    start = int(range_start) + bin_idx * args.bin_size
                    end = start + args.bin_size
                    target = float(score[batch_idx, bin_idx])
                    value = float(pred[batch_idx, bin_idx])
                    writer.writerow(
                        {
                            "chrom": chrom,
                            "start": start,
                            "end": end,
                            "range_start": int(range_start),
                            "range_end": int(range_end),
                            "target_score": target,
                            "target_ab": "A" if target >= args.target_threshold else "B",
                            "pred_score": value,
                            "pred_ab": "A" if value >= args.target_threshold else "B",
                        }
                    )
                    n_rows += 1
            if args.log_every_ranges > 0 and n_ranges % args.log_every_ranges == 0:
                print(f"processed {n_ranges}/{len(dataset)} ranges; wrote {n_rows} rows", flush=True)
    print(f"wrote {n_rows} bins from {n_ranges} ranges to {out}")


if __name__ == "__main__":
    main()
