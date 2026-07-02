from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from src.ab_compartment.datasets import CellBigWigConfig, MultiCellBPLevelRangeCompartmentBigWigFusionDataset, MultiCellRangeCompartmentBigWigFusionDataset
from src.ab_compartment.models import BPLevelFusionRangePredictor, CrossCompartmentFusionRangePredictor
from train_fusion_compartment import parse_chroms
from train_fusion_compartment_range import collate


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in str(text).replace(";", ",").split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all strides must be positive")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-cell DNA+RO-seq 2Mb compartment predictor.")
    parser.add_argument("--manifest", required=True, help="TSV with cell,fasta,plus_bigwig,minus_bigwig,target_bigwig.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-chroms", default="chr1,chr2,chr3,chr4,chr5,chr7,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22")
    parser.add_argument("--val-chroms", default="chr6")
    parser.add_argument("--range-size", type=int, default=2_000_000)
    parser.add_argument("--bin-size", type=int, default=50_000)
    parser.add_argument("--stride", type=int, default=2_000_000)
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--track-window", choices=["center", "bin"], default="center")
    parser.add_argument("--per-window-zscore", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-mode", choices=["per-bin", "bp"], default="per-bin")
    parser.add_argument("--per-range-zscore", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablation", choices=["full", "dna_only", "ro_only", "no_strand_fusion", "no_modality_gate"], default="full")
    parser.add_argument("--fusion-method", choices=["gate", "concat", "simple_concat", "channel_concat", "cross_attention"], default="gate")
    parser.add_argument("--max-train-ranges-per-cell", type=int)
    parser.add_argument("--max-val-ranges-per-cell", type=int)
    parser.add_argument("--range-cache-dir", default=None, help="Directory for cached valid genomic range indices.")
    parser.add_argument("--validate-ranges", action=argparse.BooleanOptionalAction, default=True, help="Pre-check every target bin while building range indices. Disable for C.Origami-style fast startup.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when num_workers > 0.")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True, help="Pin host memory for faster CPU-to-CUDA transfer.")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True, help="Keep DataLoader workers alive across epochs when num_workers > 0.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs. The final epoch is always validated.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decoder-layers", type=int, default=0)
    parser.add_argument("--decoder-kernel-size", type=int, default=5)
    parser.add_argument("--decoder-type", choices=["conv", "transformer"], default="conv")
    parser.add_argument(
        "--encoder-chunk-size",
        type=int,
        default=None,
        help=(
            "For bp input, encode this many bp at a time, checkpoint each encoder chunk during training, "
            "then concatenate chunk features back along bins."
        ),
    )
    parser.add_argument("--encoder-checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Checkpoint each encoder chunk during training to save memory at the cost of recompute.")
    parser.add_argument("--bp-downsample-strides", type=parse_int_tuple, default=(10, 10, 10, 10, 5), help="Comma-separated strided-conv factors from bp to bins; product should equal bin size, e.g. 10,10,10,10 for 10kb or 10,10,10,10,5 for 50kb.")
    parser.add_argument("--bp-stem-kernels", type=parse_int_tuple, default=None, help="Comma-separated odd Conv1d kernel sizes for each bp stem stage; length must match --bp-downsample-strides.")
    parser.add_argument(
        "--bp-stem-type",
        choices=[
            "strided_conv",
            "conv_pool",
            "cross_conv_pool",
            "cross_conv_pool_lite",
            "cross_poolfirst_lite",
            "cross_convfirst_lite",
        ],
        default="strided_conv",
        help=(
            "BP-level stem: strided_conv does Conv1d with stride; conv_pool does Conv1d stride 1 "
            "followed by AvgPool1d; cross_conv_pool applies strand TokenBridge after each pooling stage; "
            "cross_conv_pool_lite applies a lighter gated strand interaction after each pooling stage; "
            "cross_poolfirst_lite pools before convolution to keep bp-level activation memory low; "
            "cross_convfirst_lite uses stride-1 convolution, light strand interaction, then pooling."
        ),
    )
    parser.add_argument("--supervised-center-size", type=int, default=None, help="Only compute train/val loss and metrics on the center bp span, e.g. 2000000 inside an 8Mb input.")
    parser.add_argument(
        "--loss-type",
        choices=["mse", "mse_sign", "bce"],
        default="mse_sign",
        help="Training loss. mse uses only regression MSE; mse_sign uses MSE plus optional sign BCE; bce uses only A/B sign BCE.",
    )
    parser.add_argument("--sign-loss-weight", type=float, default=0.0, help="Auxiliary BCE weight for the A/B sign target; 0 disables it.")
    parser.add_argument("--track-noise-std", type=float, default=0.0, help="Training-only Gaussian noise std added to RO-seq tracks after loading/z-scoring.")
    parser.add_argument("--track-channel-dropout", type=float, default=0.0, help="Training-only probability to zero each RO-seq strand channel per sample.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after this many epochs without val loss improvement; 0 disables early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Minimum val loss improvement required to reset early stopping.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_manifest(path: str) -> list[CellBigWigConfig]:
    frame = pd.read_csv(path, sep="\t", comment="#")
    required = {"cell", "fasta", "plus_bigwig", "minus_bigwig", "target_bigwig"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")
    configs = []
    for row in frame.itertuples(index=False):
        configs.append(
            CellBigWigConfig(
                cell=str(row.cell),
                fasta_path=str(row.fasta),
                plus_bigwig=str(row.plus_bigwig),
                minus_bigwig=str(row.minus_bigwig),
                target_bigwig=str(row.target_bigwig),
            )
        )
    if not configs:
        raise ValueError(f"No cells found in manifest: {path}")
    return configs


def write_history(history: list[dict], out_dir: Path) -> None:
    with open(out_dir / "history.json", "w") as handle:
        json.dump(history, handle, indent=2)
    with open(out_dir / "history.jsonl", "w") as handle:
        for record in history:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    pd.DataFrame(history).to_csv(out_dir / "history.tsv", sep="\t", index=False)




def augment_tracks(tracks: torch.Tensor, noise_std: float, channel_dropout: float) -> torch.Tensor:
    if noise_std <= 0 and channel_dropout <= 0:
        return tracks
    out = tracks
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    if channel_dropout > 0:
        if not 0 <= channel_dropout < 1:
            raise ValueError("--track-channel-dropout must be in [0, 1)")
        keep = (torch.rand(out.shape[0], 1, out.shape[-1], device=out.device) >= channel_dropout).to(out.dtype)
        out = out * keep
    return out

def center_bin_slice(n_bins: int, bin_size: int, center_size: int | None) -> slice:
    if center_size is None:
        return slice(None)
    if center_size <= 0:
        raise ValueError("--supervised-center-size must be positive when set")
    if center_size % bin_size != 0:
        raise ValueError("--supervised-center-size must be divisible by --bin-size")
    center_bins = center_size // bin_size
    if center_bins > n_bins:
        raise ValueError(f"center supervision uses {center_bins} bins but range has only {n_bins}")
    start = (n_bins - center_bins) // 2
    end = start + center_bins
    return slice(start, end)


def crop_center_bins(tensor: torch.Tensor, bin_slice: slice) -> torch.Tensor:
    return tensor[:, bin_slice]

def compute_losses(
    pred: torch.Tensor,
    score: torch.Tensor,
    cls: torch.Tensor,
    threshold: float,
    loss_type: str,
    sign_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reg_loss = F.mse_loss(pred, score)
    sign_loss = F.binary_cross_entropy_with_logits(pred - threshold, cls)
    if loss_type == "mse":
        return reg_loss, reg_loss, pred.new_zeros(())
    if loss_type == "bce":
        return sign_loss, reg_loss, sign_loss
    if loss_type == "mse_sign":
        if sign_loss_weight > 0:
            return reg_loss + sign_loss_weight * sign_loss, reg_loss, sign_loss
        return reg_loss, reg_loss, pred.new_zeros(())
    raise ValueError(f"Unsupported loss type: {loss_type}")


def evaluate_center(model, loader, device, threshold: float, bin_slice: slice, loss_type: str, sign_loss_weight: float) -> dict:
    model.eval()
    losses, mse_losses, sign_losses, preds, targets, labels = [], [], [], [], [], []
    with torch.inference_mode():
        for seq, tracks, score, cls, _region in loader:
            seq = seq.to(device)
            tracks = tracks.to(device)
            score = crop_center_bins(score.to(device), bin_slice)
            cls_center = crop_center_bins(cls.to(device).float(), bin_slice)
            pred = crop_center_bins(model(seq, tracks), bin_slice)
            loss, mse_loss, sign_loss = compute_losses(pred, score, cls_center, threshold, loss_type, sign_loss_weight)
            losses.append(float(loss.detach().cpu()))
            mse_losses.append(float(mse_loss.detach().cpu()))
            sign_losses.append(float(sign_loss.detach().cpu()))
            preds.extend(pred.detach().cpu().reshape(-1).numpy().tolist())
            targets.extend(score.detach().cpu().reshape(-1).numpy().tolist())
            labels.extend(cls_center.detach().cpu().reshape(-1).numpy().tolist())
    preds_np = np.asarray(preds, dtype=np.float32)
    targets_np = np.asarray(targets, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)
    pred_cls = (preds_np >= threshold).astype(int)
    out = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mse_loss": float(np.mean(mse_losses)) if mse_losses else float("nan"),
        "sign_loss": float(np.mean(sign_losses)) if sign_losses else float("nan"),
        "mse": float(mean_squared_error(targets_np, preds_np)) if len(targets_np) else float("nan"),
        "mae": float(mean_absolute_error(targets_np, preds_np)) if len(targets_np) else float("nan"),
        "sign_accuracy": float(accuracy_score(labels_np, pred_cls)) if len(labels_np) else float("nan"),
    }
    if len(targets_np) > 1 and np.std(targets_np) > 0 and np.std(preds_np) > 0:
        out["pearson"] = float(np.corrcoef(targets_np, preds_np)[0, 1])
        out["spearman"] = float(pd.Series(targets_np).corr(pd.Series(preds_np), method="spearman"))
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
    out["auroc"] = float(roc_auc_score(labels_np, preds_np)) if len(set(labels_np.tolist())) == 2 else float("nan")
    return out

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = load_manifest(args.manifest)
    if args.input_mode == "bp":
        dataset_cls = MultiCellBPLevelRangeCompartmentBigWigFusionDataset
        common_kwargs = {
            "configs": configs,
            "range_size": args.range_size,
            "bin_size": args.bin_size,
            "stride": args.stride,
            "target_threshold": args.target_threshold,
            "per_range_zscore": args.per_range_zscore,
            "range_cache_dir": args.range_cache_dir,
            "validate_ranges": args.validate_ranges,
        }
        train_ds = dataset_cls(chromosomes=parse_chroms(args.train_chroms), max_ranges_per_cell=args.max_train_ranges_per_cell, **common_kwargs)
        val_ds = dataset_cls(chromosomes=parse_chroms(args.val_chroms), max_ranges_per_cell=args.max_val_ranges_per_cell, **common_kwargs)
    else:
        dataset_cls = MultiCellRangeCompartmentBigWigFusionDataset
        common_kwargs = {
            "configs": configs,
            "range_size": args.range_size,
            "bin_size": args.bin_size,
            "input_length": args.input_length,
            "stride": args.stride,
            "target_threshold": args.target_threshold,
            "per_window_zscore": args.per_window_zscore,
            "track_window": args.track_window,
            "range_cache_dir": args.range_cache_dir,
            "validate_ranges": args.validate_ranges,
        }
        train_ds = dataset_cls(chromosomes=parse_chroms(args.train_chroms), max_ranges_per_cell=args.max_train_ranges_per_cell, **common_kwargs)
        val_ds = dataset_cls(chromosomes=parse_chroms(args.val_chroms), max_ranges_per_cell=args.max_val_ranges_per_cell, **common_kwargs)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset: train={len(train_ds)} val={len(val_ds)}")
    downsample_factor = int(np.prod(args.bp_downsample_strides))
    if args.input_mode == "bp" and downsample_factor != args.bin_size:
        raise ValueError(
            f"--bp-downsample-strides product ({downsample_factor}) must equal --bin-size ({args.bin_size})"
        )
    bin_slice = center_bin_slice(train_ds.n_bins, args.bin_size, args.supervised_center_size)
    supervised_bins = len(range(train_ds.n_bins)[bin_slice])
    device = torch.device(args.device)

    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
        "persistent_workers": bool(args.persistent_workers and args.num_workers > 0),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    encoder_chunk_bins = None
    if args.encoder_chunk_size is not None:
        if args.input_mode != "bp":
            raise ValueError("--encoder-chunk-size is only supported for --input-mode bp")
        if args.encoder_chunk_size % args.bin_size != 0:
            raise ValueError("--encoder-chunk-size must be divisible by --bin-size")
        if args.range_size % args.encoder_chunk_size != 0:
            raise ValueError("--range-size must be divisible by --encoder-chunk-size")
        encoder_chunk_bins = args.encoder_chunk_size // args.bin_size

    if args.input_mode == "bp":
        model = BPLevelFusionRangePredictor(
            d_model=args.d_model,
            n_bins=train_ds.n_bins,
            depth=args.depth,
            num_heads=args.num_heads,
            dropout=args.dropout,
            ablation=args.ablation,
            fusion_method=args.fusion_method,
            decoder_layers=args.decoder_layers,
            decoder_kernel_size=args.decoder_kernel_size,
            decoder_type=args.decoder_type,
            encoder_chunk_bins=encoder_chunk_bins,
            encoder_checkpoint=args.encoder_checkpoint,
            bp_downsample_strides=args.bp_downsample_strides,
            bp_stem_type=args.bp_stem_type,
            bp_stem_kernels=args.bp_stem_kernels,
        ).to(device)
    else:
        model = CrossCompartmentFusionRangePredictor(
            d_model=args.d_model,
            block_size=args.input_length,
            depth=args.depth,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metadata = {
        "args": vars(args),
        "manifest": [cfg.__dict__ for cfg in configs],
        "train_summary": train_ds.summary(),
        "val_summary": val_ds.summary(),
        "n_train_ranges": len(train_ds),
        "n_val_ranges": len(val_ds),
        "n_bins_per_range": train_ds.n_bins,
        "supervised_bin_start": bin_slice.start,
        "supervised_bin_stop": bin_slice.stop,
        "n_supervised_bins": supervised_bins,
    }
    with open(out_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    best_loss = float("inf")
    best_auroc = -float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_reg_losses = []
        train_sign_losses = []
        epoch_start = time.time()
        print(
            json.dumps(
                {
                    "event": "epoch_start",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "steps_per_epoch": len(train_loader),
                    "n_train_ranges": len(train_ds),
                    "n_val_ranges": len(val_ds),
                    "n_supervised_bins": supervised_bins,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
            disable=not args.progress_bar,
        )
        for step, (seq, tracks, score, cls, _region) in enumerate(progress, start=1):
            seq = seq.to(device)
            tracks = augment_tracks(tracks.to(device), args.track_noise_std, args.track_channel_dropout)
            score = score.to(device)
            cls = cls.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            pred = crop_center_bins(model(seq, tracks), bin_slice)
            score = crop_center_bins(score, bin_slice)
            cls = crop_center_bins(cls, bin_slice)
            loss, reg_loss, sign_loss = compute_losses(
                pred,
                score,
                cls,
                args.target_threshold,
                args.loss_type,
                args.sign_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_reg_losses.append(float(reg_loss.detach().cpu()))
            train_sign_losses.append(float(sign_loss.detach().cpu()))
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == len(train_loader)):
                elapsed = time.time() - epoch_start
                ranges_per_sec = step / elapsed if elapsed > 0 else float("nan")
                train_loss_recent = float(np.mean(train_losses[-min(len(train_losses), args.log_every) :]))
                if args.progress_bar:
                    progress.set_postfix(
                        loss=f"{train_loss_recent:.4g}",
                        mse=f"{np.mean(train_reg_losses[-min(len(train_reg_losses), args.log_every) :]):.4g}",
                        bce=f"{np.mean(train_sign_losses[-min(len(train_sign_losses), args.log_every) :]):.4g}",
                        speed=f"{ranges_per_sec:.2f}r/s",
                        lr=f"{args.lr:.1e}",
                        d=args.d_model,
                        depth=args.depth,
                        heads=args.num_heads,
                        train=len(train_ds),
                        val=len(val_ds),
                    )
                else:
                    print(
                        json.dumps(
                            {
                                "event": "train_progress",
                                "epoch": epoch,
                                "step": step,
                                "steps_per_epoch": len(train_loader),
                                "progress": float(step / len(train_loader)),
                                "elapsed_sec": float(elapsed),
                                "ranges_per_sec": float(ranges_per_sec),
                                "train_loss_recent": train_loss_recent,
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )
        progress.close()
        should_validate = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_mse_loss": float(np.mean(train_reg_losses)),
            "train_sign_loss": float(np.mean(train_sign_losses)),
        }
        val_metrics = None
        if should_validate:
            print(
                json.dumps(
                    {
                        "event": "validation_start",
                        "epoch": epoch,
                        "n_val_ranges": len(val_ds),
                        "elapsed_train_sec": float(time.time() - epoch_start),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            val_metrics = evaluate_center(
                model,
                val_loader,
                device,
                args.target_threshold,
                bin_slice,
                args.loss_type,
                args.sign_loss_weight,
            )
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
        else:
            print(
                json.dumps(
                    {
                        "event": "validation_skipped",
                        "epoch": epoch,
                        "eval_every": args.eval_every,
                        "elapsed_train_sec": float(time.time() - epoch_start),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        history.append(record)
        write_history(history, out_dir)
        print(json.dumps({"event": "epoch_end", **record, "elapsed_total_sec": float(time.time() - epoch_start)}, ensure_ascii=True), flush=True)
        state = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "metrics": record, "manifest": [cfg.__dict__ for cfg in configs]}
        torch.save(state, out_dir / "last.pt")
        if val_metrics is not None:
            improved = val_metrics["loss"] < (best_loss - args.early_stopping_min_delta)
            if improved:
                best_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(state, out_dir / "best.pt")
                torch.save(state, out_dir / "best_loss.pt")
            else:
                epochs_without_improvement += 1
            val_auroc = float(val_metrics.get("auroc", float("nan")))
            if np.isfinite(val_auroc) and val_auroc > best_auroc:
                best_auroc = val_auroc
                torch.save(state, out_dir / "best_auroc.pt")

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                            "best_val_loss": float(best_loss),
                            "best_val_auroc": float(best_auroc),
                            "epochs_without_improvement": epochs_without_improvement,
                            "patience": args.early_stopping_patience,
                            "min_delta": args.early_stopping_min_delta,
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                break


if __name__ == "__main__":
    main()
