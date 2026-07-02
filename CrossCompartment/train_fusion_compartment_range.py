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

from src.ab_compartment.datasets import RangeCompartmentBigWigFusionDataset
from src.ab_compartment.models import CrossCompartmentFusionRangePredictor
from train_fusion_compartment import parse_chroms


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2Mb-range DNA+RO-seq fusion compartment predictor.")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--plus-bigwig", required=True)
    parser.add_argument("--minus-bigwig", required=True)
    parser.add_argument("--target-bigwig", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-chroms", default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22")
    parser.add_argument("--val-chroms", default="chr8")
    parser.add_argument("--range-size", type=int, default=2_000_000)
    parser.add_argument("--bin-size", type=int, default=50_000)
    parser.add_argument("--stride", type=int, default=2_000_000)
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--track-window", choices=["center", "bin"], default="center")
    parser.add_argument("--per-window-zscore", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-ranges", type=int)
    parser.add_argument("--max-val-ranges", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["mse", "huber", "mse_corr", "huber_corr"], default="mse")
    parser.add_argument("--corr-weight", type=float, default=0.1)
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def collate(batch):
    seq, tracks, score, cls, region = zip(*batch)
    return torch.stack(seq), torch.stack(tracks), torch.stack(score), torch.stack(cls), list(region)


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm()
    if float(denom.detach().cpu()) == 0.0:
        return pred.new_zeros(())
    return 1.0 - (pred * target).sum() / denom.clamp_min(1e-8)


def regression_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str, corr_weight: float, huber_delta: float) -> torch.Tensor:
    if loss_name in {"mse", "mse_corr"}:
        loss = F.mse_loss(pred, target)
    elif loss_name in {"huber", "huber_corr"}:
        loss = F.huber_loss(pred, target, delta=huber_delta)
    else:
        raise ValueError(f"Unknown regression loss: {loss_name}")
    if loss_name.endswith("_corr") and corr_weight > 0:
        loss = loss + corr_weight * correlation_loss(pred, target)
    return loss


def evaluate(model, loader, device, threshold):
    model.eval()
    losses, preds, targets, labels = [], [], [], []
    with torch.inference_mode():
        for seq, tracks, score, cls, _region in loader:
            seq = seq.to(device)
            tracks = tracks.to(device)
            score = score.to(device)
            pred = model(seq, tracks)
            loss = F.mse_loss(pred, score)
            losses.append(float(loss.detach().cpu()))
            preds.extend(pred.detach().cpu().reshape(-1).numpy().tolist())
            targets.extend(score.detach().cpu().reshape(-1).numpy().tolist())
            labels.extend(cls.reshape(-1).numpy().tolist())
    preds_np = np.asarray(preds, dtype=np.float32)
    targets_np = np.asarray(targets, dtype=np.float32)
    pred_cls = (preds_np >= threshold).astype(int)
    out = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mse": float(mean_squared_error(targets_np, preds_np)) if len(targets_np) else float("nan"),
        "mae": float(mean_absolute_error(targets_np, preds_np)) if len(targets_np) else float("nan"),
        "sign_accuracy": float(accuracy_score(labels, pred_cls)) if labels else float("nan"),
    }
    if len(targets_np) > 1 and np.std(targets_np) > 0 and np.std(preds_np) > 0:
        out["pearson"] = float(np.corrcoef(targets_np, preds_np)[0, 1])
        out["spearman"] = float(pd.Series(targets_np).corr(pd.Series(preds_np), method="spearman"))
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
    out["auroc"] = float(roc_auc_score(labels, preds_np)) if len(set(labels)) == 2 else float("nan")
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RangeCompartmentBigWigFusionDataset(
        fasta_path=args.fasta,
        plus_bigwig=args.plus_bigwig,
        minus_bigwig=args.minus_bigwig,
        target_bigwig=args.target_bigwig,
        chromosomes=parse_chroms(args.train_chroms),
        range_size=args.range_size,
        bin_size=args.bin_size,
        input_length=args.input_length,
        stride=args.stride,
        max_ranges=args.max_train_ranges,
        target_threshold=args.target_threshold,
        per_window_zscore=args.per_window_zscore,
        track_window=args.track_window,
    )
    val_ds = RangeCompartmentBigWigFusionDataset(
        fasta_path=args.fasta,
        plus_bigwig=args.plus_bigwig,
        minus_bigwig=args.minus_bigwig,
        target_bigwig=args.target_bigwig,
        chromosomes=parse_chroms(args.val_chroms),
        range_size=args.range_size,
        bin_size=args.bin_size,
        input_length=args.input_length,
        stride=args.stride,
        max_ranges=args.max_val_ranges,
        target_threshold=args.target_threshold,
        per_window_zscore=args.per_window_zscore,
        track_window=args.track_window,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    device = torch.device(args.device)
    model = CrossCompartmentFusionRangePredictor(
        d_model=args.d_model,
        block_size=args.input_length,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metadata = {"args": vars(args), "n_train_ranges": len(train_ds), "n_val_ranges": len(val_ds), "n_bins_per_range": train_ds.n_bins}
    with open(out_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
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
        for step, (seq, tracks, score, _cls, _region) in enumerate(progress, start=1):
            seq = seq.to(device)
            tracks = tracks.to(device)
            score = score.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(seq, tracks)
            loss = regression_loss(pred, score, args.loss, args.corr_weight, args.huber_delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == len(train_loader)):
                elapsed = time.time() - epoch_start
                ranges_per_sec = step / elapsed if elapsed > 0 else float("nan")
                train_loss_recent = float(np.mean(train_losses[-min(len(train_losses), args.log_every) :]))
                if args.progress_bar:
                    progress.set_postfix(
                        loss=f"{train_loss_recent:.4g}",
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
        val_metrics = evaluate(model, val_loader, device, args.target_threshold)
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(json.dumps({"event": "epoch_end", **record, "elapsed_total_sec": float(time.time() - epoch_start)}, ensure_ascii=True), flush=True)
        state = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "metrics": record}
        torch.save(state, out_dir / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(state, out_dir / "best.pt")

    with open(out_dir / "history.json", "w") as handle:
        json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()
