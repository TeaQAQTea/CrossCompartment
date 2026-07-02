from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

from src.ab_compartment.datasets import CompartmentBigWigFusionDataset
from src.ab_compartment.models import CrossCompartmentFusionCompartmentPredictor


def parse_chroms(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse DNA sequence and RO-seq strands to predict AB compartment score.")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--plus-bigwig", required=True)
    parser.add_argument("--minus-bigwig", required=True)
    parser.add_argument("--target-bigwig", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-chroms", default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22")
    parser.add_argument("--val-chroms", default="chr8")
    parser.add_argument("--bin-size", type=int, default=50000)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--max-train-bins", type=int)
    parser.add_argument("--max-val-bins", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collate(batch):
    seq, tracks, score, cls = zip(*batch)
    return torch.stack(seq), torch.stack(tracks), torch.stack(score), torch.stack(cls)


def evaluate(model, loader, device, threshold):
    model.eval()
    losses, preds, targets, labels = [], [], [], []
    with torch.inference_mode():
        for seq, tracks, score, cls in loader:
            seq = seq.to(device)
            tracks = tracks.to(device)
            score = score.to(device)
            pred = model(seq, tracks)
            loss = F.mse_loss(pred, score)
            losses.append(float(loss.detach().cpu()))
            preds.extend(pred.detach().cpu().numpy().tolist())
            targets.extend(score.detach().cpu().numpy().tolist())
            labels.extend(cls.numpy().tolist())

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
    else:
        out["pearson"] = float("nan")
    if len(set(labels)) == 2:
        out["auroc"] = float(roc_auc_score(labels, preds_np))
    else:
        out["auroc"] = float("nan")
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CompartmentBigWigFusionDataset(
        fasta_path=args.fasta,
        plus_bigwig=args.plus_bigwig,
        minus_bigwig=args.minus_bigwig,
        target_bigwig=args.target_bigwig,
        chromosomes=parse_chroms(args.train_chroms),
        bin_size=args.bin_size,
        input_length=args.input_length,
        max_bins=args.max_train_bins,
        target_threshold=args.target_threshold,
    )
    val_ds = CompartmentBigWigFusionDataset(
        fasta_path=args.fasta,
        plus_bigwig=args.plus_bigwig,
        minus_bigwig=args.minus_bigwig,
        target_bigwig=args.target_bigwig,
        chromosomes=parse_chroms(args.val_chroms),
        bin_size=args.bin_size,
        input_length=args.input_length,
        max_bins=args.max_val_bins,
        target_threshold=args.target_threshold,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    device = torch.device(args.device)
    model = CrossCompartmentFusionCompartmentPredictor(
        d_model=args.d_model,
        block_size=args.input_length,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        output="regression",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    history = []
    metadata = {
        "args": vars(args),
        "n_train_bins": len(train_ds),
        "n_val_bins": len(val_ds),
    }
    with open(out_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for seq, tracks, score, cls in train_loader:
            seq = seq.to(device)
            tracks = tracks.to(device)
            score = score.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(seq, tracks)
            loss = F.mse_loss(pred, score)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_loader, device, args.target_threshold)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=True))

        state = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "metrics": record}
        torch.save(state, out_dir / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(state, out_dir / "best.pt")

    with open(out_dir / "history.json", "w") as handle:
        json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()
