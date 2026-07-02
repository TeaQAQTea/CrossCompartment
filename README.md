# CrossCompartment

CrossCompartment predicts A/B compartment tracks from DNA sequence and paired strand-specific transcription signal tracks, such as GRO-seq, PRO-seq, or RO-seq plus/minus bigWig files.

The repository is intentionally small: it contains the model code, dataset loaders, training entrypoints, prediction scripts, and aggregation utilities needed to train a range-level compartment model and score predictions at 100 kb resolution.

## Environment

Use a Python environment with PyTorch and the scientific Python stack installed.

```bash
python -m pip install torch numpy pandas scipy scikit-learn pyBigWig pyfaidx tqdm
```

Required Python packages include `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyBigWig`, `pyfaidx`, and `tqdm`.

All commands below are run from the package directory.

```bash
cd <package-directory>
```

## Input Data

Training and prediction use four genomic inputs:

- `FASTA`: reference genome FASTA indexed with `.fai`
- `PLUS_BW`: plus-strand transcription signal bigWig
- `MINUS_BW`: minus-strand transcription signal bigWig
- `TARGET_BW`: compartment target bigWig, oriented so positive values correspond to A compartment and negative or zero values correspond to B compartment

For multi-cell training, provide a manifest TSV:

```tsv
cell	fasta	plus_bigwig	minus_bigwig	target_bigwig
CellA	${FASTA}	${CELL_A_PLUS_BW}	${CELL_A_MINUS_BW}	${CELL_A_TARGET_BW}
CellB	${FASTA}	${CELL_B_PLUS_BW}	${CELL_B_MINUS_BW}	${CELL_B_TARGET_BW}
```

## Prepare Data

The loaders read bigWig values directly. If upstream signal tracks are bedGraph-like files, prepare them before training:

```bash
sort -k1,1 -k2,2n plus.bedGraph > plus.sorted.bedGraph
sort -k1,1 -k2,2n minus.bedGraph > minus.sorted.bedGraph
bedGraphToBigWig plus.sorted.bedGraph chrom.sizes plus.bw
bedGraphToBigWig minus.sorted.bedGraph chrom.sizes minus.bw
```

Use chromosome sizes that match the FASTA index:

```bash
cut -f1,2 genome.fa.fai > chrom.sizes
```

The target compartment track should also be a bigWig. It should be oriented so positive values represent A compartment and negative or zero values represent B compartment. If your target is an E1/eigenvector track with arbitrary sign, orient it with an external signal such as ATAC-seq before training.

For multi-cell training, write the manifest after all bigWigs are prepared:

```bash
cat > "${MANIFEST}" <<'EOF'
cell	fasta	plus_bigwig	minus_bigwig	target_bigwig
SampleA	${FASTA}	${SAMPLE_A_PLUS_BW}	${SAMPLE_A_MINUS_BW}	${SAMPLE_A_TARGET_BW}
SampleB	${FASTA}	${SAMPLE_B_PLUS_BW}	${SAMPLE_B_MINUS_BW}	${SAMPLE_B_TARGET_BW}
EOF
```

Make sure chromosome naming is consistent across FASTA, signal bigWigs, target bigWigs, and command-line chromosome lists, for example `chr1` rather than `1`.

## Train

Single-cell range-level training:

```bash
python train_fusion_compartment_range.py \
  --fasta "${FASTA}" \
  --plus-bigwig "${PLUS_BW}" \
  --minus-bigwig "${MINUS_BW}" \
  --target-bigwig "${TARGET_BW}" \
  --output-dir "${MODEL_DIR}" \
  --train-chroms chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22 \
  --val-chroms chr8 \
  --range-size 1650000 \
  --bin-size 2200 \
  --stride 550000 \
  --target-threshold 0.0 \
  --batch-size 4 \
  --epochs 20
```

Multi-cell training:

```bash
python train_multicell_fusion_compartment_range.py \
  --manifest "${MANIFEST}" \
  --output-dir "${MODEL_DIR}" \
  --train-chroms chr1,chr2,chr3,chr4,chr5,chr7,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22 \
  --val-chroms chr6 \
  --input-mode bp \
  --range-size 1650000 \
  --bin-size 2200 \
  --stride 550000 \
  --target-threshold 0.0 \
  --batch-size 4 \
  --epochs 20
```

Training writes `last.pt`, best checkpoints, `metadata.json`, and training history files into `--output-dir`.

## Predict

Use the generic launch script for a checkpoint and a plus/minus bigWig pair:

```bash
PYTHON=python \
CKPT="${CKPT}" \
CELL=Sample \
FASTA="${FASTA}" \
PLUS_BW="${PLUS_BW}" \
MINUS_BW="${MINUS_BW}" \
TARGET_BW="${TARGET_BW}" \
OUT_DIR="${PREDICTION_DIR}" \
CHROMS=chr8 \
GPU=0 \
scripts/predict_compartment.sh
```

The script first streams per-bin predictions with `scripts/predict_ranges.py`, then aggregates predictions to 100 kb bins with `scripts/aggregate_predictions.py`.

For direct Python prediction without the shell wrapper:

```bash
python scripts/predict_ranges.py \
  --checkpoint "${CKPT}" \
  --fasta "${FASTA}" \
  --plus-bigwig "${PLUS_BW}" \
  --minus-bigwig "${MINUS_BW}" \
  --target-bigwig "${TARGET_BW}" \
  --chroms chr8 \
  --output "${PREDICTION_TSV}" \
  --range-size 1650000 \
  --bin-size 2200 \
  --stride 550000 \
  --target-threshold 0.0 \
  --batch-size 4
```

Aggregate streamed predictions:

```bash
python scripts/aggregate_predictions.py \
  --prediction-dir "${PREDICTION_DIR}" \
  --prediction-set predictions \
  --output-prefix "${AGGREGATE_PREFIX}" \
  --cells Sample \
  --chroms chr8 \
  --label-bw Sample="${TARGET_BW}" \
  --center-size 1100000 \
  --bin-size 100000 \
  --threshold 0.0
```

The aggregated outputs are:

- `*.per_100kb.tsv`: per-bin target and prediction scores
- `*.metrics.tsv`: AUROC, AUPR, Pearson, Spearman, and count summaries
