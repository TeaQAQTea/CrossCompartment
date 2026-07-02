#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-python}"
CKPT="${CKPT:?set CKPT to a trained checkpoint path}"
FASTA="${FASTA:?set FASTA to a reference genome FASTA path}"
PLUS_BW="${PLUS_BW:?set PLUS_BW to a plus-strand signal bigWig path}"
MINUS_BW="${MINUS_BW:?set MINUS_BW to a minus-strand signal bigWig path}"
TARGET_BW="${TARGET_BW:?set TARGET_BW to a compartment target bigWig path}"
CELL="${CELL:-Sample}"
OUT_DIR="${OUT_DIR:?set OUT_DIR to a prediction output directory}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RANGE_SIZE="${RANGE_SIZE:-1650000}"
BIN_SIZE="${BIN_SIZE:-2200}"
STRIDE="${STRIDE:-550000}"
CENTER_SIZE="${CENTER_SIZE:-1100000}"
TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.0}"
CHROMS="${CHROMS:-chr8}"
PREDICTION_SET="${PREDICTION_SET:-predictions}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${OUT_DIR}/compartment_center1p1mb_100kb}"

mkdir -p "${OUT_DIR}"
output="${OUT_DIR}/${CELL}_${PREDICTION_SET}_test_predictions.tsv"
rm -f "${output}"

IFS=',' read -r -a chrom_array <<< "${CHROMS}"
for chrom in "${chrom_array[@]}"; do
  echo "[predict] ${CELL} ${chrom}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/predict_ranges.py \
    --checkpoint "${CKPT}" \
    --fasta "${FASTA}" \
    --plus-bigwig "${PLUS_BW}" \
    --minus-bigwig "${MINUS_BW}" \
    --target-bigwig "${TARGET_BW}" \
    --chroms "${chrom}" \
    --output "${output}" \
    --range-size "${RANGE_SIZE}" \
    --bin-size "${BIN_SIZE}" \
    --stride "${STRIDE}" \
    --input-length 1024 \
    --target-threshold "${TARGET_THRESHOLD}" \
    --no-per-range-zscore \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --log-every-ranges 200 \
    --append
done

"${PYTHON}" scripts/aggregate_predictions.py \
  --prediction-dir "${OUT_DIR}" \
  --prediction-set "${PREDICTION_SET}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --cells "${CELL}" \
  --chroms "${CHROMS}" \
  --label-bw "${CELL}=${TARGET_BW}" \
  --center-size "${CENTER_SIZE}" \
  --bin-size 100000 \
  --threshold "${TARGET_THRESHOLD}"

echo "[done] ${OUT_DIR}"
