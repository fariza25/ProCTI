#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Run ProCTI ablation experiments
#
# This script:
#   - runs ablation variants A,B,C,D,E
#   - across seeds 1..10
#   - for Beijing, Gait, PhysioNet, Stock, Weather
#   - using shared evaluation maskbanks
#
# Edit the USER PATHS section before running.
# =========================================================

# -----------------------------
# USER PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${SCRIPT_DIR}"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BEIJING_SCRIPT="${SCRIPT_DIR}/ablation_procti_beijing.py"
GAIT_SCRIPT="${SCRIPT_DIR}/ablation_procti_gait.py"
PHYSIONET_SCRIPT="${SCRIPT_DIR}/ablation_procti_physionet.py"
STOCK_SCRIPT="${SCRIPT_DIR}/ablation_procti_stock.py"
WEATHER_SCRIPT="${SCRIPT_DIR}/ablation_procti_weather.py"

# Data paths
BEIJING_SHARED_DATA_DIR="${ROOT_DIR}/data/beijing"
GAIT_DATA_DIR="${ROOT_DIR}/data/automatic-ou-gaitdata"
PHYSIONET_CSV="${ROOT_DIR}/data/physionet2019.csv"
STOCK_CSV="${ROOT_DIR}/data/stock_data.csv"
WEATHER_CSV="${ROOT_DIR}/data/weather.csv"

# Shared maskbank dirs
BEIJING_MASK_DIR="${ROOT_DIR}/maskbanks/beijing"
GAIT_MASK_DIR="${ROOT_DIR}/maskbanks/gait"
PHYSIONET_MASK_DIR="${ROOT_DIR}/maskbanks/physionet"
STOCK_MASK_DIR="${ROOT_DIR}/maskbanks/stock"
WEATHER_MASK_DIR="${ROOT_DIR}/maskbanks/weather"

# Output root
OUT_ROOT="${SCRIPT_DIR}/procti_ablation_runs"

mkdir -p "${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}/metrics"
mkdir -p "${OUT_ROOT}/saved_arrays"
mkdir -p "${OUT_ROOT}/saved_arrays/beijing"
mkdir -p "${OUT_ROOT}/saved_arrays/gait"
mkdir -p "${OUT_ROOT}/saved_arrays/physionet"
mkdir -p "${OUT_ROOT}/saved_arrays/weather"

# Device
DEVICE="cuda:1"

# Seeds and variants
SEEDS=(1 2 3 4 5 6 7 8 9 10)
ABLATIONS=(A B C D E)

# Common settings
LM=6.0
RATIOS="0.10,0.30,0.50,0.70"
N_SAMPLES=20
EPOCHS=50

# Sequence lengths
BEIJING_SEQ_LEN=96
GAIT_SEQ_LEN=128
PHYSIONET_SEQ_LEN=96
STOCK_SEQ_LEN=48
WEATHER_SEQ_LEN=96

# Batch sizes
BEIJING_BATCH=64
GAIT_BATCH=64
PHYSIONET_BATCH=8
STOCK_BATCH=64
WEATHER_BATCH=64

mkdir -p "${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}/metrics"
mkdir -p "${OUT_ROOT}/saved_arrays"

echo "===================================================="
echo "Running ProCTI ablation experiments"
echo "===================================================="

for ABL in "${ABLATIONS[@]}"; do
  echo "------------------------------"
  echo "Ablation ${ABL}"
  echo "------------------------------"

  for SEED in "${SEEDS[@]}"; do
    echo "[run] ablation=${ABL} seed=${SEED}"

    python "${BEIJING_SCRIPT}" \
      --ablation "${ABL}" \
      --shared_data_dir "${BEIJING_SHARED_DATA_DIR}" \
      --code_dir "${CODE_DIR}" \
      --seq_len "${BEIJING_SEQ_LEN}" \
      --batch "${BEIJING_BATCH}" \
      --epochs "${EPOCHS}" \
      --lm "${LM}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --n_samples "${N_SAMPLES}" \
      --eval_masked_ratios "${RATIOS}" \
      --use_shared_evalmask \
      --shared_evalmask_dir "${BEIJING_MASK_DIR}" \
      --save_test_arrays \
      --save_dir "${OUT_ROOT}/saved_arrays/beijing/${ABL}/seed${SEED}" \
      --out_txt "${OUT_ROOT}/metrics/beijing_${ABL}_seed${SEED}_metrics.tsv" \
      > "${OUT_ROOT}/logs/beijing_${ABL}_seed${SEED}.log" 2>&1

    python "${GAIT_SCRIPT}" \
      --ablation "${ABL}" \
      --data_dir "${GAIT_DATA_DIR}" \
      --code_dir "${CODE_DIR}" \
      --seq_len "${GAIT_SEQ_LEN}" \
      --batch "${GAIT_BATCH}" \
      --epochs "${EPOCHS}" \
      --lm "${LM}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --n_samples "${N_SAMPLES}" \
      --eval_masked_ratios "${RATIOS}" \
      --use_shared_evalmask \
      --shared_evalmask_dir "${GAIT_MASK_DIR}" \
      --save_test_arrays \
      --save_dir "${OUT_ROOT}/saved_arrays/gait/${ABL}/seed${SEED}" \
      --out_txt "${OUT_ROOT}/metrics/gait_${ABL}_seed${SEED}_metrics.tsv" \
      > "${OUT_ROOT}/logs/gait_${ABL}_seed${SEED}.log" 2>&1

    python "${PHYSIONET_SCRIPT}" \
      --ablation "${ABL}" \
      --csv "${PHYSIONET_CSV}" \
      --code_dir "${CODE_DIR}" \
      --seq_len "${PHYSIONET_SEQ_LEN}" \
      --batch "${PHYSIONET_BATCH}" \
      --epochs "${EPOCHS}" \
      --lm "${LM}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --n_samples "${N_SAMPLES}" \
      --eval_masked_ratios "${RATIOS}" \
      --use_shared_evalmask \
      --shared_evalmask_dir "${PHYSIONET_MASK_DIR}" \
      --save_test_arrays \
      --save_dir "${OUT_ROOT}/saved_arrays/physionet/${ABL}/seed${SEED}" \
      --out_txt "${OUT_ROOT}/metrics/physionet_${ABL}_seed${SEED}_metrics.tsv" \
      > "${OUT_ROOT}/logs/physionet_${ABL}_seed${SEED}.log" 2>&1

    python "${STOCK_SCRIPT}" \
      --ablation "${ABL}" \
      --csv "${STOCK_CSV}" \
      --code_dir "${CODE_DIR}" \
      --seq_len "${STOCK_SEQ_LEN}" \
      --batch "${STOCK_BATCH}" \
      --epochs "${EPOCHS}" \
      --lm "${LM}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --n_samples "${N_SAMPLES}" \
      --eval_masked_ratios "${RATIOS}" \
      --use_shared_evalmask \
      --shared_evalmask_dir "${STOCK_MASK_DIR}" \
      --out_txt "${OUT_ROOT}/metrics/stock_${ABL}_seed${SEED}_metrics.tsv" \
      > "${OUT_ROOT}/logs/stock_${ABL}_seed${SEED}.log" 2>&1

    python "${WEATHER_SCRIPT}" \
      --ablation "${ABL}" \
      --csv "${WEATHER_CSV}" \
      --code_dir "${CODE_DIR}" \
      --seq_len "${WEATHER_SEQ_LEN}" \
      --batch "${WEATHER_BATCH}" \
      --epochs "${EPOCHS}" \
      --lm "${LM}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --n_samples "${N_SAMPLES}" \
      --eval_masked_ratios "${RATIOS}" \
      --use_shared_evalmask \
      --shared_evalmask_dir "${WEATHER_MASK_DIR}" \
      --save_test_arrays \
      --save_dir "${OUT_ROOT}/saved_arrays/weather/${ABL}/seed${SEED}" \
      --out_txt "${OUT_ROOT}/metrics/weather_${ABL}_seed${SEED}_metrics.tsv" \
      > "${OUT_ROOT}/logs/weather_${ABL}_seed${SEED}.log" 2>&1

  done
done

echo "===================================================="
echo "Done."
echo "Logs:    ${OUT_ROOT}/logs"
echo "Metrics: ${OUT_ROOT}/metrics"
echo "Arrays:  ${OUT_ROOT}/saved_arrays"
echo "===================================================="
