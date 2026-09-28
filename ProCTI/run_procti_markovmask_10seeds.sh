#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Full pipeline:
#   1) run the 5 ProCTI scripts for seeds 1..10
#
# Edit the paths in the "USER PATHS" section before running.
# =========================================================

# -----------------------------
# USER PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${SCRIPT_DIR}"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BEIJING_RUN_SCRIPT="${SCRIPT_DIR}/markovmask_procti_beijing.py"
GAIT_RUN_SCRIPT="${SCRIPT_DIR}/markovmask_procti_gait.py"
PHYSIONET_RUN_SCRIPT="${SCRIPT_DIR}/markovmask_procti_physionet.py"
STOCK_RUN_SCRIPT="${SCRIPT_DIR}/markovmask_procti_stock.py"
WEATHER_RUN_SCRIPT="${SCRIPT_DIR}/markovmask_procti_weather.py"

# Shared maskbank dirs
BEIJING_MASK_DIR="${ROOT_DIR}/maskbanks/beijing"
GAIT_MASK_DIR="${ROOT_DIR}/maskbanks/gait"
PHYSIONET_MASK_DIR="${ROOT_DIR}/maskbanks/physionet"
STOCK_MASK_DIR="${ROOT_DIR}/maskbanks/stock"
WEATHER_MASK_DIR="${ROOT_DIR}/maskbanks/weather"

# Data paths
BEIJING_SHARED_DATA_DIR="${ROOT_DIR}/data/beijing"
GAIT_DATA_DIR="${ROOT_DIR}/data/automatic-ou-gaitdata"
PHYSIONET_CSV="${ROOT_DIR}/data/physionet2019.csv"
STOCK_CSV="${ROOT_DIR}/data/stock_data.csv"
WEATHER_CSV="${ROOT_DIR}/data/weather.csv"

# Output root
OUT_ROOT="${SCRIPT_DIR}/procti_10seed_pipeline"

# Device
DEVICE="cuda:1"

# Seeds
SEEDS=(1 2 3 4 5 6 7 8 9 10)

# Common evaluation setup
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

mkdir -p "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}/metrics"
mkdir -p "${OUT_ROOT}/saved_arrays"


echo "===================================================="
echo "Step 1/1: Running ProCTI scripts for seeds 1..10"
echo "===================================================="

for SEED in "${SEEDS[@]}"; do
  echo "[run] seed=${SEED}"

  # -------------------------
  # Beijing
  # -------------------------
  python "${BEIJING_RUN_SCRIPT}" \
    --shared_data_dir "${BEIJING_SHARED_DATA_DIR}" \
    --code_dir "${CODE_DIR}" \
    --seq_len "${BEIJING_SEQ_LEN}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --n_samples "${N_SAMPLES}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${BEIJING_MASK_DIR}" \
    --save_test_arrays \
    --save_dir "${OUT_ROOT}/saved_arrays/beijing/seed${SEED}" \
    --out_txt "${OUT_ROOT}/metrics/beijing_seed${SEED}_metrics.tsv" \
    > "${OUT_ROOT}/logs/beijing_seed${SEED}.log" 2>&1

  # -------------------------
  # Gait
  # -------------------------
  python "${GAIT_RUN_SCRIPT}" \
    --data_dir "${GAIT_DATA_DIR}" \
    --code_dir "${CODE_DIR}" \
    --seq_len "${GAIT_SEQ_LEN}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --n_samples "${N_SAMPLES}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${GAIT_MASK_DIR}" \
    --save_test_arrays \
    --save_dir "${OUT_ROOT}/saved_arrays/gait/seed${SEED}" \
    --out_txt "${OUT_ROOT}/metrics/gait_seed${SEED}_metrics.tsv" \
    > "${OUT_ROOT}/logs/gait_seed${SEED}.log" 2>&1

  # -------------------------
  # PhysioNet
  # -------------------------
  python "${PHYSIONET_RUN_SCRIPT}" \
    --csv "${PHYSIONET_CSV}" \
    --code_dir "${CODE_DIR}" \
    --seq_len "${PHYSIONET_SEQ_LEN}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --n_samples "${N_SAMPLES}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${PHYSIONET_MASK_DIR}" \
    --save_test_arrays \
    --save_dir "${OUT_ROOT}/saved_arrays/physionet/seed${SEED}" \
    --out_txt "${OUT_ROOT}/metrics/physionet_seed${SEED}_metrics.tsv" \
    > "${OUT_ROOT}/logs/physionet_seed${SEED}.log" 2>&1

  # -------------------------
  # Stock
  # -------------------------
  python "${STOCK_RUN_SCRIPT}" \
    --csv "${STOCK_CSV}" \
    --code_dir "${CODE_DIR}" \
    --seq_len "${STOCK_SEQ_LEN}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --n_samples "${N_SAMPLES}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${STOCK_MASK_DIR}" \
    --save_test_arrays "${OUT_ROOT}/saved_arrays/stock/seed${SEED}_r0.30.npz" \
    --save_test_arrays_ratio 0.30 \
    --out_txt "${OUT_ROOT}/metrics/stock_seed${SEED}_metrics.tsv" \
    > "${OUT_ROOT}/logs/stock_seed${SEED}.log" 2>&1

  # -------------------------
  # Weather
  # -------------------------
  python "${WEATHER_RUN_SCRIPT}" \
    --csv "${WEATHER_CSV}" \
    --code_dir "${CODE_DIR}" \
    --seq_len "${WEATHER_SEQ_LEN}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --n_samples "${N_SAMPLES}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${WEATHER_MASK_DIR}" \
    --save_test_arrays \
    --save_dir "${OUT_ROOT}/saved_arrays/weather/seed${SEED}" \
    --out_txt "${OUT_ROOT}/metrics/weather_seed${SEED}_metrics.tsv" \
    > "${OUT_ROOT}/logs/weather_seed${SEED}.log" 2>&1

done

echo "===================================================="
echo "Done."
echo "Logs:      ${OUT_ROOT}/logs"
echo "Metrics:   ${OUT_ROOT}/metrics"
echo "Arrays:    ${OUT_ROOT}/saved_arrays"
echo "===================================================="
