#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Full pipeline:
#   1) generate shared maskbanks for seeds 1..10
#
# Edit the paths in the "USER PATHS" section before running.
# =========================================================

# -----------------------------
# USER PATHS
# -----------------------------
CODE_DIR="./ProCTI"

BEIJING_MASK_SCRIPT="make_beijing_maskbank.py"
GAIT_MASK_SCRIPT="make_gait_maskbank.py"
PHYSIONET_MASK_SCRIPT="make_physionet_maskbank.py"
STOCK_MASK_SCRIPT="make_stock_maskbank.py"
WEATHER_MASK_SCRIPT="make_weather_maskbank.py"

# Data paths
BEIJING_SHARED_DATA_DIR="/data/beijing"
GAIT_DATA_DIR="/data/automatic-ou-gaitdata"
PHYSIONET_CSV="/data/physionet2019.csv"
STOCK_CSV="/data/stock_data.csv"
WEATHER_CSV="/data/weather.csv"

# Output root
OUT_ROOT="./ProCTI"

# Device
DEVICE="cuda:0"

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

mkdir -p "${OUT_ROOT}/maskbanks"

BEIJING_MASK_DIR="${OUT_ROOT}/maskbanks/beijing"
GAIT_MASK_DIR="${OUT_ROOT}/maskbanks/gait"
PHYSIONET_MASK_DIR="${OUT_ROOT}/maskbanks/physionet"
STOCK_MASK_DIR="${OUT_ROOT}/maskbanks/stock"
WEATHER_MASK_DIR="${OUT_ROOT}/maskbanks/weather"

mkdir -p "${BEIJING_MASK_DIR}" "${GAIT_MASK_DIR}" "${PHYSIONET_MASK_DIR}" "${STOCK_MASK_DIR}" "${WEATHER_MASK_DIR}"

echo "===================================================="
echo "Generating maskbanks for seeds 1..10"
echo "===================================================="

for SEED in "${SEEDS[@]}"; do
  echo "[maskbank] seed=${SEED}"

  # Beijing
  python "${BEIJING_MASK_SCRIPT}" \
    --shared_data_dir "${BEIJING_SHARED_DATA_DIR}" \
    --seq_len "${BEIJING_SEQ_LEN}" \
    --lm "${LM}" \
    --eval_masked_ratios "${RATIOS}" \
    --seed "${SEED}" \
    --out_dir "${BEIJING_MASK_DIR}" \
    > "${OUT_ROOT}/logs/make_beijing_maskbank_seed${SEED}.log" 2>&1

  # Gait
  python "${GAIT_MASK_SCRIPT}" \
    --data_dir "${GAIT_DATA_DIR}" \
    --seq_len "${GAIT_SEQ_LEN}" \
    --lm "${LM}" \
    --eval_masked_ratios "${RATIOS}" \
    --seed "${SEED}" \
    --out_dir "${GAIT_MASK_DIR}" \
    > "${OUT_ROOT}/logs/make_gait_maskbank_seed${SEED}.log" 2>&1

  # PhysioNet
  python "${PHYSIONET_MASK_SCRIPT}" \
    --csv "${PHYSIONET_CSV}" \
    --seq_len "${PHYSIONET_SEQ_LEN}" \
    --lm "${LM}" \
    --eval_masked_ratios "${RATIOS}" \
    --seed "${SEED}" \
    --out_dir "${PHYSIONET_MASK_DIR}" \
    > "${OUT_ROOT}/logs/make_physionet_maskbank_seed${SEED}.log" 2>&1

  # Stock (IMPORTANT: include val masks)
  python "${STOCK_MASK_SCRIPT}" \
    --csv "${STOCK_CSV}" \
    --seq_len "${STOCK_SEQ_LEN}" \
    --lm "${LM}" \
    --eval_masked_ratios "${RATIOS}" \
    --seed "${SEED}" \
    --include_val \
    --out_dir "${STOCK_MASK_DIR}" \
    > "${OUT_ROOT}/logs/make_stock_maskbank_seed${SEED}.log" 2>&1

  # Weather (IMPORTANT: include val masks)
  python "${WEATHER_MASK_SCRIPT}" \
    --csv "${WEATHER_CSV}" \
    --seq_len "${WEATHER_SEQ_LEN}" \
    --lm "${LM}" \
    --eval_masked_ratios "${RATIOS}" \
    --seed "${SEED}" \
    --include_val \
    --out_dir "${WEATHER_MASK_DIR}" \
    > "${OUT_ROOT}/logs/make_weather_maskbank_seed${SEED}.log" 2>&1
done
