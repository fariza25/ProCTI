#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Run MTSCI Markov-mask and Channel-drop experiments
# for 10 seeds and save per-run + combined metrics.
# =========================================================

# -----------------------------
# USER PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${BASELINES_DIR}/.." && pwd)"

# Folder containing MTSCI codebase (models/, dataloader/, etc.)
MTSCI_ROOT="${SCRIPT_DIR}"

# Folder containing the experiment runner scripts
SCRIPT_DIR_RUN="${SCRIPT_DIR}"

# Markov scripts
BEIJING_MARKOV_SCRIPT="${SCRIPT_DIR_RUN}/markovmask_mtsci_beijing.py"
GAIT_MARKOV_SCRIPT="${SCRIPT_DIR_RUN}/markovmask_mtsci_gait.py"
PHYSIONET_MARKOV_SCRIPT="${SCRIPT_DIR_RUN}/markovmask_mtsci_physionet.py"
STOCK_MARKOV_SCRIPT="${SCRIPT_DIR_RUN}/markovmask_mtsci_stock.py"
WEATHER_MARKOV_SCRIPT="${SCRIPT_DIR_RUN}/markovmask_mtsci_weather.py"

# Channel-drop scripts
BEIJING_DROP_SCRIPT="${SCRIPT_DIR_RUN}/channeldrop_mtsci_beijing.py"
GAIT_DROP_SCRIPT="${SCRIPT_DIR_RUN}/channeldrop_mtsci_gait.py"
PHYSIONET_DROP_SCRIPT="${SCRIPT_DIR_RUN}/channeldrop_mtsci_physionet_5feat.py"
STOCK_DROP_SCRIPT="${SCRIPT_DIR_RUN}/channeldrop_mtsci_stock.py"
WEATHER_DROP_SCRIPT="${SCRIPT_DIR_RUN}/channeldrop_mtsci_weather.py"

# Dataset paths for Markov runs
BEIJING_SHARED_DATA_DIR="${ROOT_DIR}/data/beijing"
GAIT_DATA_DIR="${ROOT_DIR}/data/automatic-ou-gaitdata"
PHYSIONET_CSV="${ROOT_DIR}/data/physionet2019.csv"
STOCK_CSV="${ROOT_DIR}/data/stock_data.csv"
WEATHER_CSV="${ROOT_DIR}/data/weather.csv"

# Channel-drop asset root
CHANNELDROP_ROOT="${ROOT_DIR}/channeldropassets"

# Output root
OUT_ROOT="${SCRIPT_DIR}/mtsci_runs"

# Device
DEVICE="cuda:0"

# Seeds
SEEDS=(1 2 3 4 5 6 7 8 9 10)

# Common settings
EPOCHS=50
LM=6.0
RATIOS="0.10,0.30,0.50,0.70"
NSAMPLE=50

# Sequence lengths
BEIJING_SEQ_LEN=96
GAIT_SEQ_LEN=128
PHYSIONET_SEQ_LEN=96
STOCK_SEQ_LEN=48
WEATHER_SEQ_LEN=96

# Batch sizes
BEIJING_BATCH_MARKOV=32
GAIT_BATCH_MARKOV=8
PHYSIONET_BATCH_MARKOV=8
STOCK_BATCH_MARKOV=32
WEATHER_BATCH_MARKOV=32

BEIJING_BATCH_DROP=32
GAIT_BATCH_DROP=8
PHYSIONET_BATCH_DROP=8
STOCK_BATCH_DROP=16
WEATHER_BATCH_DROP=32

mkdir -p "${OUT_ROOT}/markov/logs" "${OUT_ROOT}/markov/metrics"
mkdir -p "${OUT_ROOT}/channeldrop/logs" "${OUT_ROOT}/channeldrop/metrics"

MARKOV_MASTER="${OUT_ROOT}/markov/metrics/all_markov_metrics.tsv"
CHANNELDROP_MASTER="${OUT_ROOT}/channeldrop/metrics/all_channeldrop_metrics.tsv"

echo -e "dataset\tseed\tr_masked\tMAE\tMSE\tRMSE" > "${MARKOV_MASTER}"
echo -e "dataset\tseed\tprotocol\tsplit\tMAE\tMSE\tRMSE" > "${CHANNELDROP_MASTER}"

# =========================================================
# Step 1: Markov-mask experiments
# =========================================================
echo "===================================================="
echo "Running MTSCI markov-mask experiments"
echo "===================================================="

for SEED in "${SEEDS[@]}"; do
  echo "[markov] seed=${SEED}"

  # Beijing
  RUN_METRICS="${OUT_ROOT}/markov/metrics/beijing_seed${SEED}_metrics.tsv"
  python "${BEIJING_MARKOV_SCRIPT}" \
   --shared_data_dir "${BEIJING_SHARED_DATA_DIR}" \
   --seq_len "${BEIJING_SEQ_LEN}" \
   --epochs "${EPOCHS}" \
   --batch_size "${BEIJING_BATCH_MARKOV}" \
   --seed "${SEED}" \
   --nsample "${NSAMPLE}" \
   --lm "${LM}" \
   --eval_masked_ratios "${RATIOS}" \
   --device "${DEVICE}" \
   --out_txt "${RUN_METRICS}" \
   > "${OUT_ROOT}/markov/logs/beijing_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="beijing" -v sd="${SEED}" \
   'NR>1 && NF>=10 {printf "%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $7, $8, $9, $10}' \
   "${RUN_METRICS}" >> "${MARKOV_MASTER}"

  # Gait
  RUN_METRICS="${OUT_ROOT}/markov/metrics/gait_seed${SEED}_metrics.tsv"
  python "${GAIT_MARKOV_SCRIPT}" \
   --data_dir "${GAIT_DATA_DIR}" \
   --seq_len "${GAIT_SEQ_LEN}" \
   --epochs "${EPOCHS}" \
   --batch_size "${GAIT_BATCH_MARKOV}" \
   --seed "${SEED}" \
   --nsample "${NSAMPLE}" \
   --lm "${LM}" \
   --eval_masked_ratios "${RATIOS}" \
   --device "${DEVICE}" \
   --out_txt "${RUN_METRICS}" \
   > "${OUT_ROOT}/markov/logs/gait_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="gait" -v sd="${SEED}" \
   'NR>1 && NF>=10 {printf "%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $7, $8, $9, $10}' \
   "${RUN_METRICS}" >> "${MARKOV_MASTER}"

  # PhysioNet
  RUN_METRICS="${OUT_ROOT}/markov/metrics/physionet_seed${SEED}_metrics.tsv"
  python "${PHYSIONET_MARKOV_SCRIPT}" \
   --csv "${PHYSIONET_CSV}" \
   --seq_len "${PHYSIONET_SEQ_LEN}" \
   --epochs "${EPOCHS}" \
   --batch_size "${PHYSIONET_BATCH_MARKOV}" \
   --seed "${SEED}" \
   --nsample "${NSAMPLE}" \
   --lm "${LM}" \
   --eval_masked_ratios "${RATIOS}" \
   --device "${DEVICE}" \
   --out_txt "${RUN_METRICS}" \
   > "${OUT_ROOT}/markov/logs/physionet_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="physionet" -v sd="${SEED}" \
   'NR>1 && NF>=10 {printf "%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $7, $8, $9, $10}' \
   "${RUN_METRICS}" >> "${MARKOV_MASTER}"

  # Stock
  RUN_METRICS="${OUT_ROOT}/markov/metrics/stock_seed${SEED}_metrics.tsv"
  python "${STOCK_MARKOV_SCRIPT}" \
   --csv "${STOCK_CSV}" \
   --seq_len "${STOCK_SEQ_LEN}" \
   --epochs "${EPOCHS}" \
   --batch_size "${STOCK_BATCH_MARKOV}" \
   --seed "${SEED}" \
   --nsample "${NSAMPLE}" \
   --lm "${LM}" \
   --eval_masked_ratios "${RATIOS}" \
   --device "${DEVICE}" \
   --out_txt "${RUN_METRICS}" \
   > "${OUT_ROOT}/markov/logs/stock_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="stock" -v sd="${SEED}" \
   'NR>1 && NF>=10 {printf "%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $7, $8, $9, $10}' \
   "${RUN_METRICS}" >> "${MARKOV_MASTER}"

  # Weather
  RUN_METRICS="${OUT_ROOT}/markov/metrics/weather_seed${SEED}_metrics.tsv"
  python "${WEATHER_MARKOV_SCRIPT}" \
   --csv "${WEATHER_CSV}" \
   --seq_len "${WEATHER_SEQ_LEN}" \
   --epochs "${EPOCHS}" \
   --batch_size "${WEATHER_BATCH_MARKOV}" \
   --seed "${SEED}" \
   --nsample "${NSAMPLE}" \
   --lm "${LM}" \
   --eval_masked_ratios "${RATIOS}" \
   --device "${DEVICE}" \
   --out_txt "${RUN_METRICS}" \
   > "${OUT_ROOT}/markov/logs/weather_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="weather" -v sd="${SEED}" \
   'NR>1 && NF>=10 {printf "%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $7, $8, $9, $10}' \
   "${RUN_METRICS}" >> "${MARKOV_MASTER}"
done

# =========================================================
# Step 2: Channel-drop experiments
# =========================================================
echo "===================================================="
echo "Running MTSCI channel-drop experiments"
echo "===================================================="

render_mask_name() {
  local fmt="$1"
  local seq="$2"
  local proto="$3"
  local seed="$4"
  fmt="${fmt//%SEQ%/${seq}}"
  fmt="${fmt//%PROTO%/${proto}}"
  fmt="${fmt//%SEED%/${seed}}"
  echo "${fmt}"
}

get_asset_dir() {
  local dataset="$1"
  local seed="$2"
  case "${dataset}" in
    weather)   echo "${CHANNELDROP_ROOT}/shared_weather_channeldrop_assets" ;;
    stock)     echo "${CHANNELDROP_ROOT}/shared_stock_channeldrop_assets" ;;
    beijing)   echo "${CHANNELDROP_ROOT}/shared_beijing_channeldrop_assets" ;;
    gait)      echo "${CHANNELDROP_ROOT}/shared_gait_channeldrop_assets_seed${seed}" ;;
    physionet) echo "${CHANNELDROP_ROOT}/shared_physionet_channeldrop_assets_seed${seed}" ;;
    *) echo "Unknown dataset: ${dataset}" >&2; exit 1 ;;
  esac
}

declare -A SCRIPT_MAP
declare -A SEQ_LEN_MAP
declare -A BATCH_MAP
declare -A VAL_MASK_FMT
declare -A TEST_MASK_FMT

SCRIPT_MAP[weather]="${WEATHER_DROP_SCRIPT}"
SCRIPT_MAP[gait]="${GAIT_DROP_SCRIPT}"
SCRIPT_MAP[physionet]="${PHYSIONET_DROP_SCRIPT}"
SCRIPT_MAP[stock]="${STOCK_DROP_SCRIPT}"
SCRIPT_MAP[beijing]="${BEIJING_DROP_SCRIPT}"

SEQ_LEN_MAP[weather]="${WEATHER_SEQ_LEN}"
SEQ_LEN_MAP[gait]="${GAIT_SEQ_LEN}"
SEQ_LEN_MAP[physionet]="${PHYSIONET_SEQ_LEN}"
SEQ_LEN_MAP[stock]="${STOCK_SEQ_LEN}"
SEQ_LEN_MAP[beijing]="${BEIJING_SEQ_LEN}"

BATCH_MAP[weather]="${WEATHER_BATCH_DROP}"
BATCH_MAP[gait]="${GAIT_BATCH_DROP}"
BATCH_MAP[physionet]="${PHYSIONET_BATCH_DROP}"
BATCH_MAP[stock]="${STOCK_BATCH_DROP}"
BATCH_MAP[beijing]="${BEIJING_BATCH_DROP}"

VAL_MASK_FMT[weather]="weather_seq%SEQ%_val_%PROTO%_seed%SEED%.npz"
TEST_MASK_FMT[weather]="weather_seq%SEQ%_test_%PROTO%_seed%SEED%.npz"
VAL_MASK_FMT[gait]="gait_seq%SEQ%_val_%PROTO%_seed%SEED%.npz"
TEST_MASK_FMT[gait]="gait_seq%SEQ%_test_%PROTO%_seed%SEED%.npz"
VAL_MASK_FMT[physionet]="physionet_seq%SEQ%_val_%PROTO%_seed%SEED%.npz"
TEST_MASK_FMT[physionet]="physionet_seq%SEQ%_test_%PROTO%_seed%SEED%.npz"
VAL_MASK_FMT[stock]="stock_seq%SEQ%_val_%PROTO%_seed%SEED%.npz"
TEST_MASK_FMT[stock]="stock_seq%SEQ%_test_%PROTO%_seed%SEED%.npz"
VAL_MASK_FMT[beijing]="beijing_seq%SEQ%_val_%PROTO%_seed%SEED%.npz"
TEST_MASK_FMT[beijing]="beijing_seq%SEQ%_test_%PROTO%_seed%SEED%.npz"

for dataset in beijing gait physionet stock weather; do
  script="${SCRIPT_MAP[$dataset]}"
  seq_len="${SEQ_LEN_MAP[$dataset]}"
  batch="${BATCH_MAP[$dataset]}"

  for seed in "${SEEDS[@]}"; do
    asset_dir="$(get_asset_dir "${dataset}" "${seed}")"
    [[ -d "${asset_dir}" ]] || { echo "Missing asset dir: ${asset_dir}" >&2; exit 1; }

    for proto in drop1 drop2; do
      val_mask="$(render_mask_name "${VAL_MASK_FMT[$dataset]}" "${seq_len}" "${proto}" "${seed}")"
      test_mask="$(render_mask_name "${TEST_MASK_FMT[$dataset]}" "${seq_len}" "${proto}" "${seed}")"

      val_mask_path="${asset_dir}/${val_mask}"
      test_mask_path="${asset_dir}/${test_mask}"

      [[ -f "${val_mask_path}" ]] || { echo "Missing val maskbank: ${val_mask_path}" >&2; exit 1; }
      [[ -f "${test_mask_path}" ]] || { echo "Missing test maskbank: ${test_mask_path}" >&2; exit 1; }

      RUN_METRICS="${OUT_ROOT}/channeldrop/metrics/${dataset}_seed${seed}_${proto}_metrics.tsv"
      WORK_DIR="${OUT_ROOT}/channeldrop/work/${dataset}/${proto}/seed${seed}"
      mkdir -p "${WORK_DIR}"

      python "${script}" \
        --mtsci_root "${MTSCI_ROOT}" \
        --asset_dir "${asset_dir}" \
        --seq_len "${seq_len}" \
        --batch "${batch}" \
        --epochs "${EPOCHS}" \
        --seed "${seed}" \
        --nsample "${NSAMPLE}" \
        --device "${DEVICE}" \
        --val_maskbank "${val_mask_path}" \
        --test_maskbank "${test_mask_path}" \
        --protocol "${proto}" \
        --work_dir "${WORK_DIR}" \
        --out_txt "${RUN_METRICS}" \
        > "${OUT_ROOT}/channeldrop/logs/${dataset}_seed${seed}_${proto}.log" 2>&1

      awk -F '\t' -v ds="${dataset}" -v sd="${seed}" -v pr="${proto}" \
        'NR>1 && NF>=5 {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, pr, $1, $3, $4, $5}' \
        "${RUN_METRICS}" >> "${CHANNELDROP_MASTER}"
    done
  done
done

echo "===================================================="
echo "Done."
echo "Markov logs:      ${OUT_ROOT}/markov/logs"
echo "Markov metrics:   ${OUT_ROOT}/markov/metrics"
echo "Channel logs:     ${OUT_ROOT}/channeldrop/logs"
echo "Channel metrics:  ${OUT_ROOT}/channeldrop/metrics"
echo "===================================================="
