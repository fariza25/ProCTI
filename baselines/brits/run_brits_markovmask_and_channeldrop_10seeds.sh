#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Run BRITS Markov-mask and Channel-drop experiments
# for 10 seeds and save per-run + combined metrics.
#
# Edit the USER PATHS section before running.
# =========================================================

# -----------------------------
# USER PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${BASELINES_DIR}/.." && pwd)"

SPIN_ROOT="${BASELINES_DIR}/spin"

# Markov-mask data / maskbank paths
BEIJING_SHARED_DATA_DIR="${ROOT_DIR}/data/beijing"
GAIT_DATA_DIR="${ROOT_DIR}/data/automatic-ou-gaitdata"
PHYSIONET_CSV="${ROOT_DIR}/data/physionet2019.csv"
STOCK_CSV="${ROOT_DIR}/data/stock_data.csv"
WEATHER_CSV="${ROOT_DIR}/data/weather.csv"

BEIJING_MASK_DIR="${ROOT_DIR}/maskbanks/beijing"
GAIT_MASK_DIR="${ROOT_DIR}/maskbanks/gait"
PHYSIONET_MASK_DIR="${ROOT_DIR}/maskbanks/physionet"
STOCK_MASK_DIR="${ROOT_DIR}/maskbanks/stock"
WEATHER_MASK_DIR="${ROOT_DIR}/maskbanks/weather"

# Channel-drop assets root
CHANNELDROP_ROOT="${ROOT_DIR}/channeldropassets"

# Output root
OUT_ROOT="${SCRIPT_DIR}/brits_runs"

# Device
DEVICE="cuda:0"

# Seeds
SEEDS=(1 2 3 4 5 6 7 8 9 10)

# Common settings
EPOCHS=50
LM=6.0
RATIOS="0.10,0.30,0.50,0.70"

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
STOCK_BATCH_DROP=32
WEATHER_BATCH_DROP=32

mkdir -p "${OUT_ROOT}/markov/logs" "${OUT_ROOT}/markov/metrics"
mkdir -p "${OUT_ROOT}/channeldrop/logs" "${OUT_ROOT}/channeldrop/metrics"

MARKOV_MASTER="${OUT_ROOT}/markov/metrics/all_markov_metrics.tsv"
CHANNELDROP_MASTER="${OUT_ROOT}/channeldrop/metrics/all_channeldrop_metrics.tsv"

echo -e "dataset\tseed\tsplit\tr_masked\tr_observed\tMAE\tMSE\tRMSE" > "${MARKOV_MASTER}"
echo -e "dataset\tseed\tprotocol\tsplit\tMAE\tMSE\tRMSE" > "${CHANNELDROP_MASTER}"

# =========================================================
# Step 1: Markov-mask experiments
# =========================================================
echo "===================================================="
echo "Running BRITS markov-mask experiments"
echo "===================================================="

for SEED in "${SEEDS[@]}"; do
  echo "[markov] seed=${SEED}"

  RUN_METRICS="${OUT_ROOT}/markov/metrics/beijing_seed${SEED}_metrics.tsv"
  python "${SCRIPT_DIR}/markovmask_brits_beijing.py" \
    --shared_data_dir "${BEIJING_SHARED_DATA_DIR}" \
    --seq_len "${BEIJING_SEQ_LEN}" \
    --batch "${BEIJING_BATCH_MARKOV}" \
    --epochs "${EPOCHS}" \
    --train_masked_ratio 0.15 \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${BEIJING_MASK_DIR}" \
    --out_txt "${RUN_METRICS}" \
    > "${OUT_ROOT}/markov/logs/beijing_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="beijing" -v sd="${SEED}" 'NR>1 && NF>=6 {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $1, $2, $3, $4, $5, $6}' "${RUN_METRICS}" >> "${MARKOV_MASTER}"

  RUN_METRICS="${OUT_ROOT}/markov/metrics/gait_seed${SEED}_metrics.tsv"
  python "${SCRIPT_DIR}/markovmask_brits_gait.py" \
    --spin_root "${SPIN_ROOT}" \
    --data_dir "${GAIT_DATA_DIR}" \
    --seq_len "${GAIT_SEQ_LEN}" \
    --batch "${GAIT_BATCH_MARKOV}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${GAIT_MASK_DIR}" \
    --out_txt "${RUN_METRICS}" \
    > "${OUT_ROOT}/markov/logs/gait_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="gait" -v sd="${SEED}" 'NR>1 && NF>=6 {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $1, $2, $3, $4, $5, $6}' "${RUN_METRICS}" >> "${MARKOV_MASTER}"

  RUN_METRICS="${OUT_ROOT}/markov/metrics/physionet_seed${SEED}_metrics.tsv"
  python "${SCRIPT_DIR}/markovmask_brits_physionet.py" \
    --spin_root "${SPIN_ROOT}" \
    --csv "${PHYSIONET_CSV}" \
    --seq_len "${PHYSIONET_SEQ_LEN}" \
    --batch "${PHYSIONET_BATCH_MARKOV}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${PHYSIONET_MASK_DIR}" \
    --out_txt "${RUN_METRICS}" \
    > "${OUT_ROOT}/markov/logs/physionet_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="physionet" -v sd="${SEED}" 'NR>1 && NF>=6 {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $1, $2, $3, $4, $5, $6}' "${RUN_METRICS}" >> "${MARKOV_MASTER}"

  RUN_METRICS="${OUT_ROOT}/markov/metrics/stock_seed${SEED}_metrics.tsv"
  python "${SCRIPT_DIR}/markovmask_brits_stock.py" \
    --spin_root "${SPIN_ROOT}" \
    --csv "${STOCK_CSV}" \
    --seq_len "${STOCK_SEQ_LEN}" \
    --batch "${STOCK_BATCH_MARKOV}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${STOCK_MASK_DIR}" \
    --out_txt "${RUN_METRICS}" \
    > "${OUT_ROOT}/markov/logs/stock_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="stock" -v sd="${SEED}" 'NR>1 && NF>=6 {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $1, $2, $3, $4, $5, $6}' "${RUN_METRICS}" >> "${MARKOV_MASTER}"

  RUN_METRICS="${OUT_ROOT}/markov/metrics/weather_seed${SEED}_metrics.tsv"
  python "${SCRIPT_DIR}/markovmask_brits_weather.py" \
    --spin_root "${SPIN_ROOT}" \
    --csv "${WEATHER_CSV}" \
    --seq_len "${WEATHER_SEQ_LEN}" \
    --batch "${WEATHER_BATCH_MARKOV}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${WEATHER_MASK_DIR}" \
    --out_txt "${RUN_METRICS}" \
    > "${OUT_ROOT}/markov/logs/weather_seed${SEED}.log" 2>&1
  awk -F '\t' -v ds="weather" -v sd="${SEED}" 'NR>1 && NF>=6 {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, $1, $2, $3, $4, $5, $6}' "${RUN_METRICS}" >> "${MARKOV_MASTER}"
done

# =========================================================
# Step 2: Channel-drop experiments
# =========================================================
echo "===================================================="
echo "Running BRITS channel-drop experiments"
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

SCRIPT_MAP[weather]="${SCRIPT_DIR}/channeldrop_brits_weather.py"
SCRIPT_MAP[gait]="${SCRIPT_DIR}/channeldrop_brits_gait.py"
SCRIPT_MAP[physionet]="${SCRIPT_DIR}/channeldrop_brits_physionet_5feat.py"
SCRIPT_MAP[stock]="${SCRIPT_DIR}/channeldrop_brits_stock.py"
SCRIPT_MAP[beijing]="${SCRIPT_DIR}/channeldrop_brits_beijing.py"

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

      python "${script}" \
        --spin_root "${SPIN_ROOT}" \
        --asset_dir "${asset_dir}" \
        --seq_len "${seq_len}" \
        --batch "${batch}" \
        --epochs "${EPOCHS}" \
        --seed "${seed}" \
        --device "${DEVICE}" \
        --val_maskbank "${val_mask_path}" \
        --test_maskbank "${test_mask_path}" \
        --protocol "${proto}" \
        --out_txt "${RUN_METRICS}" \
        > "${OUT_ROOT}/channeldrop/logs/${dataset}_seed${seed}_${proto}.log" 2>&1

      awk -F '\t' -v ds="${dataset}" -v sd="${seed}" -v pr="${proto}" 'NR>1 && NF>=5 {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ds, sd, pr, $1, $3, $4, $5}' "${RUN_METRICS}" >> "${CHANNELDROP_MASTER}"
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
