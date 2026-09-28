#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# CSDI Markov-mask + Channel-drop (10 seeds)
# =========================================================

# -----------------------------
# PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${BASELINES_DIR}/.." && pwd)"

# Data
BEIJING_DATA="${ROOT_DIR}/data/beijing"
GAIT_DATA="${ROOT_DIR}/data/automatic-ou-gaitdata"
PHYSIONET_CSV="${ROOT_DIR}/data/physionet2019.csv"
STOCK_CSV="${ROOT_DIR}/data/stock_data.csv"
WEATHER_CSV="${ROOT_DIR}/data/weather.csv"

# Shared maskbanks (Markov)
BEIJING_MASK="${ROOT_DIR}/maskbanks/beijing"
GAIT_MASK="${ROOT_DIR}/maskbanks/gait"
PHYSIONET_MASK="${ROOT_DIR}/maskbanks/physionet"
STOCK_MASK="${ROOT_DIR}/maskbanks/stock"
WEATHER_MASK="${ROOT_DIR}/maskbanks/weather"

# Channel-drop assets
CHANNELDROP_ROOT="${ROOT_DIR}/channeldropassets"

# Output
OUT_ROOT="${SCRIPT_DIR}/csdi_runs"
DEVICE="cuda:0"

SEEDS=(1 2 3 4 5 6 7 8 9 10)

EPOCHS=50
LM=6.0
RATIOS="0.10,0.30,0.50,0.70"

mkdir -p "${OUT_ROOT}/markov/logs" "${OUT_ROOT}/markov/metrics"
mkdir -p "${OUT_ROOT}/channeldrop/logs" "${OUT_ROOT}/channeldrop/metrics"

MARKOV_MASTER="${OUT_ROOT}/markov/all_markov.tsv"
CHANNEL_MASTER="${OUT_ROOT}/channeldrop/all_channeldrop.tsv"

echo -e "dataset\tseed\tsplit\tr_masked\tMAE\tMSE\tRMSE" > "${MARKOV_MASTER}"
echo -e "dataset\tseed\tprotocol\tsplit\tMAE\tMSE\tRMSE" > "${CHANNEL_MASTER}"

# =========================================================
# MARKOV EXPERIMENTS
# =========================================================
echo "Running CSDI Markov experiments..."

for seed in "${SEEDS[@]}"; do

  # ---------------- BEIJING ----------------
  out="${OUT_ROOT}/markov/metrics/beijing_seed${seed}.tsv"
  python "${SCRIPT_DIR}/markovmask_csdi_beijing.py" \
    --shared_data_dir "${BEIJING_DATA}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${BEIJING_MASK}" \
    --out_txt "${out}" \
    > "${OUT_ROOT}/markov/logs/beijing_${seed}.log" 2>&1

  awk -F'\t' -v ds="beijing" -v sd="${seed}" 'NR>1 {print ds"\t"sd"\t"$0}' "${out}" >> "${MARKOV_MASTER}"

  # ---------------- GAIT ----------------
  out="${OUT_ROOT}/markov/metrics/gait_seed${seed}.tsv"
  python "${SCRIPT_DIR}/markovmask_csdi_gait.py" \
    --data_dir "${GAIT_DATA}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${GAIT_MASK}" \
    --out_txt "${out}" \
    > "${OUT_ROOT}/markov/logs/gait_${seed}.log" 2>&1

  awk -F'\t' -v ds="gait" -v sd="${seed}" 'NR>1 {print ds"\t"sd"\t"$0}' "${out}" >> "${MARKOV_MASTER}"

  # ---------------- PHYSIONET ----------------
  out="${OUT_ROOT}/markov/metrics/physionet_seed${seed}.tsv"
  python "${SCRIPT_DIR}/markovmask_csdi_physionet.py" \
    --csv "${PHYSIONET_CSV}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${PHYSIONET_MASK}" \
    --out_txt "${out}" \
    > "${OUT_ROOT}/markov/logs/physionet_${seed}.log" 2>&1

  awk -F'\t' -v ds="physionet" -v sd="${seed}" 'NR>1 {print ds"\t"sd"\t"$0}' "${out}" >> "${MARKOV_MASTER}"

  # ---------------- STOCK ----------------
  out="${OUT_ROOT}/markov/metrics/stock_seed${seed}.tsv"
  python "${SCRIPT_DIR}/markovmask_csdi_stock.py" \
    --csv "${STOCK_CSV}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${STOCK_MASK}" \
    --out_txt "${out}" \
    > "${OUT_ROOT}/markov/logs/stock_${seed}.log" 2>&1

  awk -F'\t' -v ds="stock" -v sd="${seed}" 'NR>1 {print ds"\t"sd"\t"$0}' "${out}" >> "${MARKOV_MASTER}"

  # ---------------- WEATHER ----------------
  out="${OUT_ROOT}/markov/metrics/weather_seed${seed}.tsv"
  python "${SCRIPT_DIR}/markovmask_csdi_weather.py" \
    --csv "${WEATHER_CSV}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --lm "${LM}" \
    --device "${DEVICE}" \
    --eval_masked_ratios "${RATIOS}" \
    --use_shared_evalmask \
    --shared_evalmask_dir "${WEATHER_MASK}" \
    --out_txt "${out}" \
    > "${OUT_ROOT}/markov/logs/weather_${seed}.log" 2>&1

  awk -F'\t' -v ds="weather" -v sd="${seed}" 'NR>1 {print ds"\t"sd"\t"$0}' "${out}" >> "${MARKOV_MASTER}"

done

# =========================================================
# CHANNEL-DROP EXPERIMENTS
# =========================================================
echo "Running CSDI channel-drop experiments..."

for dataset in beijing gait physionet stock weather; do
  for seed in "${SEEDS[@]}"; do
    for proto in drop1 drop2; do

      case "${dataset}" in
        beijing)   script="channeldrop_csdi_beijing.py"; seq=96 ;;
        gait)      script="channeldrop_csdi_gait.py"; seq=128 ;;
        physionet) script="channeldrop_csdi_physionet_5feat.py"; seq=96 ;;
        stock)     script="channeldrop_csdi_stock.py"; seq=48 ;;
        weather)   script="channeldrop_csdi_weather.py"; seq=96 ;;
      esac

      asset_dir="${CHANNELDROP_ROOT}/shared_${dataset}_channeldrop_assets"
      if [[ "${dataset}" == "gait" || "${dataset}" == "physionet" ]]; then
        asset_dir="${asset_dir}_seed${seed}"
      fi

      val_mask="${asset_dir}/${dataset}_seq${seq}_val_${proto}_seed${seed}.npz"
      test_mask="${asset_dir}/${dataset}_seq${seq}_test_${proto}_seed${seed}.npz"

      out="${OUT_ROOT}/channeldrop/metrics/${dataset}_${seed}_${proto}.tsv"

      if [[ "${dataset}" == "beijing" ]]; then
        python "${SCRIPT_DIR}/${script}" \
          --asset_dir "${asset_dir}" \
          --dataset_name "beijing" \
          --seq_len "${seq}" \
          --seed "${seed}" \
          --device "${DEVICE}" \
          --val_maskbank "${val_mask}" \
          --test_maskbank "${test_mask}" \
          --protocol "${proto}" \
          --out_txt "${out}" \
          > "${OUT_ROOT}/channeldrop/logs/${dataset}_${seed}_${proto}.log" 2>&1
      else
        python "${SCRIPT_DIR}/${script}" \
          --asset_dir "${asset_dir}" \
          --seq_len "${seq}" \
          --seed "${seed}" \
          --device "${DEVICE}" \
          --val_maskbank "${val_mask}" \
          --test_maskbank "${test_mask}" \
          --protocol "${proto}" \
          --out_txt "${out}" \
          > "${OUT_ROOT}/channeldrop/logs/${dataset}_${seed}_${proto}.log" 2>&1
      fi

      awk -F'\t' -v ds="${dataset}" -v sd="${seed}" -v pr="${proto}" 'NR>1 {print ds"\t"sd"\t"pr"\t"$0}' "${out}" >> "${CHANNEL_MASTER}"

    done
  done
done

echo "Done."
echo "Markov metrics:   ${MARKOV_MASTER}"
echo "Channel metrics:  ${CHANNEL_MASTER}"
echo "Logs root:        ${OUT_ROOT}"
