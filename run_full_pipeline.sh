#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Full ProCTI + baselines pipeline
#
# Place this file inside: procti_submission_code/
#
# It will:
#   1) generate shared maskbanks (seeds 1..10)
#   2) generate channel-drop assets (seeds 1..10)
#   3) run ProCTI:
#        - run_procti_markovmask_10seeds.sh
#        - run_procti_channeldrop_available_seeds.sh
#        - run_procti_ablations.sh
#   4) run all baseline markov/channel-drop scripts
# =========================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="${ROOT_DIR}/_pipeline_logs"

SEEDS=(1 2 3 4 5 6 7 8 9 10)

LM="6.0"
RATIOS="0.10,0.30,0.50,0.70"

# -----------------------------
# Paths
# -----------------------------
DATA_DIR="${ROOT_DIR}/data"
MASKBANK_ROOT="${ROOT_DIR}/maskbanks"
CHANNELDROP_ROOT="${ROOT_DIR}/channeldropassets"

PROCTI_DIR="${ROOT_DIR}/ProCTI"
BASELINES_DIR="${ROOT_DIR}/baselines"

mkdir -p "${LOG_ROOT}"
mkdir -p "${MASKBANK_ROOT}"
mkdir -p "${CHANNELDROP_ROOT}"

# -----------------------------
# Helpers
# -----------------------------
timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

run_and_log() {
  local name="$1"
  shift
  local logfile="${LOG_ROOT}/${name}.log"

  echo "[$(timestamp)] START ${name}"
  "$@" > "${logfile}" 2>&1
  echo "[$(timestamp)] DONE  ${name}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: Missing required directory: ${path}" >&2
    exit 1
  fi
}

# -----------------------------
# Validate key directories
# -----------------------------
require_dir "${DATA_DIR}"
require_dir "${PROCTI_DIR}"
require_dir "${BASELINES_DIR}"

# ProCTI bash runners
require_file "${PROCTI_DIR}/run_procti_markovmask_10seeds.sh"
require_file "${PROCTI_DIR}/run_procti_channeldrop_available_seeds.sh"
require_file "${PROCTI_DIR}/run_procti_ablations.sh"

# Root-level generator scripts
require_file "${ROOT_DIR}/make_beijing_maskbank.py"
require_file "${ROOT_DIR}/make_gait_maskbank.py"
require_file "${ROOT_DIR}/make_physionet_maskbank.py"
require_file "${ROOT_DIR}/make_stock_maskbank.py"
require_file "${ROOT_DIR}/make_weather_maskbank.py"

require_file "${ROOT_DIR}/make_beijing_channeldrop_assets.py"
require_file "${ROOT_DIR}/make_gait_channeldrop_assets_userwise.py"
require_file "${ROOT_DIR}/make_physionet_channeldrop_assets_patientwise.py"
require_file "${ROOT_DIR}/make_weather_channeldrop_assets.py"

# Stock channel-drop generator:
# try the generic helper first, then a stock-specific helper if present.
STOCK_CHDROP_GEN=""
if [[ -f "${ROOT_DIR}/make_shared_channeldrop_assets.py" ]]; then
  STOCK_CHDROP_GEN="${ROOT_DIR}/make_shared_channeldrop_assets.py"
elif [[ -f "${ROOT_DIR}/make_stock_channeldrop_assets.py" ]]; then
  STOCK_CHDROP_GEN="${ROOT_DIR}/make_stock_channeldrop_assets.py"
else
  echo "ERROR: Missing stock channel-drop generator." >&2
  echo "Expected one of:" >&2
  echo "  ${ROOT_DIR}/make_shared_channeldrop_assets.py" >&2
  echo "  ${ROOT_DIR}/make_stock_channeldrop_assets.py" >&2
  exit 1
fi

# -----------------------------
# Dataset-specific constants
# -----------------------------
BEIJING_DATA_DIR="${DATA_DIR}/beijing_benchmark/shared"
BEIJING_DATACSV_DIR="${DATA_DIR}/beijing"
GAIT_DATA_DIR="${DATA_DIR}/automatic-ou-gaitdata"
PHYSIONET_CSV="${DATA_DIR}/physionet2019.csv"
STOCK_CSV="${DATA_DIR}/stock_data.csv"
WEATHER_CSV="${DATA_DIR}/weather.csv"

require_dir "${BEIJING_DATA_DIR}"
require_dir "${GAIT_DATA_DIR}"
require_file "${PHYSIONET_CSV}"
require_file "${STOCK_CSV}"
require_file "${WEATHER_CSV}"

# -----------------------------
# 1) Generate shared maskbanks
# -----------------------------
mkdir -p \
  "${MASKBANK_ROOT}/beijing" \
  "${MASKBANK_ROOT}/gait" \
  "${MASKBANK_ROOT}/physionet" \
  "${MASKBANK_ROOT}/stock" \
  "${MASKBANK_ROOT}/weather"

echo "===================================================="
echo "Generating shared maskbanks"
echo "===================================================="

for seed in "${SEEDS[@]}"; do
 run_and_log "maskbank_beijing_seed${seed}" \
   "${PYTHON_BIN}" "${ROOT_DIR}/make_beijing_maskbank.py" \
     --shared_data_dir "${BEIJING_DATA_DIR}" \
     --seq_len 96 \
     --lm "${LM}" \
     --eval_masked_ratios "${RATIOS}" \
     --seed "${seed}" \
     --out_dir "${MASKBANK_ROOT}/beijing"

 run_and_log "maskbank_gait_seed${seed}" \
   "${PYTHON_BIN}" "${ROOT_DIR}/make_gait_maskbank.py" \
     --data_dir "${GAIT_DATA_DIR}" \
     --seq_len 128 \
     --lm "${LM}" \
     --eval_masked_ratios "${RATIOS}" \
     --seed "${seed}" \
     --out_dir "${MASKBANK_ROOT}/gait"

 run_and_log "maskbank_physionet_seed${seed}" \
   "${PYTHON_BIN}" "${ROOT_DIR}/make_physionet_maskbank.py" \
     --csv "${PHYSIONET_CSV}" \
     --seq_len 96 \
     --lm "${LM}" \
     --eval_masked_ratios "${RATIOS}" \
     --seed "${seed}" \
     --out_dir "${MASKBANK_ROOT}/physionet"

 run_and_log "maskbank_stock_seed${seed}" \
   "${PYTHON_BIN}" "${ROOT_DIR}/make_stock_maskbank.py" \
     --csv "${STOCK_CSV}" \
     --seq_len 48 \
     --lm "${LM}" \
     --eval_masked_ratios "${RATIOS}" \
     --seed "${seed}" \
     --include_val \
     --out_dir "${MASKBANK_ROOT}/stock"

 run_and_log "maskbank_weather_seed${seed}" \
   "${PYTHON_BIN}" "${ROOT_DIR}/make_weather_maskbank.py" \
     --csv "${WEATHER_CSV}" \
     --seq_len 96 \
     --lm "${LM}" \
     --eval_masked_ratios "${RATIOS}" \
     --seed "${seed}" \
     --include_val \
     --out_dir "${MASKBANK_ROOT}/weather"
done

# # -----------------------------
# # 2) Generate channel-drop assets
# # -----------------------------
mkdir -p \
  "${CHANNELDROP_ROOT}/shared_beijing_channeldrop_assets" \
  "${CHANNELDROP_ROOT}/shared_stock_channeldrop_assets" \
  "${CHANNELDROP_ROOT}/shared_weather_channeldrop_assets"

echo "===================================================="
echo "Generating channel-drop assets"
echo "===================================================="

for seed in "${SEEDS[@]}"; do
  run_and_log "chdrop_beijing_seed${seed}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/make_beijing_channeldrop_assets.py" \
      --data_dir "${BEIJING_DATACSV_DIR}" \
      --seq_len 96 \
      --seed "${seed}" \
      --out_dir "${CHANNELDROP_ROOT}/shared_beijing_channeldrop_assets"

  run_and_log "chdrop_weather_seed${seed}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/make_weather_channeldrop_assets.py" \
      --csv "${WEATHER_CSV}" \
      --seq_len 96 \
      --seed "${seed}" \
      --out_dir "${CHANNELDROP_ROOT}/shared_weather_channeldrop_assets"

  # stock
  if [[ "$(basename "${STOCK_CHDROP_GEN}")" == "make_shared_channeldrop_assets.py" ]]; then
    run_and_log "chdrop_stock_seed${seed}" \
      "${PYTHON_BIN}" "${STOCK_CHDROP_GEN}" \
        --dataset stock \
        --csv "${STOCK_CSV}" \
        --seq_len 48 \
        --seed "${seed}" \
        --out_dir "${CHANNELDROP_ROOT}/shared_stock_channeldrop_assets"
  else
    run_and_log "chdrop_stock_seed${seed}" \
      "${PYTHON_BIN}" "${STOCK_CHDROP_GEN}" \
        --csv "${STOCK_CSV}" \
        --seq_len 48 \
        --seed "${seed}" \
        --out_dir "${CHANNELDROP_ROOT}/shared_stock_channeldrop_assets"
  fi

  run_and_log "chdrop_physionet_seed${seed}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/make_physionet_channeldrop_assets_patientwise.py" \
      --csv "${PHYSIONET_CSV}" \
      --seq_len 96 \
      --seed "${seed}" \
      --out_dir "${CHANNELDROP_ROOT}/shared_physionet_channeldrop_assets_seed${seed}"

  run_and_log "chdrop_gait_seed${seed}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/make_gait_channeldrop_assets_userwise.py" \
      --data_dir "${GAIT_DATA_DIR}" \
      --seq_len 128 \
      --seed "${seed}" \
      --out_dir "${CHANNELDROP_ROOT}/shared_gait_channeldrop_assets_seed${seed}"
done

# -----------------------------
# 3) Run ProCTI scripts
# -----------------------------
echo "===================================================="
echo "Running ProCTI scripts"
echo "===================================================="

run_and_log "procti_markovmask_10seeds" \
 bash "${PROCTI_DIR}/run_procti_markovmask_10seeds.sh"

run_and_log "procti_channeldrop_available_seeds" \
bash "${PROCTI_DIR}/run_procti_channeldrop_available_seeds.sh"

run_and_log "procti_ablations" \
bash "${PROCTI_DIR}/run_procti_ablations.sh"

# -----------------------------
# 4) Run baseline scripts
# -----------------------------
echo "===================================================="
echo "Running baseline scripts"
echo "===================================================="

declare -a BASELINE_RUNNERS=(
  "${BASELINES_DIR}/brits/run_brits_markovmask_and_channeldrop_10seeds.sh"
  "${BASELINES_DIR}/csdi/run_csdi_markovmask_and_channeldrop_10seeds.sh"
  "${BASELINES_DIR}/Diffusion-TS/run_diffusionts_markovmask_and_channeldrop_10seeds.sh"
  "${BASELINES_DIR}/FGTI/run_fgti_markovmask_and_channeldrop_10seeds.sh"
  "${BASELINES_DIR}/itransformer/run_itransformer_markovmask_and_channeldrop_10seeds.sh"
  "${BASELINES_DIR}/MTSCI/run_mtsci_markovmask_and_channeldrop_10seeds.sh"
  "${BASELINES_DIR}/PaD-TS/run_padts_markovmask_and_channeldrop_10seeds.sh"
  "${BASELINES_DIR}/SCINet/run_scinet_markovmask_and_channeldrop_10seeds.sh"
  "${BASELINES_DIR}/tider/run_tider_markovmask_and_channeldrop_10seeds.sh"
)

for runner in "${BASELINE_RUNNERS[@]}"; do
 require_file "${runner}"
 name="$(basename "$(dirname "${runner}")")_pipeline"
 run_and_log "${name}" bash "${runner}"
done

echo "===================================================="
echo "ALL DONE"
echo "===================================================="
echo "Maskbanks:           ${MASKBANK_ROOT}"
echo "Channel-drop assets: ${CHANNELDROP_ROOT}"
echo "Top-level logs:      ${LOG_ROOT}"
echo "===================================================="
