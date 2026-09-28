#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Run ProCTI channel-drop experiments for all AVAILABLE seeds
# found in the channeldropassets directory.
#
# It auto-detects seeds separately for:
#   - beijing
#   - gait
#   - physionet
#   - stock
#   - weather
#
# And for both protocols:
#   - drop1
#   - drop2
#
# It only runs a seed/protocol if BOTH val and test maskbanks exist.
# =========================================================

# -----------------------------
# USER PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${SCRIPT_DIR}"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHANNELDROP_ROOT="${ROOT_DIR}/channeldropassets"

BEIJING_SCRIPT="${SCRIPT_DIR}/channeldrop_procti_beijing.py"
GAIT_SCRIPT="${SCRIPT_DIR}/channeldrop_procti_gait.py"
PHYSIONET_SCRIPT="${SCRIPT_DIR}/channeldrop_procti_physionet_5feat.py"
STOCK_SCRIPT="${SCRIPT_DIR}/channeldrop_procti_stock.py"
WEATHER_SCRIPT="${SCRIPT_DIR}/channeldrop_procti_weather.py"

OUT_ROOT="${SCRIPT_DIR}/procti_channeldrop_runs"

# Device
DEVICE="cuda:1"

# Training/eval settings
EPOCHS=50
LM=6.0
N_SAMPLES=20

# Sequence lengths
BEIJING_SEQ_LEN=96
GAIT_SEQ_LEN=128
PHYSIONET_SEQ_LEN=96
STOCK_SEQ_LEN=48
WEATHER_SEQ_LEN=96

# Batch sizes (edit if needed)
BEIJING_BATCH=16
GAIT_BATCH=8
PHYSIONET_BATCH=8
STOCK_BATCH=64
WEATHER_BATCH=64

mkdir -p "${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}/metrics"
mkdir -p "${OUT_ROOT}/saved_arrays"

collect_seeds_shared_dir() {
  local dir="$1"
  local prefix="$2"
  local protocol="$3"

  python - "$dir" "$prefix" "$protocol" << 'PY'
import os, re, sys
dir_, prefix, protocol = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.isdir(dir_):
    sys.exit(0)

pat_val = re.compile(rf"^{re.escape(prefix)}_val_{re.escape(protocol)}_seed(\d+)\.npz$")
seeds = []
files = set(os.listdir(dir_))
for fn in files:
    m = pat_val.match(fn)
    if not m:
        continue
    s = m.group(1)
    test_fn = f"{prefix}_test_{protocol}_seed{s}.npz"
    if test_fn in files:
        seeds.append(int(s))
print(" ".join(str(x) for x in sorted(set(seeds))))
PY
}

collect_seeds_seeded_dirs() {
  local root="$1"
  local stem="$2"
  local prefix="$3"
  local protocol="$4"

  python - "$root" "$stem" "$prefix" "$protocol" << 'PY'
import os, re, sys
root, stem, prefix, protocol = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
if not os.path.isdir(root):
    sys.exit(0)

dir_pat = re.compile(rf"^{re.escape(stem)}(\d+)$")
seeds = []
for name in os.listdir(root):
    m = dir_pat.match(name)
    if not m:
        continue
    s = int(m.group(1))
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        continue
    val_fn = f"{prefix}_val_{protocol}_seed{s}.npz"
    test_fn = f"{prefix}_test_{protocol}_seed{s}.npz"
    if os.path.exists(os.path.join(d, val_fn)) and os.path.exists(os.path.join(d, test_fn)):
        seeds.append(s)
print(" ".join(str(x) for x in sorted(set(seeds))))
PY
}

run_dataset_shared() {
  local dataset="$1"
  local script="$2"
  local asset_dir="$3"
  local prefix="$4"
  local seq_len="$5"
  local batch="$6"

  for protocol in drop1 drop2; do
    mapfile -t seeds < <(collect_seeds_shared_dir "${asset_dir}" "${prefix}" "${protocol}" | tr ' ' '\n' | sed '/^$/d')
    if [[ ${#seeds[@]} -eq 0 ]]; then
      echo "[skip] ${dataset} ${protocol}: no matching seeds found in ${asset_dir}"
      continue
    fi

    echo "[info] ${dataset} ${protocol}: found seeds ${seeds[*]}"

    for seed in "${seeds[@]}"; do
      local val_mask="${asset_dir}/${prefix}_val_${protocol}_seed${seed}.npz"
      local test_mask="${asset_dir}/${prefix}_test_${protocol}_seed${seed}.npz"

      echo "[run] ${dataset} ${protocol} seed=${seed}"

      local extra_args=()
      if [[ "${dataset}" == "weather" ]]; then
        extra_args+=(--save_test_arrays --save_dir "${OUT_ROOT}/saved_arrays/${dataset}/${protocol}/seed${seed}")
      fi

      python "${script}" \
        --asset_dir "${asset_dir}" \
        --code_dir "${CODE_DIR}" \
        --seq_len "${seq_len}" \
        --batch "${batch}" \
        --epochs "${EPOCHS}" \
        --lm "${LM}" \
        --seed "${seed}" \
        --device "${DEVICE}" \
        --n_samples "${N_SAMPLES}" \
        --val_maskbank "${val_mask}" \
        --test_maskbank "${test_mask}" \
        --protocol "${protocol}" \
        --out_txt "${OUT_ROOT}/metrics/${dataset}_${protocol}_seed${seed}_metrics.tsv" \
        "${extra_args[@]}" \
        > "${OUT_ROOT}/logs/${dataset}_${protocol}_seed${seed}.log" 2>&1
    done
  done
}

run_dataset_seeded_dirs() {
  local dataset="$1"
  local script="$2"
  local root="$3"
  local dir_stem="$4"
  local prefix="$5"
  local seq_len="$6"
  local batch="$7"

  for protocol in drop1 drop2; do
    mapfile -t seeds < <(collect_seeds_seeded_dirs "${root}" "${dir_stem}" "${prefix}" "${protocol}" | tr ' ' '\n' | sed '/^$/d')
    if [[ ${#seeds[@]} -eq 0 ]]; then
      echo "[skip] ${dataset} ${protocol}: no matching seeds found under ${root}"
      continue
    fi

    echo "[info] ${dataset} ${protocol}: found seeds ${seeds[*]}"

    for seed in "${seeds[@]}"; do
      local asset_dir="${root}/${dir_stem}${seed}"
      local val_mask="${asset_dir}/${prefix}_val_${protocol}_seed${seed}.npz"
      local test_mask="${asset_dir}/${prefix}_test_${protocol}_seed${seed}.npz"

      echo "[run] ${dataset} ${protocol} seed=${seed}"

      python "${script}" \
        --asset_dir "${asset_dir}" \
        --code_dir "${CODE_DIR}" \
        --seq_len "${seq_len}" \
        --batch "${batch}" \
        --epochs "${EPOCHS}" \
        --lm "${LM}" \
        --seed "${seed}" \
        --device "${DEVICE}" \
        --n_samples "${N_SAMPLES}" \
        --val_maskbank "${val_mask}" \
        --test_maskbank "${test_mask}" \
        --protocol "${protocol}" \
        --out_txt "${OUT_ROOT}/metrics/${dataset}_${protocol}_seed${seed}_metrics.tsv" \
        > "${OUT_ROOT}/logs/${dataset}_${protocol}_seed${seed}.log" 2>&1
    done
  done
}

echo "===================================================="
echo "Running ProCTI channel-drop experiments"
echo "===================================================="

run_dataset_shared \
 "beijing" \
 "${BEIJING_SCRIPT}" \
 "${CHANNELDROP_ROOT}/shared_beijing_channeldrop_assets" \
 "beijing_seq${BEIJING_SEQ_LEN}" \
 "${BEIJING_SEQ_LEN}" \
 "${BEIJING_BATCH}"

run_dataset_seeded_dirs \
 "gait" \
 "${GAIT_SCRIPT}" \
 "${CHANNELDROP_ROOT}" \
 "shared_gait_channeldrop_assets_seed" \
 "gait_seq${GAIT_SEQ_LEN}" \
 "${GAIT_SEQ_LEN}" \
 "${GAIT_BATCH}"

run_dataset_seeded_dirs \
 "physionet" \
 "${PHYSIONET_SCRIPT}" \
 "${CHANNELDROP_ROOT}" \
 "shared_physionet_channeldrop_assets_seed" \
 "physionet_seq${PHYSIONET_SEQ_LEN}" \
 "${PHYSIONET_SEQ_LEN}" \
 "${PHYSIONET_BATCH}"

run_dataset_shared \
 "stock" \
 "${STOCK_SCRIPT}" \
 "${CHANNELDROP_ROOT}/shared_stock_channeldrop_assets" \
 "stock_seq${STOCK_SEQ_LEN}" \
 "${STOCK_SEQ_LEN}" \
 "${STOCK_BATCH}"

run_dataset_shared \
  "weather" \
  "${WEATHER_SCRIPT}" \
  "${CHANNELDROP_ROOT}/shared_weather_channeldrop_assets" \
  "weather_seq${WEATHER_SEQ_LEN}" \
  "${WEATHER_SEQ_LEN}" \
  "${WEATHER_BATCH}"

echo "===================================================="
echo "Done."
echo "Logs:    ${OUT_ROOT}/logs"
echo "Metrics: ${OUT_ROOT}/metrics"
echo "Arrays:  ${OUT_ROOT}/saved_arrays"
echo "===================================================="
