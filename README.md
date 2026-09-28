# ProCTI Pipeline

This folder contains the full experimental pipeline for ProCTI and multiple baseline methods to reproduce the main results of "ProCTI: Prototype-Refined Global Conditioning for Diffusion-Based Time Series Imputation", accepted at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026).

The pipeline automatically:

1. Creates shared Markov-mask evaluation maskbanks
2. Creates channel-drop benchmark assets
3. Runs ProCTI experiments (random Markov missingness, attribute-wise missingness, ablations)
4. Runs all baseline experiments
5. Saves logs, outputs, metrics, and intermediate assets

------------------------------------------------------------
Repository Structure
------------------------------------------------------------

procti_submission_code/

│── run_full_pipeline.sh

│── requirements.txt
│── data/
│── maskbanks/
│── channeldropassets/
│── ProCTI/
│── baselines/
│── _pipeline_logs/

------------------------------------------------------------
1. Create Python Virtual Environment, Install Dependencies
------------------------------------------------------------

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

Depending on your CUDA / GPU setup, you may need to reinstall PyTorch separately.

------------------------------------------------------------
3. Run Full Experimental Pipeline
------------------------------------------------------------

chmod +x run_full_pipeline.sh
./run_full_pipeline.sh

------------------------------------------------------------
What run_full_pipeline.sh Does
------------------------------------------------------------

Step 1 — Generate Shared Markov Maskbanks

Creates identical evaluation masks for fair benchmarking across models.

Generated for:
- Beijing
- Gait
- PhysioNet
- Stock
- Weather

Scripts used:
make_beijing_maskbank.py
make_gait_maskbank.py
make_physionet_maskbank.py
make_stock_maskbank.py
make_weather_maskbank.py

Output:
maskbanks/

------------------------------------------------------------

Step 2 — Generate Channel-Drop Assets (Structured missingness)


Generated for:
- Beijing
- Weather
- Stock
- PhysioNet
- Gait

Scripts used:
make_beijing_channeldrop_assets.py
make_weather_channeldrop_assets.py
make_shared_channeldrop_assets.py
make_physionet_channeldrop_assets_patientwise.py
make_gait_channeldrop_assets_userwise.py

Output:
channeldropassets/

------------------------------------------------------------

Step 3 — Run ProCTI Experiments

Runs:
ProCTI/run_procti_markovmask_10seeds.sh
ProCTI/run_procti_channeldrop_available_seeds.sh
ProCTI/run_procti_ablations.sh

Includes:
- Markov masking experiments
- Structured misingness/Channel-drop experiments
- Ablation studies

------------------------------------------------------------

Step 4 — Run Baseline Pipelines

Runs all benchmark baselines:

- BRITS
- CSDI
- Diffusion-TS
- FGTI
- iTransformer
- MTSCI
- PaD-TS
- SCINet
- TIDER

Scripts automatically launched:
baselines/*/run_*_markovmask_and_channeldrop_10seeds.sh

------------------------------------------------------------
Logs
------------------------------------------------------------

All pipeline logs are saved to:
_pipeline_logs/

Each step gets its own log file.

All model logs are saved inside each model directory.
Examples:
ProCTI/procti_10seed_pipeline/logs
ProCTI/procti_channeldrop_runs/logs
ProCTI/procti_ablation_runs/logs
baselines/FGTI/fgti_runs/markov/logs
baselines/csdi/csdi_runs/markov/logs

------------------------------------------------------------
Outputs
------------------------------------------------------------

Results are saved inside each model directory.

Examples:
ProCTI/procti_10seed_pipeline/metrics
ProCTI/procti_channeldrop_runs/metrics
ProCTI/procti_ablation_runs/metrics
baselines/csdi/csdi_runs/markov/metrics
baselines/FGTI/fgti_runs/markov/metrics
baselines/FGTI/fgti_runs/channeldrop/metrics

------------------------------------------------------------
Reproducibility
------------------------------------------------------------

Seeds used by default:
1 2 3 4 5 6 7 8 9 10

Mask ratios:
10%, 30%, 50%, 70%

------------------------------------------------------------
Quick Start
------------------------------------------------------------

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh

------------------------------------------------------------
External Assets
------------------------------------------------------------

The following github repositories were accessed to run the baseline models:
1. BRITS: https://github.com/Graph-Machine-Learning-Group/spin/tree/main
2. CSDI: https://github.com/ermongroup/CSDI/tree/main
3. SCINet: https://github.com/cure-lab/SCINet
4. TIDER: https://github.com/liuwj2000/TIDER
6. MTSCI: https://github.com/JeremyChou28/MTSCI
7. Diffusion-TS: https://github.com/Y-debug-sys/Diffusion-TS/tree/main
8. iTransformer: https://github.com/thuml/Time-Series-Library
9. FGTI: https://github.com/FGTI2024/FGTI24/tree/main
10. PaD-TS: https://github.com/wmd3i/PaD-TS
All repositories were used subject to their respective open-source licenses.
