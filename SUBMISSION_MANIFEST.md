# Turbofan RUL Prediction - Submission Manifest

## Project Structure
- Root directory containing all project files
- `src/` - all source code modules
- `configs/` - experiment and model configurations
- `data/raw/` - original NASA C-MAPSS FD001 datasets
- `data/processed/` - preprocessed and scaled datasets
- `experiments/runs/` - 22 trained model runs
- `experiments/results/` - evaluation results and figures
- `experiments/registry.csv` - experiment registry
- `tests/` - test suite (111 tests)
- `demo/` - Streamlit web application
- `report/` - written report and presentation
- `scripts/` - automation scripts

## Datasets
### Raw NASA FD001
- `data/raw/train_FD001.txt` - 20,631 rows, 100 engines, SHA-256: 963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8
- `data/raw/test_FD001.txt` - 13,096 rows, 100 engines, SHA-256: 3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851
- `data/raw/RUL_FD001.txt` - 100 rows (one RUL per test engine), SHA-256: a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca

### Processed Datasets
- `data/processed/train_FD001.csv` - training set (80 engines, 16,561 rows)
- `data/processed/val_FD001.csv` - validation set (20 engines, 4,070 rows)
- `data/processed/test_FD001.csv` - official test set (100 engines, 13,096 rows)
- `data/processed/fd001_metadata.json` - preprocessing metadata
- `data/processed/fd001_scaler.joblib` - fitted StandardScaler
- `data/processed/fd001_test_rul.csv` - test RUL labels

## Model Checkpoints (22 runs)
All runs use seeds 42, 123, 3407 and rul_cap=125:
1. lstm_cap125_seed42 / lstm_cap125_seed123 / lstm_cap125_seed3407
2. transformer_cap125_seed42 / transformer_cap125_seed123 / transformer_cap125_seed3407
3. cnn_bilstm_attention_cap125_seed42 / cnn_bilstm_attention_cap125_seed123 / cnn_bilstm_attention_cap125_seed3407
4. cnn_bilstm_mean_cap125_seed42 / cnn_bilstm_mean_cap125_seed123 / cnn_bilstm_mean_cap125_seed3407 (ablation)
5. random_forest_cap125_seed42 (baseline) -> `checkpoints/random_forest.joblib`
5b. rf_150e_sqrt_msl1_md20_seed42 (hyper-parameter study winner) -> `checkpoints/random_forest_optimized.joblib`
6. Data strategy studies: lstm_cap50, lstm_cap100, lstm_window20, lstm_window50
7. Hyperparameter studies: lstm_hp_lr0.0005, lstm_hp_lr0.002, lstm_hp_hidden32, lstm_hp_hidden128
8. Ablation study (cnn_bilstm_mean)
9. Smoke test (lstm_smoke)

Each run directory contains:
- config.json - experiment configuration
- summary.json - training summary with metrics
- metrics.json - validation metrics
- best_model.pt - trained model weights
- training_history.csv - per-epoch history
- best_predictions.csv - validation predictions

## Evaluation Results
### Official FD001 Test Set (100 engines)
- Transformer: RMSE 15.54, MAE 11.96, Bias +1.67, R2 0.860, NASA 363.22
- LSTM: RMSE 16.93, MAE 12.13, Bias -3.37, R2 0.834, NASA 453.96
- CNN-BiLSTM-Attention: RMSE 17.32, MAE 13.21, Bias -0.38, R2 0.826, NASA 507.22
- Random Forest: RMSE 18.35, MAE 13.79, Bias +3.07, R2 0.805, NASA 733.90

### Validation (3 seeds, mean ± std)
- LSTM: 13.71 ± 0.60
- Transformer: 14.21 ± 0.96
- Hybrid (mean pooling): 14.98 ± 0.65
- Random Forest: 18.55

## Installation
1. Clone repository or use as-is
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure NASA C-MAPSS FD001 files are in `data/raw/` (included in this submission)
4. Run preprocessing: `python -m src.datasets.preprocessing`
5. Run tests: `python -m pytest -q`
6. Train models: `python scripts/run_suite.py configs/experiments/main.json`
7. Run demo: `streamlit run demo/app.py --server.port 8501`
8. Verify without a browser: `python scripts/verify_models.py` and `python scripts/verify_gui.py`

## Stabilization
See `FINAL_PROJECT_STABILIZATION_REPORT.md` for the final fix list, the
verification output of every model and the known limitations.

## References
- NASA PHM Dataset: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
- C-MAPSS Documentation: https://github.com/NASA-Prognostics/CMAPSS