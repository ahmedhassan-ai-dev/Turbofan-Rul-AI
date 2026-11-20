# Validation summary per model (main runs, 3 seeds)

| model | val_rmse_mean | val_rmse_std | val_rmse_min | val_rmse_max | seeds |
| --- | --- | --- | --- | --- | --- |
| lstm | 13.7052 | 0.6039 | 13.1157 | 14.3226 | 3 |
| transformer | 14.206 | 0.9554 | 13.2268 | 15.1358 | 3 |
| cnn_bilstm_attention | 14.9839 | 0.6516 | 14.4657 | 15.7154 | 3 |
| random_forest | 18.5496 | - | 18.5496 | 18.5496 | 1 |


# All registered experiments

| experiment_id | tag | model | seed | window_size | rul_cap | learning_rate | batch_size | epochs_ran | best_epoch | val_rmse | val_mae | val_bias | train_seconds | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_bilstm_mean_cap125_seed42 | ablation | cnn_bilstm_mean | 42 | 30 | 125 | 0.001 | 64 | 19 | 9 | 15.278 | 11.687 | -0.1301 | 108.23 | completed |
| cnn_bilstm_mean_cap125_seed123 | ablation | cnn_bilstm_mean | 123 | 30 | 125 | 0.001 | 64 | 13 | 3 | 15.5221 | 11.7684 | 0.6602 | 70.44 | completed |
| cnn_bilstm_mean_cap125_seed3407 | ablation | cnn_bilstm_mean | 3407 | 30 | 125 | 0.001 | 64 | 15 | 5 | 15.6061 | 12.7717 | 5.1159 | 81.54 | completed |
| lstm_cap50_seed42 | data_strategy | lstm | 42 | 30 | 50 | 0.001 | 64 | 18 | 8 | 29.9551 | 22.4686 | -19.3595 | 51.9 | completed |
| lstm_cap100_seed42 | data_strategy | lstm | 42 | 30 | 100 | 0.001 | 64 | 12 | 2 | 11.3387 | 8.5986 | 0.1918 | 33.11 | completed |
| lstm_window20_cap125_seed42 | data_strategy | lstm | 42 | 20 | 125 | 0.001 | 64 | 19 | 9 | 16.1363 | 12.679 | 4.0755 | 46.21 | completed |
| lstm_window50_cap125_seed42 | data_strategy | lstm | 42 | 50 | 125 | 0.001 | 64 | 15 | 5 | 11.6048 | 9.1441 | 3.025 | 69.66 | completed |
| lstm_hp_lr0.0005_seed42 | hyperparameter | lstm | 42 | 30 | 125 | 0.0005 | 64 | 19 | 9 | 14.124 | 11.2397 | 3.0172 | 53.24 | completed |
| lstm_hp_lr0.002_seed42 | hyperparameter | lstm | 42 | 30 | 125 | 0.002 | 64 | 18 | 8 | 14.6938 | 11.6349 | 3.0576 | 52.36 | completed |
| lstm_hp_hidden32_seed42 | hyperparameter | lstm | 42 | 30 | 125 | 0.001 | 64 | 19 | 9 | 13.8721 | 10.2964 | 3.3568 | 54.1 | completed |
| lstm_hp_hidden128_seed42 | hyperparameter | lstm | 42 | 30 | 125 | 0.001 | 64 | 13 | 3 | 14.1432 | 10.8533 | 3.9384 | 38.11 | completed |
| cnn_bilstm_attention_cap125_seed42 | main | cnn_bilstm_attention | 42 | 30 | 125 | 0.001 | 64 | 13 | 3 | 14.7706 | 11.7638 | 0.4515 | 75.27 | completed |
| cnn_bilstm_attention_cap125_seed123 | main | cnn_bilstm_attention | 123 | 30 | 125 | 0.001 | 64 | 13 | 3 | 14.4657 | 11.5662 | 0.1837 | 74.33 | completed |
| cnn_bilstm_attention_cap125_seed3407 | main | cnn_bilstm_attention | 3407 | 30 | 125 | 0.001 | 64 | 12 | 2 | 15.7154 | 12.2705 | 0.2471 | 68.27 | completed |
| lstm_cap125_seed42 | main | lstm | 42 | 30 | 125 | 0.001 | 64 | 15 | 5 | 13.6773 | 10.5706 | 3.4073 | 46.21 | completed |
| lstm_cap125_seed123 | main | lstm | 123 | 30 | 125 | 0.001 | 64 | 13 | 3 | 13.1157 | 10.3073 | -0.0116 | 38.14 | completed |
| lstm_cap125_seed3407 | main | lstm | 3407 | 30 | 125 | 0.001 | 64 | 12 | 2 | 14.3226 | 11.0825 | 4.7616 | 34.72 | completed |
| random_forest_cap125_seed42 | main | random_forest | 42 | 30 | 125 | - | - | 1 | 1 | 18.5496 | 13.3507 | 4.6463 | 8.51 | completed |
| transformer_cap125_seed42 | main | transformer | 42 | 30 | 125 | 0.001 | 64 | 15 | 5 | 15.1358 | 11.3293 | 3.2787 | 45.17 | completed |
| transformer_cap125_seed123 | main | transformer | 123 | 30 | 125 | 0.001 | 64 | 13 | 3 | 14.2555 | 10.8994 | 4.0361 | 38.15 | completed |
| transformer_cap125_seed3407 | main | transformer | 3407 | 30 | 125 | 0.001 | 64 | 41 | 31 | 13.2268 | 9.8475 | 3.5535 | 121.56 | completed |
| lstm_smoke | smoke | lstm | 42 | 30 | 125 | 0.001 | 64 | 2 | 2 | 21.1699 | 17.332 | 8.1026 | 6.1 | completed |


# Official FD001 test results

| model | experiment_id | rmse | mae | bias | r2 | nasa_score | nasa_score_mean | rmse_capped_125 | mae_capped_125 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | transformer | 15.5414 | 11.9635 | 1.6699 | 0.8601 | 363.2171 | 3.6322 | 14.7073 | 10.8706 |
| lstm | lstm | 16.9256 | 12.1273 | -3.3706 | 0.8341 | 453.9588 | 4.5396 | 15.6458 | 11.0422 |
| cnn_bilstm_attention | cnn_bilstm_attention | 17.3178 | 13.2126 | -0.3772 | 0.8263 | 507.217 | 5.0722 | 16.0229 | 12.1426 |
| random_forest | random_forest | 18.3486 | 13.7916 | 3.069 | 0.805 | 733.901 | 7.339 | 17.1179 | 12.7216 |


# Attention ablation (identical architecture, mean pooling instead of attention)

| variant | seeds | val_rmse_mean | val_rmse_std | val_mae_mean | val_nasa_mean |
| --- | --- | --- | --- | --- | --- |
| CNN-BiLSTM + attention | 3 | 14.9839 | 0.6516 | 11.8668 | 3.7923 |
| CNN-BiLSTM + mean pooling | 3 | 15.4687 | 0.1704 | 12.0757 | 3.9713 |


# Computational cost

| model | experiment_id | parameters | size_mb | training_seconds | training_device | latency_ms_batch1 | latency_ms_batch100 | throughput_engines_per_s | benchmark_device | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | lstm_cap125_seed123 | 56641 | 0.216 | 38.14 | cuda | 2.498 | 2.698 | 400.4 | cuda | 13.1157 |
| transformer | transformer_cap125_seed3407 | 103105 | 0.393 | 121.56 | cuda | 1.752 | 1.588 | 570.7 | cuda | 13.2268 |
| cnn_bilstm_attention | cnn_bilstm_attention_cap125_seed123 | 198209 | 0.756 | 74.33 | cuda | 4.923 | 6.203 | 203.1 | cuda | 14.4657 |
| random_forest | random_forest_cap125_seed42 | 606408 | 41.685 | 8.91 | cpu | 35.721 | 42.2 | 28 | cuda | 18.5496 |


# RMSE by RUL horizon (official test)

| horizon | cnn_bilstm_attention | lstm | random_forest | transformer |
| --- | --- | --- | --- | --- |
| Long (>100) | 18.5503 | 20.0072 | 19.6825 | 17.0818 |
| Medium (51-100) | 20.1259 | 20.3136 | 21.8634 | 19.479 |
| Short (<=50) | 12.1397 | 6.5325 | 11.8456 | 7.0145 |
