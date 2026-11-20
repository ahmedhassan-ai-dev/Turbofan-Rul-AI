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
