# Computational cost

| model | experiment_id | parameters | size_mb | training_seconds | training_device | latency_ms_batch1 | latency_ms_batch100 | throughput_engines_per_s | benchmark_device | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | lstm_cap125_seed123 | 56641 | 0.216 | 38.14 | cuda | 2.498 | 2.698 | 400.4 | cuda | 13.1157 |
| transformer | transformer_cap125_seed3407 | 103105 | 0.393 | 121.56 | cuda | 1.752 | 1.588 | 570.7 | cuda | 13.2268 |
| cnn_bilstm_attention | cnn_bilstm_attention_cap125_seed123 | 198209 | 0.756 | 74.33 | cuda | 4.923 | 6.203 | 203.1 | cuda | 14.4657 |
| random_forest | random_forest_cap125_seed42 | 606408 | 41.685 | 8.91 | cpu | 35.721 | 42.2 | 28 | cuda | 18.5496 |
