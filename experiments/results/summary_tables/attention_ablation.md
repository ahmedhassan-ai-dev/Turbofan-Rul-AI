# Attention ablation (identical architecture, mean pooling instead of attention)

| variant | seeds | val_rmse_mean | val_rmse_std | val_mae_mean | val_nasa_mean |
| --- | --- | --- | --- | --- | --- |
| CNN-BiLSTM + attention | 3 | 14.9839 | 0.6516 | 11.8668 | 3.7923 |
| CNN-BiLSTM + mean pooling | 3 | 15.4687 | 0.1704 | 12.0757 | 3.9713 |
