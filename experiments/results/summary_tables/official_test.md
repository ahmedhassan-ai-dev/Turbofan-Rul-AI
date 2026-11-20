# Official FD001 test results

| model | experiment_id | rmse | mae | bias | r2 | nasa_score | nasa_score_mean | rmse_capped_125 | mae_capped_125 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | transformer | 15.5414 | 11.9635 | 1.6699 | 0.8601 | 363.2171 | 3.6322 | 14.7073 | 10.8706 |
| lstm | lstm | 16.9256 | 12.1273 | -3.3706 | 0.8341 | 453.9588 | 4.5396 | 15.6458 | 11.0422 |
| cnn_bilstm_attention | cnn_bilstm_attention | 17.3178 | 13.2126 | -0.3772 | 0.8263 | 507.217 | 5.0722 | 16.0229 | 12.1426 |
| random_forest | random_forest | 18.3486 | 13.7916 | 3.069 | 0.805 | 733.901 | 7.339 | 17.1179 | 12.7216 |
