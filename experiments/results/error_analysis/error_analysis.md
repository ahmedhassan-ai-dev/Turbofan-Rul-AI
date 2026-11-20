# Error Analysis — Official FD001 Test

All values below are computed from `experiments/results/official_test/*_predictions.csv`.

Horizon definition: Short <= 50 cycles, Medium 51-100, Long > 100 cycles.

## Overall metrics

| model | rmse | mae | bias | r2 | nasa_score | nasa_score_mean |
| --- | --- | --- | --- | --- | --- | --- |
| transformer | 15.5414 | 11.9635 | 1.6699 | 0.8601 | 363.2171 | 3.6322 |
| lstm | 16.9256 | 12.1273 | -3.3706 | 0.8341 | 453.9588 | 4.5396 |
| cnn_bilstm_attention | 17.3178 | 13.2126 | -0.3772 | 0.8263 | 507.217 | 5.0722 |
| random_forest | 18.3486 | 13.7916 | 3.069 | 0.805 | 733.901 | 7.339 |

## RMSE by RUL horizon

| model | horizon | n_engines | rmse | mae | bias |
| --- | --- | --- | --- | --- | --- |
| cnn_bilstm_attention | Long (>100) | 33 | 18.5503 | 14.4931 | -10.7706 |
| cnn_bilstm_attention | Medium (51-100) | 34 | 20.1259 | 15.6006 | 3.9882 |
| cnn_bilstm_attention | Short (<=50) | 33 | 12.1397 | 9.4717 | 5.5186 |
| lstm | Long (>100) | 33 | 20.0072 | 16.0631 | -10.3046 |
| lstm | Medium (51-100) | 34 | 20.3136 | 15.4679 | 1.0638 |
| lstm | Short (<=50) | 33 | 6.5325 | 4.7497 | -1.0052 |
| random_forest | Long (>100) | 33 | 19.6825 | 15.7045 | -10.2283 |
| random_forest | Medium (51-100) | 34 | 21.8634 | 17.786 | 13.14 |
| random_forest | Short (<=50) | 33 | 11.8456 | 7.7632 | 5.9902 |
| transformer | Long (>100) | 33 | 17.0818 | 13.4886 | -4.9008 |
| transformer | Medium (51-100) | 34 | 19.479 | 16.5737 | 6.3553 |
| transformer | Short (<=50) | 33 | 7.0145 | 5.6884 | 3.4133 |

## Worst 10 predictions (per model, by absolute error)

| model | engine_id | true_rul | predicted_rul | error | horizon |
| --- | --- | --- | --- | --- | --- |
| cnn_bilstm_attention | 93 | 85 | 36.0905 | -48.9095 | Medium (51-100) |
| cnn_bilstm_attention | 27 | 66 | 109.2799 | 43.2799 | Medium (51-100) |
| cnn_bilstm_attention | 12 | 124 | 81.4467 | -42.5533 | Long (>100) |
| cnn_bilstm_attention | 79 | 63 | 105.4342 | 42.4342 | Medium (51-100) |
| cnn_bilstm_attention | 25 | 145 | 110.9869 | -34.0131 | Long (>100) |
| cnn_bilstm_attention | 55 | 137 | 105.5515 | -31.4485 | Long (>100) |
| cnn_bilstm_attention | 89 | 136 | 104.5834 | -31.4166 | Long (>100) |
| cnn_bilstm_attention | 43 | 59 | 89.9369 | 30.9369 | Medium (51-100) |
| cnn_bilstm_attention | 51 | 114 | 84.6408 | -29.3592 | Long (>100) |
| cnn_bilstm_attention | 45 | 114 | 85.2849 | -28.7151 | Long (>100) |
| lstm | 51 | 114 | 64.5337 | -49.4663 | Long (>100) |
| lstm | 67 | 77 | 121.1192 | 44.1192 | Medium (51-100) |
| lstm | 25 | 145 | 101.9924 | -43.0076 | Long (>100) |
| lstm | 93 | 85 | 43.7201 | -41.2799 | Medium (51-100) |
| lstm | 79 | 63 | 99.8407 | 36.8407 | Medium (51-100) |
| lstm | 11 | 97 | 60.2591 | -36.7409 | Medium (51-100) |
| lstm | 3 | 69 | 34.9712 | -34.0288 | Medium (51-100) |
| lstm | 70 | 94 | 60.3599 | -33.6401 | Medium (51-100) |
| lstm | 45 | 114 | 81.1356 | -32.8644 | Long (>100) |
| lstm | 13 | 95 | 62.2993 | -32.7007 | Medium (51-100) |
| random_forest | 79 | 63 | 113.8944 | 50.8944 | Medium (51-100) |
| random_forest | 43 | 59 | 104.6835 | 45.6835 | Medium (51-100) |
| random_forest | 93 | 85 | 43.3021 | -41.6979 | Medium (51-100) |
| random_forest | 45 | 114 | 72.6694 | -41.3306 | Long (>100) |
| random_forest | 72 | 50 | 90.7326 | 40.7326 | Short (<=50) |
| random_forest | 55 | 137 | 99.5444 | -37.4556 | Long (>100) |
| random_forest | 50 | 79 | 114.9934 | 35.9934 | Medium (51-100) |
| random_forest | 12 | 124 | 88.6756 | -35.3244 | Long (>100) |
| random_forest | 27 | 66 | 100.1773 | 34.1773 | Medium (51-100) |
| random_forest | 25 | 145 | 111.7024 | -33.2976 | Long (>100) |
| transformer | 45 | 114 | 71.7492 | -42.2508 | Long (>100) |
| transformer | 30 | 115 | 72.8018 | -42.1982 | Long (>100) |
| transformer | 10 | 96 | 56.2694 | -39.7306 | Medium (51-100) |
| transformer | 86 | 89 | 123.2224 | 34.2224 | Medium (51-100) |
| transformer | 48 | 92 | 123.8614 | 31.8614 | Medium (51-100) |
| transformer | 93 | 85 | 54.1981 | -30.8019 | Medium (51-100) |
| transformer | 55 | 137 | 106.8437 | -30.1563 | Long (>100) |
| transformer | 6 | 93 | 122.5875 | 29.5875 | Medium (51-100) |
| transformer | 54 | 97 | 125.238 | 28.238 | Medium (51-100) |
| transformer | 2 | 98 | 124.1599 | 26.1599 | Medium (51-100) |

## How to read this

* Positive error means the model predicted more remaining life than the engine had (late prediction), which the NASA score penalises more heavily.
* Horizon tables use the *true* RUL, not the prediction, so a model cannot inflate its short-horizon score by predicting a small value everywhere.
