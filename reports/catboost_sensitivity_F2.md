# CatBoost F2 sensitivity

Validation average precision; mean and sample standard deviation over seeds 0, 1, 2. Each point varies one parameter from the time-CV-selected baseline. These scores use validation early stopping and are not final-test estimates.

| Parameter | Value | Mean AP | Std AP | Tuned baseline |
| --- | --- | ---: | ---: | --- |
| depth | 3 | 0.472985 | 0.008987 |  |
| depth | 4 | 0.504193 | 0.000459 |  |
| depth | 6 | 0.544887 | 0.002039 |  |
| depth | 8 | 0.565107 | 0.016040 | yes |
| depth | 10 | 0.598488 | 0.013367 |  |
| learning_rate | 0.01 | 0.481524 | 0.001777 |  |
| learning_rate | 0.03 | 0.524484 | 0.002430 |  |
| learning_rate | 0.05 | 0.548748 | 0.003210 |  |
| learning_rate | 0.1 | 0.565107 | 0.016040 | yes |
| learning_rate | 0.3 | 0.563956 | 0.030104 |  |
| l2_leaf_reg | 0.1 | 0.588298 | 0.008076 |  |
| l2_leaf_reg | 1.0 | 0.576630 | 0.012657 |  |
| l2_leaf_reg | 3.0 | 0.565107 | 0.016040 | yes |
| l2_leaf_reg | 10.0 | 0.552590 | 0.003668 |  |
| l2_leaf_reg | 30.0 | 0.508983 | 0.021793 |  |
| scale_pos_weight | 1.0 | 0.565107 | 0.016040 | yes |
| scale_pos_weight | 3.0 | 0.583156 | 0.013650 |  |
| scale_pos_weight | 10.0 | 0.582538 | 0.004194 |  |
| scale_pos_weight | 30.0 | 0.579641 | 0.002395 |  |
| scale_pos_weight | 100.0 | 0.543125 | 0.007126 |  |
| scale_pos_weight | balanced | 0.585619 | 0.001074 |  |
