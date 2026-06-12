SEC AAER publication results summary
====================================

Model comparison:
                           model model_family  roc_auc  pr_auc_average_precision  brier_score  n_train  n_test  positives_train  positives_test  lambda_ai  pos_weight  degree teacher_selected_on_validation  selection_metric
                LightGBM Teacher   AI teacher 0.727202                  0.031533     0.001619      NaN   24629              NaN              40        NaN         NaN     NaN                            NaN               NaN
            AI-guided VC Student AI-guided VC 0.728152                  0.006355     0.421233  76474.0   24629            186.0              40       0.05        20.0     3.0               LightGBM Teacher validation PR-AUC
VC Student (validation-selected) AI-guided VC 0.719150                  0.005171     0.147523  76474.0   24629            186.0              40       0.00         5.0     3.0               LightGBM Teacher validation PR-AUC
                        LightGBM     Baseline 0.704144                  0.004531     0.004452  76474.0   24629            186.0              40        NaN         NaN     NaN                            NaN               NaN
             Logistic Regression     Baseline 0.657158                  0.003440     0.228737  76474.0   24629            186.0              40        NaN         NaN     NaN                            NaN               NaN
                     Extra Trees     Baseline 0.587377                  0.002512     0.067581  76474.0   24629            186.0              40        NaN         NaN     NaN                            NaN               NaN
                   Random Forest     Baseline 0.563679                  0.002197     0.004433  76474.0   24629            186.0              40        NaN         NaN     NaN                            NaN               NaN

Top-screening table:
                           model screening_rule  selected_n  true_positives_found  precision  recall  lift_over_random
                LightGBM Teacher         Top 1%         246                     4   0.016260   0.100         10.011789
                LightGBM Teacher         Top 5%        1231                     8   0.006499   0.200          4.001462
                LightGBM Teacher        Top 10%        2463                    10   0.004060   0.250          2.499898
                LightGBM Teacher        Top 100         100                     2   0.020000   0.050         12.314500
                LightGBM Teacher        Top 250         250                     4   0.016000   0.100          9.851600
                LightGBM Teacher        Top 500         500                     5   0.010000   0.125          6.157250
             Logistic Regression         Top 1%         246                     0   0.000000   0.000          0.000000
             Logistic Regression         Top 5%        1231                     7   0.005686   0.175          3.501279
             Logistic Regression        Top 10%        2463                    10   0.004060   0.250          2.499898
             Logistic Regression        Top 100         100                     0   0.000000   0.000          0.000000
             Logistic Regression        Top 250         250                     0   0.000000   0.000          0.000000
             Logistic Regression        Top 500         500                     3   0.006000   0.075          3.694350
                   Random Forest         Top 1%         246                     1   0.004065   0.025          2.502947
                   Random Forest         Top 5%        1231                     3   0.002437   0.075          1.500548
                   Random Forest        Top 10%        2463                     6   0.002436   0.150          1.499939
                   Random Forest        Top 100         100                     0   0.000000   0.000          0.000000
                   Random Forest        Top 250         250                     1   0.004000   0.025          2.462900
                   Random Forest        Top 500         500                     1   0.002000   0.025          1.231450
                     Extra Trees         Top 1%         246                     1   0.004065   0.025          2.502947
                     Extra Trees         Top 5%        1231                     4   0.003249   0.100          2.000731
                     Extra Trees        Top 10%        2463                     5   0.002030   0.125          1.249949
                     Extra Trees        Top 100         100                     0   0.000000   0.000          0.000000
                     Extra Trees        Top 250         250                     1   0.004000   0.025          2.462900
                     Extra Trees        Top 500         500                     2   0.004000   0.050          2.462900
                        LightGBM         Top 1%         246                     2   0.008130   0.050          5.005894
                        LightGBM         Top 5%        1231                     6   0.004874   0.150          3.001097
                        LightGBM        Top 10%        2463                     8   0.003248   0.200          1.999919
                        LightGBM        Top 100         100                     1   0.010000   0.025          6.157250
                        LightGBM        Top 250         250                     2   0.008000   0.050          4.925800
                        LightGBM        Top 500         500                     4   0.008000   0.100          4.925800
VC Student (validation-selected)         Top 1%         246                     3   0.012195   0.075          7.508841
VC Student (validation-selected)         Top 5%        1231                     8   0.006499   0.200          4.001462
VC Student (validation-selected)        Top 10%        2463                    14   0.005684   0.350          3.499858
VC Student (validation-selected)        Top 100         100                     0   0.000000   0.000          0.000000
VC Student (validation-selected)        Top 250         250                     3   0.012000   0.075          7.388700
VC Student (validation-selected)        Top 500         500                     6   0.012000   0.150          7.388700
            AI-guided VC Student         Top 1%         246                     3   0.012195   0.075          7.508841
            AI-guided VC Student         Top 5%        1231                    10   0.008123   0.250          5.001828
            AI-guided VC Student        Top 10%        2463                    13   0.005278   0.325          3.249868
            AI-guided VC Student        Top 100         100                     1   0.010000   0.025          6.157250
            AI-guided VC Student        Top 250         250                     3   0.012000   0.075          7.388700
            AI-guided VC Student        Top 500         500                     7   0.014000   0.175          8.620150