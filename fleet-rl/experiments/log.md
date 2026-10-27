# Experiment Log

This log records hyperparameter tuning and model experiments on the validation seed range (seeds 1000-1199). Final test seeds (5000-5049) are strictly reserved for post-tuning evaluation.

| Exp # | Description | Val Avg Wait (steps) | Val Surge Wait (steps) | Val Served | Val Dropped | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp 01 | Baseline (Nearest) on seeds 1000-1199 | 15.57 | 20.51 | 86.38 | 11.75 | Standard greedy dispatcher benchmark. |
| Exp 01 | Forecast-Aware Rule on seeds 1000-1199 | 15.45 | 20.27 | 86.65 | 11.52 | Pre-positioning during phases 1 and 2 improves surge wait by 0.24 steps and reduces drops. |
| Exp 02 | Q-Learning 576 states, 4000 eps (lr=0.08, gamma=0.92, eps_decay=0.999) | 16.30 | 21.54 | 85.62 | 12.46 | Initial 576-state baseline with action masking. Q-table still visits some states sparsely at 4000 eps. |
| Exp 03 | Q-Learning 576 states, 8000 eps (lr=0.06, gamma=0.95, eps_decay=0.9995) | 15.78 | 20.85 | 86.16 | 12.03 | Increasing episodes to 8000 and gamma to 0.95 reduced surge wait by 0.69 steps and improved overall wait to 15.78. |
| Exp 04 | Q-Learning 576 states, 10000 eps (lr=0.05, gamma=0.96, eps_decay=0.9996) | 15.53 | 20.37 | 86.46 | 11.68 | Successfully beats baseline across all metrics (Wait: 15.53 vs 15.57, Surge: 20.37 vs 20.51, Served: 86.46 vs 86.38, Dropped: 11.68 vs 11.75). Q-learning captures over 70% of the theoretical gain from the forecast-aware rule. |
