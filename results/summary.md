**BIWI-A: mean AP / AUC (%) ± std over 5 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 55.91±7.05 | **73.07±6.66** | **71.93±8.34** |
| AUC | 74.58±5.93 | **86.00±4.01** | **85.85±4.19** |

Selected hyper-parameters per round: ITML: {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}; ITML+: {'gamma': 1.0, 'lam': 1.0}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 10.0}, {'gamma': 1.0, 'lam': 0.1}, {'gamma': 1.0, 'lam': 0.01}

**BIWI-B: mean AP / AUC (%) ± std over 5 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 73.14±5.09 | **82.48±2.35** | **82.34±2.40** |
| AUC | 69.73±6.38 | **80.55±2.35** | **80.37±2.59** |

Selected hyper-parameters per round: ITML: {'gamma': 0.1}, {'gamma': 1.0}, {'gamma': 0.1}, {'gamma': 0.316}, {'gamma': 0.316}; ITML+: {'gamma': 0.1, 'lam': 1.0}, {'gamma': 0.316, 'lam': 0.316}, {'gamma': 0.1, 'lam': 0.316}, {'gamma': 0.316, 'lam': 0.1}, {'gamma': 0.316, 'lam': 0.1}

**BIWI-A-20rounds: mean AP / AUC (%) ± std over 20 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 55.59±7.99 | **72.99±8.37** | **72.56±8.61** |
| AUC | 74.23±5.94 | **85.20±5.79** | 85.11±5.79 |

Selected hyper-parameters per round: ITML: {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}; ITML+: {'gamma': 1.0, 'lam': 1.0}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 10.0}, {'gamma': 1.0, 'lam': 0.1}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 0.316}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 0.0316}, {'gamma': 1.0, 'lam': 3.16}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 0.1}, {'gamma': 1.0, 'lam': 3.16}, {'gamma': 1.0, 'lam': 0.0316}, {'gamma': 1.0, 'lam': 0.316}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 0.1}, {'gamma': 1.0, 'lam': 0.0316}

**BIWI-B-20rounds: mean AP / AUC (%) ± std over 20 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 71.63±5.85 | **80.50±6.14** | **80.29±6.19** |
| AUC | 68.15±6.06 | **78.54±6.88** | **78.26±6.95** |

Selected hyper-parameters per round: ITML: {'gamma': 0.1}, {'gamma': 1.0}, {'gamma': 0.1}, {'gamma': 0.316}, {'gamma': 0.316}, {'gamma': 0.316}, {'gamma': 0.0316}, {'gamma': 0.1}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 0.0316}, {'gamma': 0.316}, {'gamma': 0.316}, {'gamma': 0.316}, {'gamma': 0.1}, {'gamma': 0.316}, {'gamma': 0.1}, {'gamma': 0.316}; ITML+: {'gamma': 0.1, 'lam': 1.0}, {'gamma': 0.316, 'lam': 0.316}, {'gamma': 0.1, 'lam': 0.316}, {'gamma': 0.316, 'lam': 0.1}, {'gamma': 0.316, 'lam': 0.1}, {'gamma': 0.316, 'lam': 0.0316}, {'gamma': 0.0316, 'lam': 0.1}, {'gamma': 0.1, 'lam': 0.01}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 0.316, 'lam': 0.1}, {'gamma': 1.0, 'lam': 0.316}, {'gamma': 1.0, 'lam': 0.316}, {'gamma': 0.0316, 'lam': 0.01}, {'gamma': 0.316, 'lam': 0.1}, {'gamma': 0.316, 'lam': 0.0316}, {'gamma': 0.0316, 'lam': 0.0316}, {'gamma': 0.1, 'lam': 0.1}, {'gamma': 0.1, 'lam': 0.0316}, {'gamma': 0.1, 'lam': 0.0316}, {'gamma': 0.316, 'lam': 0.0316}

**Re-identification: mean rank-1 (%) ± std over 5 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| Cross (Walking analogue) | **47.76±3.98** | **46.72±2.85** | **48.64±4.57** |
| Seq (Still analogue) | 82.56±1.31 | **85.60±1.88** | **85.76±1.46** |

**ITML vs ITML+ with fewer training pairs (Fig. 3 analogue)**

| | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
|---|---|---|---|---|---|
| AP ITML+ | 69.57±7.88 | 71.19±7.85 | 71.45±8.30 | 72.00±7.48 | 71.93±8.34 |
| AP ITML | 69.93±7.70 | 72.12±7.20 | 72.73±6.83 | 72.43±7.05 | 73.07±6.66 |
| AUC ITML+ | 84.49±4.31 | 85.52±4.04 | 85.63±4.19 | 85.94±3.85 | 85.85±4.19 |
| AUC ITML | 84.69±4.24 | 85.72±4.09 | 85.93±4.01 | 85.87±3.87 | 86.00±4.01 |

**biwi_a_full: test AP (%) per gamma, mean over rounds**

| gamma | 1e-04 | 3e-04 | 1e-03 | 3e-03 | 1e-02 | 3e-02 | 1e-01 | 3e-01 | 1e+00 |
|---|---|---|---|---|---|---|---|---|---|
| ITML | 55.95 | 56.05 | 56.35 | 57.21 | 59.40 | 63.78 | 69.39 | 72.34 | 73.07 |
| ITML+ (lambda by val.) | 66.42 | 66.42 | 66.42 | 66.42 | 66.44 | 66.48 | 68.98 | 71.39 | 71.93 |
| ITML+ (lambda=100) | 66.42 | 66.42 | 66.42 | 66.42 | 66.44 | 66.48 | 66.59 | 66.84 | 67.45 |

**biwi_b_full: test AP (%) per gamma, mean over rounds**

| gamma | 1e-04 | 3e-04 | 1e-03 | 3e-03 | 1e-02 | 3e-02 | 1e-01 | 3e-01 | 1e+00 |
|---|---|---|---|---|---|---|---|---|---|
| ITML | 73.28 | 73.53 | 74.16 | 75.59 | 78.09 | 80.94 | 82.43 | 82.46 | 82.17 |
| ITML+ (lambda by val.) | 80.06 | 80.06 | 80.07 | 80.09 | 79.74 | 80.76 | 82.41 | 82.34 | 81.98 |
| ITML+ (lambda=100) | 80.06 | 80.06 | 80.07 | 80.09 | 80.17 | 80.35 | 80.70 | 81.22 | 81.56 |

**uw_cross: test rank1 (%) per gamma, mean over rounds**

| gamma | 1e-04 | 3e-04 | 1e-03 | 3e-03 | 1e-02 | 3e-02 | 1e-01 | 3e-01 | 1e+00 |
|---|---|---|---|---|---|---|---|---|---|
| ITML | 47.76 | 47.92 | 48.72 | 50.00 | 50.56 | 50.72 | 48.56 | 46.88 | 42.96 |
| ITML+ (lambda by val.) | 48.88 | 48.88 | 49.04 | 49.28 | 49.76 | 50.80 | 49.68 | 47.52 | 44.72 |
| ITML+ (lambda=100) | 48.24 | 48.24 | 48.16 | 48.16 | 48.16 | 48.32 | 47.60 | 46.96 | 45.36 |

**uw_seq: test rank1 (%) per gamma, mean over rounds**

| gamma | 1e-04 | 3e-04 | 1e-03 | 3e-03 | 1e-02 | 3e-02 | 1e-01 | 3e-01 | 1e+00 |
|---|---|---|---|---|---|---|---|---|---|
| ITML | 82.64 | 82.64 | 82.80 | 83.36 | 84.32 | 85.44 | 85.44 | 85.36 | 85.52 |
| ITML+ (lambda by val.) | 83.76 | 83.76 | 83.76 | 83.92 | 83.68 | 85.44 | 84.88 | 85.44 | 85.36 |
| ITML+ (lambda=100) | 83.68 | 83.68 | 83.76 | 83.76 | 83.92 | 84.00 | 84.32 | 84.40 | 83.92 |

**BIWI-A, extended gamma grid (1e-4 ... 1e2): mean (%) ± std**

| | ITML | ITML+ |
|---|---|---|
| test AP | **73.43±6.83** | **72.22±8.58** |
| test AUC | **86.09±4.15** | **85.92±4.23** |

Selected hyper-parameters per round: ITML: {'gamma': 100.0}, {'gamma': 100.0}, {'gamma': 31.6}, {'gamma': 1.0}, {'gamma': 10.0}; ITML+: {'gamma': 100.0, 'lam': 100.0}, {'gamma': 100.0, 'lam': 0.01}, {'gamma': 10.0, 'lam': 100.0}, {'gamma': 1.0, 'lam': 0.1}, {'gamma': 10.0, 'lam': 0.01}

**BIWI-B, extended gamma grid (1e-4 ... 1e2): mean (%) ± std**

| | ITML | ITML+ |
|---|---|---|
| test AP | **82.50±2.36** | **82.34±2.40** |
| test AUC | **80.51±2.39** | **80.37±2.59** |

Selected hyper-parameters per round: ITML: {'gamma': 0.1}, {'gamma': 3.16}, {'gamma': 0.1}, {'gamma': 0.316}, {'gamma': 0.316}; ITML+: {'gamma': 0.1, 'lam': 1.0}, {'gamma': 0.316, 'lam': 0.316}, {'gamma': 0.1, 'lam': 0.316}, {'gamma': 0.316, 'lam': 0.1}, {'gamma': 0.316, 'lam': 0.1}

**UW, extended gamma grid (1e-4 ... 1e2): mean (%) ± std**

| | ITML | ITML+ |
|---|---|---|
| test_cross rank1 | **46.72±2.85** | **48.64±4.57** |
| test_seq rank1 | **85.60±1.88** | **85.76±1.46** |

Selected hyper-parameters per round: ITML: {'gamma': 1.0}, {'gamma': 0.316}, {'gamma': 1.0}, {'gamma': 0.1}, {'gamma': 1.0}; ITML+: {'gamma': 1.0, 'lam': 0.01}, {'gamma': 0.316, 'lam': 0.1}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 0.1, 'lam': 0.01}, {'gamma': 0.316, 'lam': 0.1}

**Training time (s), single CPU thread (Table IV analogue)**

| | ITML | ITML+ |
|---|---|---|
| Time | 0.69±0.15 | 1.46±0.22 |
