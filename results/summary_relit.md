**BIWI-A-relit: mean AP / AUC (%) ± std over 5 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 36.21±3.52 | **47.04±5.29** | **48.86±5.24** |
| AUC | 61.06±4.03 | **73.08±4.53** | **74.14±4.33** |

Selected hyper-parameters per round: ITML: {'gamma': 0.1}, {'gamma': 0.316}, {'gamma': 0.1}, {'gamma': 0.316}, {'gamma': 1.0}; ITML+: {'gamma': 0.316, 'lam': 10.0}, {'gamma': 0.316, 'lam': 1.0}, {'gamma': 0.1, 'lam': 0.316}, {'gamma': 0.316, 'lam': 1.0}, {'gamma': 1.0, 'lam': 10.0}

**BIWI-B-relit: mean AP / AUC (%) ± std over 5 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 61.53±2.99 | **68.79±3.64** | **69.77±4.01** |
| AUC | 59.40±3.14 | **68.80±4.07** | **69.20±4.40** |

Selected hyper-parameters per round: ITML: {'gamma': 0.1}, {'gamma': 0.1}, {'gamma': 0.0316}, {'gamma': 0.1}, {'gamma': 0.316}; ITML+: {'gamma': 0.316, 'lam': 31.6}, {'gamma': 0.1, 'lam': 0.01}, {'gamma': 0.0316, 'lam': 0.0316}, {'gamma': 0.1, 'lam': 0.1}, {'gamma': 0.1, 'lam': 0.1}

**BIWI-A-relit-20rounds: mean AP / AUC (%) ± std over 20 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 34.21±3.28 | 45.14±4.64 | **46.57±5.14** |
| AUC | 59.34±3.55 | **71.60±4.38** | **72.07±4.77** |

Selected hyper-parameters per round: ITML: {'gamma': 0.1}, {'gamma': 0.316}, {'gamma': 0.1}, {'gamma': 0.316}, {'gamma': 1.0}, {'gamma': 0.316}, {'gamma': 0.316}, {'gamma': 0.316}, {'gamma': 0.1}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 0.316}, {'gamma': 1.0}, {'gamma': 0.1}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 0.316}, {'gamma': 1.0}, {'gamma': 0.316}; ITML+: {'gamma': 0.316, 'lam': 10.0}, {'gamma': 0.316, 'lam': 1.0}, {'gamma': 0.1, 'lam': 0.316}, {'gamma': 0.316, 'lam': 1.0}, {'gamma': 1.0, 'lam': 10.0}, {'gamma': 1.0, 'lam': 100.0}, {'gamma': 0.316, 'lam': 3.16}, {'gamma': 0.316, 'lam': 3.16}, {'gamma': 0.0001, 'lam': 100.0}, {'gamma': 1.0, 'lam': 10.0}, {'gamma': 1.0, 'lam': 31.6}, {'gamma': 0.316, 'lam': 3.16}, {'gamma': 0.316, 'lam': 0.1}, {'gamma': 0.316, 'lam': 3.16}, {'gamma': 0.1, 'lam': 10.0}, {'gamma': 0.000316, 'lam': 31.6}, {'gamma': 1.0, 'lam': 3.16}, {'gamma': 0.316, 'lam': 1.0}, {'gamma': 0.0316, 'lam': 100.0}, {'gamma': 0.0001, 'lam': 31.6}

**BIWI-B-relit-20rounds: mean AP / AUC (%) ± std over 20 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 60.47±2.79 | 67.10±4.10 | **68.05±4.29** |
| AUC | 58.34±2.83 | **66.44±4.87** | **66.80±5.03** |

Selected hyper-parameters per round: ITML: {'gamma': 0.1}, {'gamma': 0.1}, {'gamma': 0.0316}, {'gamma': 0.1}, {'gamma': 0.316}, {'gamma': 0.1}, {'gamma': 0.0316}, {'gamma': 0.1}, {'gamma': 0.0316}, {'gamma': 0.316}, {'gamma': 0.1}, {'gamma': 0.1}, {'gamma': 0.0316}, {'gamma': 0.0316}, {'gamma': 0.0316}, {'gamma': 0.0316}, {'gamma': 0.1}, {'gamma': 0.0316}, {'gamma': 0.1}, {'gamma': 0.1}; ITML+: {'gamma': 0.316, 'lam': 31.6}, {'gamma': 0.1, 'lam': 0.01}, {'gamma': 0.0316, 'lam': 0.0316}, {'gamma': 0.1, 'lam': 0.1}, {'gamma': 0.1, 'lam': 0.1}, {'gamma': 0.1, 'lam': 0.01}, {'gamma': 0.0316, 'lam': 0.0316}, {'gamma': 0.316, 'lam': 3.16}, {'gamma': 0.0316, 'lam': 0.0316}, {'gamma': 0.0001, 'lam': 3.16}, {'gamma': 0.1, 'lam': 0.0316}, {'gamma': 0.1, 'lam': 0.1}, {'gamma': 0.0316, 'lam': 0.0316}, {'gamma': 0.0316, 'lam': 0.1}, {'gamma': 0.0316, 'lam': 0.0316}, {'gamma': 0.0316, 'lam': 0.316}, {'gamma': 0.1, 'lam': 0.0316}, {'gamma': 0.0316, 'lam': 0.1}, {'gamma': 0.0316, 'lam': 0.0316}, {'gamma': 0.0001, 'lam': 1.0}

**BIWI-A-nolight-control: mean AP / AUC (%) ± std over 5 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 55.93±7.04 | **73.10±6.75** | **71.98±8.42** |
| AUC | 74.57±5.87 | **85.97±4.06** | **85.82±4.25** |

Selected hyper-parameters per round: ITML: {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}, {'gamma': 1.0}; ITML+: {'gamma': 1.0, 'lam': 1.0}, {'gamma': 1.0, 'lam': 0.01}, {'gamma': 1.0, 'lam': 10.0}, {'gamma': 1.0, 'lam': 0.1}, {'gamma': 1.0, 'lam': 0.01}

**BIWI-B-nolight-control: mean AP / AUC (%) ± std over 5 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| AP | 73.15±5.06 | **82.60±2.30** | **82.48±2.33** |
| AUC | 69.75±6.34 | **80.55±2.50** | **80.43±2.61** |

Selected hyper-parameters per round: ITML: {'gamma': 0.1}, {'gamma': 1.0}, {'gamma': 0.316}, {'gamma': 0.316}, {'gamma': 0.316}; ITML+: {'gamma': 0.316, 'lam': 3.16}, {'gamma': 0.316, 'lam': 0.316}, {'gamma': 0.1, 'lam': 0.316}, {'gamma': 0.316, 'lam': 0.1}, {'gamma': 0.316, 'lam': 0.1}

**Re-identification: mean rank-1 (%) ± std over 5 rounds**

| | L2 | ITML | ITML+ |
|---|---|---|---|
| Cross (Walking analogue) | **27.76±2.41** | **27.76±4.49** | **26.40±4.50** |
| Seq (Still analogue) | **57.28±2.54** | **58.32±5.46** | **59.44±5.69** |

**ITML vs ITML+ with fewer training pairs (Fig. 3 analogue)**

| | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
|---|---|---|---|---|---|
| AP ITML+ | 47.51±6.08 | 48.84±5.59 | 48.47±5.36 | 48.97±5.61 | 48.86±5.24 |
| AP ITML | 46.36±5.67 | 47.33±5.65 | 47.41±5.33 | 47.26±5.14 | 47.04±5.29 |
| AUC ITML+ | 73.00±5.14 | 74.31±4.50 | 73.90±4.43 | 74.19±4.84 | 74.14±4.33 |
| AUC ITML | 72.37±4.91 | 73.30±4.72 | 73.32±4.58 | 73.13±4.50 | 73.08±4.53 |

**biwi_a_relit: test AP (%) per gamma, mean over rounds**

| gamma | 1e-04 | 3e-04 | 1e-03 | 3e-03 | 1e-02 | 3e-02 | 1e-01 | 3e-01 | 1e+00 |
|---|---|---|---|---|---|---|---|---|---|
| ITML | 36.21 | 36.22 | 36.28 | 36.63 | 38.20 | 42.00 | 46.89 | 47.54 | 46.76 |
| ITML+ (lambda by val.) | 48.40 | 48.40 | 48.40 | 48.41 | 48.46 | 48.45 | 48.58 | 48.81 | 48.46 |
| ITML+ (lambda=100) | 48.40 | 48.40 | 48.40 | 48.41 | 48.42 | 48.45 | 48.48 | 48.60 | 48.76 |

**biwi_b_relit: test AP (%) per gamma, mean over rounds**

| gamma | 1e-04 | 3e-04 | 1e-03 | 3e-03 | 1e-02 | 3e-02 | 1e-01 | 3e-01 | 1e+00 |
|---|---|---|---|---|---|---|---|---|---|
| ITML | 61.70 | 61.98 | 62.58 | 63.83 | 66.35 | 68.98 | 69.17 | 68.17 | 66.63 |
| ITML+ (lambda by val.) | 68.82 | 68.82 | 68.82 | 68.83 | 68.85 | 69.61 | 69.83 | 68.87 | 68.38 |
| ITML+ (lambda=100) | 68.24 | 68.24 | 68.24 | 68.24 | 68.25 | 68.26 | 68.32 | 68.52 | 68.87 |
