#!/usr/bin/env bash
# Every experiment in the README, in order. Run from anywhere:
#   PYTHON=python ITMLPLUS_JOBS=8 bash scripts/run_all.sh
# Data is expected in ../data (see README); results go to results/, figures to figures/.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
P=../data/processed
A=$P/biwi_a_full_glbp.npz
B=$P/biwi_b_full_glbp.npz
step() { echo "=== $(date '+%F %T') $*"; "$PY" "$@" || echo "!!! FAILED: $*"; }

# data: BIWI face crops + G-LBP, and the re-lit / control copies
step scripts/prepare_biwi.py --tag full
step scripts/make_relit.py --name biwi_a_full
step scripts/make_relit.py --name biwi_b_full

# paper protocol (Tables I-IV, Fig. 2-4)
step scripts/run_verif.py --data $A --protocol eurocom --mode main --out verif_biwi_a_full.json
step scripts/run_verif.py --data $B --protocol curtin --mode main --out verif_biwi_b_full.json
step scripts/run_reid.py --rounds 5 --out reid_uw.json
step scripts/run_verif.py --data $B --protocol curtin --mode partial --out partial_biwi_b_full.json
step scripts/run_verif.py --data $A --protocol eurocom --mode trainsize --out trainsize_biwi_a_full.json
step scripts/analyze_distances.py
step scripts/run_timing.py --data $A --results verif_biwi_a_full.json --out timing_biwi_a_full.json

# sensitivity checks: wider gamma grid, 30 sweeps
step scripts/run_extended_gamma.py --dataset biwi_a --data $A --out ext_biwi_a_full.json
step scripts/run_extended_gamma.py --dataset biwi_b --data $B --out ext_biwi_b_full.json
step scripts/run_extended_gamma.py --dataset uw --out ext_uw.json
step scripts/run_sweeps.py --data $A --protocol eurocom --out sweeps30_biwi_a_full.json

# controlled experiment: random re-lighting of RGB (depth unchanged) and its control
step scripts/run_verif.py --data $P/biwi_a_full_relit_glbp.npz --protocol eurocom --mode main --out verif_biwi_a_relit.json
step scripts/run_verif.py --data $P/biwi_b_full_relit_glbp.npz --protocol curtin --mode main --out verif_biwi_b_relit.json
step scripts/run_verif.py --data $P/biwi_a_full_nolight_glbp.npz --protocol eurocom --mode main --out verif_biwi_a_nolight.json
step scripts/run_verif.py --data $P/biwi_b_full_nolight_glbp.npz --protocol curtin --mode main --out verif_biwi_b_nolight.json
step scripts/run_verif.py --data $P/biwi_b_full_relit_glbp.npz --protocol curtin --mode partial --out partial_biwi_b_relit.json
step scripts/run_verif.py --data $P/biwi_a_full_relit_glbp.npz --protocol eurocom --mode trainsize --out trainsize_biwi_a_relit.json
step scripts/run_reid.py --rounds 5 --relit --out reid_uw_relit.json

# rounds 5-19 for the key comparisons (20 rounds in total)
step scripts/run_more_rounds.py --data $A --protocol eurocom --out more_biwi_a_full.json
step scripts/run_more_rounds.py --data $B --protocol curtin --out more_biwi_b_full.json
step scripts/run_more_rounds.py --data $P/biwi_a_full_relit_glbp.npz --protocol eurocom --out more_biwi_a_relit.json
step scripts/run_more_rounds.py --data $P/biwi_b_full_relit_glbp.npz --protocol curtin --out more_biwi_b_relit.json

# upper bound: depth also available at test time (prints a table)
step scripts/diag_upper_bound.py --data $A --protocol eurocom

# tables (results/summary*.md) and figures
step scripts/summarize.py \
  --verif BIWI-A=verif_biwi_a_full.json BIWI-B=verif_biwi_b_full.json \
          BIWI-A-20rounds=verif_biwi_a_full.json+more_biwi_a_full.json \
          BIWI-B-20rounds=verif_biwi_b_full.json+more_biwi_b_full.json \
  --reid reid_uw.json --partial partial_biwi_b_full.json --trainsize trainsize_biwi_a_full.json \
  --grid biwi_a_full=verif_biwi_a_full.json:test:AP biwi_b_full=verif_biwi_b_full.json:test:AP \
         uw_cross=reid_uw.json:test_cross:rank1 uw_seq=reid_uw.json:test_seq:rank1 \
  --extended BIWI-A=ext_biwi_a_full.json:test:AP:test:AUC BIWI-B=ext_biwi_b_full.json:test:AP:test:AUC \
             UW=ext_uw.json:test_cross:rank1:test_seq:rank1 \
  --timing timing_biwi_a_full.json --out summary.md
step scripts/summarize.py \
  --verif BIWI-A-relit=verif_biwi_a_relit.json BIWI-B-relit=verif_biwi_b_relit.json \
          BIWI-A-relit-20rounds=verif_biwi_a_relit.json+more_biwi_a_relit.json \
          BIWI-B-relit-20rounds=verif_biwi_b_relit.json+more_biwi_b_relit.json \
          BIWI-A-nolight-control=verif_biwi_a_nolight.json BIWI-B-nolight-control=verif_biwi_b_nolight.json \
  --reid reid_uw_relit.json --partial partial_biwi_b_relit.json --trainsize trainsize_biwi_a_relit.json \
  --grid biwi_a_relit=verif_biwi_a_relit.json:test:AP biwi_b_relit=verif_biwi_b_relit.json:test:AP \
  --figtag _relit --out summary_relit.md
echo "=== $(date '+%F %T') all done"
