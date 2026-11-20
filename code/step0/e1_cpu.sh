#!/bin/bash
# E1 harness v6.4 (Part C v6.4 C5 4; CPU, no GPU; round 4 C5.4 fixes): TS through base6sy.py -> base6s.train, DRAW 0, SUB_H 0.5, seed 3,
# 60-pass schedule cut by STOP_AFTER 6 (54 updates). A old rule (eval at pass 6), B old rule again, C EVAL_N=60 (eval at passes 1-6),
# D = C + BROKEN_E1 (the eval draws RNG: must diverge). Each arm: its own EMPTY checkpoint dir, a pinned environment (env -i), rc recorded.
cd code || exit 9
E=${SCRATCH_DIR:-scratch}/e1v64
[ -e $E ] && { echo "refuse: $E exists (fresh run only)"; exit 8; }
mkdir -p $E
for R in A B C D; do
  mkdir $E/ck_$R
  X=""; [ $R = C ] && X="EVAL_N=60"; [ $R = D ] && X="EVAL_N=60 BROKEN_E1=1"
  env -i HOME=$HOME PATH=/usr/bin:/bin DEV=cpu THREADS=4 RIGHT=12 NORM=causal MIX=A WARP=W AUG=noise DRAW=0 SUB_H=0.5 STOP_AFTER=6 \
      CKDIR=$E/ck_$R TRACE=$E/trace_$R.txt $X \
      python3 -I base6sy.py TS e1v64_$R 3 60 > $E/log_$R.txt 2>&1
  echo "$R rc $?" >> $E/status.txt
  env -i HOME=$HOME DEV=cpu > $E/env_$R.txt; echo "$X" >> $E/env_$R.txt
done
echo DONE >> $E/status.txt
