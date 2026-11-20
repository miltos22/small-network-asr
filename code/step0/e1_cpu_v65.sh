#!/bin/bash
# E1 harness v6.5 (rule: step0/e1_compare_v65.py, written before this runs). CPU only. Two lanes in parallel: lane 1 = TS, S, R (0.5 h),
# lane 2 = P1, G1 (1 h). Each arm: own EMPTY checkpoint dir, pinned env (env -i), rc recorded. Recipe env as v6.4 (MIX=A WARP=W AUG=noise).
cd code || exit 9
E=${SCRATCH_DIR:-scratch}/e1v65
[ -e $E ] && { echo "refuse: $E exists (fresh run only)"; exit 8; }
mkdir -p $E
one() {  # path arms subh script argsprefix extraenv
  P=$1; ARMS=$2; SUBH=$3; SCRIPT=$4; PRE=$5; XE=$6; mkdir -p $E/$P
  for R in $(echo $ARMS | grep -o .); do
    mkdir $E/$P/ck_$R
    X=""; [ $R = C ] && X="EVAL_N=60"; [ $R = D ] && X="EVAL_N=60 BROKEN_E1=1"
    env -i HOME=$HOME PATH=/usr/bin:/bin DEV=cpu THREADS=4 RIGHT=12 NORM=causal MIX=A WARP=W AUG=noise DRAW=0 SUB_H=$SUBH STOP_AFTER=6 \
        CKDIR=$E/$P/ck_$R TRACE=$E/$P/trace_$R.txt $XE $X \
        python3 -I $SCRIPT $PRE e1v65_${P}_$R 3 60 > $E/$P/log_$R.txt 2>&1
    echo "$R rc $?" >> $E/$P/status.txt
  done
  echo DONE >> $E/$P/status.txt
}
( one TS ABCD 0.5 base6sy.py TS ""; one S ACD 0.5 base6s.py S ""; one R ACD 0.5 base6s_R.py "" "RWIRE_SEED=3" ) &
L1=$!
( one P1 ACD 1 base6s.py P ""; one G1 ACD 1 base6s.py G "" ) &
L2=$!
wait $L1; wait $L2; echo ALLDONE >> $E/status.txt
