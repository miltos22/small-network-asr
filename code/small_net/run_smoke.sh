#!/bin/bash
# Step 0.2 timing smoke: 1 epoch of S and R on full CIFAR (45k), seed 0, for the 20-min-per-run trigger.
Q=scratch/heartbeat; L=$Q/GPU.lockdir; I=${SCRATCH_DIR:-scratch}/base6
I=${SCRATCH_DIR:-scratch}/small_net; cd code/small_net
PY="python3 -I"; export CUDA_VISIBLE_DEVICES=0
take(){ QL="worker | $1 | $2 | 0.9 | $(date -Iseconds)"
  if [ -d $L ] || grep -qv '^#' $Q/GPU_QUEUE.txt; then echo "$QL" >> $Q/GPU_QUEUE.txt; echo "queued $1"
    until [ ! -d $L ] && [ "$(grep -v '^#' $Q/GPU_QUEUE.txt | head -1)" = "$QL" ] && mkdir $L 2>/dev/null; do sleep 60; done
    grep -vxF "$QL" $Q/GPU_QUEUE.txt > $Q/GPU_QUEUE.txt.tmp && mv $Q/GPU_QUEUE.txt.tmp $Q/GPU_QUEUE.txt
  else until mkdir $L 2>/dev/null; do sleep 60; done; fi
  OWN="worker-$1"; echo "$OWN $(date -Iseconds)" > $L/owner; echo "lock taken $1 $(date +%H:%M:%S)"; }
rel(){ [ -n "$OWN" ] && [ -f $L/owner ] && grep -q "^$OWN " $L/owner && { OWN=; rm -f $L/owner; rmdir $L 2>/dev/null; echo "lock released $(date +%H:%M:%S)"; }; }
run(){ f=$1_smoke_ep$2.log; nice -n 5 $PY cifar_q1.py $1 0 ep=$2 eval_every=1 > $I/logs/$f 2>&1; rc=$?; cp $I/logs/$f logs/; echo "$1 ep$2 rc $rc $(date +%H:%M:%S)"; tail -2 logs/$f; }
take CIFARsmoke 6
run S 1; run R 1; run S 2
rel; echo done $(date +%H:%M:%S)
