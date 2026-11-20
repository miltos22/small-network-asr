#!/bin/bash
# Step 0.2 baselines (MASTER_PLAN 4a.2): full CIFAR (45k) S seeds 0-2 + ResNet-32 seed 1, 200 epochs (smoke: ~3.6 s/epoch -> ~12-13 min/run, under the 20-min trigger).
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
run(){ f=$1_s$2_ep200.log; nice -n 5 $PY cifar_q1.py $1 $2 ep=200 eval_every=10 out=results/$1_s$2_ep200.json > $I/logs/$f 2>&1; rc=$?; cp $I/logs/$f logs/; echo "$1 s$2 rc $rc $(date +%H:%M:%S)"; tail -1 logs/$f; }
take CIFARbase 55
run S 0; run S 1; run S 2; run R 1
rel; echo done $(date +%H:%M:%S)
