# C9 6 (v6.5, 20 Nov 2026, worker): the ONLY way a Tranche 1 runner launches GPU work. Source it, then:
#   take <name> <est_min>   queue + GPU.lockdir (copied from code/small_net/run_base.sh); exports GATE_OWNER=worker-<name>
#   gjob <ledger_row_key> <cmd ...>   = gpu_gate2.py run (cap / spend / lock-owner checks, own process group, allowance timeout, one charge)
#   rel                     releases the lock only if this runner owns it
# Launch audit: step0/audit_launch.py refuses a runner that starts a GPU python any other way.
Q=${GQ_TEST:-scratch/heartbeat}; L=$Q/GPU.lockdir  # GQ_TEST only for the lib smoke; the gate always checks the real lock
GATE=code/step0/gpu_gate2.py
GPY="python3 -I"
take(){ QL="worker | $1 | $2 | 0.9 | $(date -Iseconds)"
  if [ -d $L ] || grep -qv '^#' $Q/GPU_QUEUE.txt; then echo "$QL" >> $Q/GPU_QUEUE.txt; echo "queued $1"
    until [ ! -d $L ] && [ "$(grep -v '^#' $Q/GPU_QUEUE.txt | head -1)" = "$QL" ] && mkdir $L 2>/dev/null; do sleep 60; done
    grep -vxF "$QL" $Q/GPU_QUEUE.txt > $Q/GPU_QUEUE.txt.tmp && mv $Q/GPU_QUEUE.txt.tmp $Q/GPU_QUEUE.txt
  else until mkdir $L 2>/dev/null; do sleep 60; done; fi
  OWN="worker-$1"; echo "$OWN $(date -Iseconds)" > $L/owner; export GATE_OWNER="$OWN"; echo "lock taken $1 $(date +%H:%M:%S)"; }
rel(){ [ -n "$OWN" ] && [ -f $L/owner ] && grep -q "^$OWN " $L/owner && { OWN=; unset GATE_OWNER; rm -f $L/owner; rmdir $L 2>/dev/null; echo "lock released $(date +%H:%M:%S)"; }; }
gjob(){ local k=$1; shift; $GPY $GATE run "$k" -- "$@"; local rc=$?; echo "gjob $k rc $rc $(date +%H:%M:%S)"; return $rc; }
