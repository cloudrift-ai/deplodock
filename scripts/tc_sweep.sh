#!/usr/bin/env bash
# Profile one knob config for tensor-pipe utilization + elapsed time.
# Usage: tc_sweep.sh "KNOB_STRING"
set -u
KNOBS="$1"
M="torch.randn(2048,2048,dtype=torch.float16,device='cuda')"
export EMMY_KNOBS="$KNOBS"
out=$(ncu --target-processes all -k "regex:k_matmul" -c 1 \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
  ./venv/bin/emmy run --code "a=$M;b=$M;torch.matmul(a,b)" 2>&1)
if ! echo "$out" | grep -q "k_matmul"; then
  echo "FAILED: $KNOBS"
  echo "$out" | tail -5
  exit 1
fi
tc=$(echo "$out" | awk '/sm__pipe_tensor_cycles_active/ {print $NF}')
cyc=$(echo "$out" | awk '/sm__cycles_elapsed.avg /{gsub(/,/,"",$NF); print $NF}')
ghz=$(echo "$out" | awk '/sm__cycles_elapsed.avg.per_second/ {print $NF}')
sm=$(echo "$out" | awk '/sm__throughput.avg/ {print $NF}')
occ=$(echo "$out" | awk '/sm__warps_active.avg/ {print $NF}')
us=$(awk "BEGIN{printf \"%.1f\", $cyc/($ghz*1000)}")
printf "tc=%6s%%  sm=%6s%%  occ=%6s%%  t=%7sus  | %s\n" "$tc" "$sm" "$occ" "$us" "$KNOBS"
