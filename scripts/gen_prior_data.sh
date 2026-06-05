#!/usr/bin/env bash
# Generate a multi-op learned-prior dataset by tuning a spread of representative
# ops into ONE shared prior JSON + tune DB. The accumulated value-of-position
# rows feed scripts/train_prior.py (offline fit) and scripts/prior_bakeoff.py
# (model comparison).
#
# Env (override as needed):
#   PRIOR_OUT   prior JSON to accumulate into   (default /tmp/bakeoff/prior.json)
#   TUNE_DB     tune SQLite cache               (default /tmp/bakeoff/autotune.db)
#   PATIENCE    inner-search patience per op    (default 25)
#
# Usage:  bash scripts/gen_prior_data.sh
set -u
cd "$(dirname "$0")/.."
export DEPLODOCK_PRIOR_FILE="${PRIOR_OUT:-/tmp/bakeoff/prior.json}"
export DEPLODOCK_TUNE_DB="${TUNE_DB:-/tmp/bakeoff/autotune.db}"
mkdir -p "$(dirname "$DEPLODOCK_PRIOR_FILE")"
DD=./venv/bin/deplodock
PAT="${PATIENCE:-25}"

# name|code — ungated single-kernel ops spanning the structural archetypes
# (pointwise / row-reduction / row-softmax / matmul) at a couple of shapes each.
OPS=(
  "pointwise_gelu|torch.nn.GELU()(torch.randn(8,2048,2048))"
  "pointwise_add|(torch.randn(8,2048,2048)*1.5+torch.randn(8,2048,2048))"
  "reduce_rmsnorm|torch.nn.RMSNorm(2048)(torch.randn(1,64,2048))"
  "reduce_rmsnorm_wide|torch.nn.RMSNorm(4096)(torch.randn(1,32,4096))"
  "reduce_layernorm|torch.nn.LayerNorm(2048)(torch.randn(1,128,2048))"
  "softmax_row|torch.nn.Softmax(dim=-1)(torch.randn(8,1024,1024))"
  "softmax_wide|torch.nn.Softmax(dim=-1)(torch.randn(4,512,4096))"
  "matmul_square|torch.nn.Linear(2048,2048,bias=False)(torch.randn(512,2048))"
  "matmul_tall|torch.nn.Linear(1024,4096,bias=False)(torch.randn(2048,1024))"
)

first=1
for entry in "${OPS[@]}"; do
  name="${entry%%|*}"; code="${entry#*|}"
  clean=""; [ $first -eq 1 ] && clean="--clean"; first=0  # clean once: start fresh
  echo "######## TUNE $name $clean ########"
  ts=$(date +%s)
  timeout 400 $DD tune --code "$code" $clean --patience "$PAT" -q 2>&1 \
    | tr '\r' '\n' | grep -aiE "tune\] done|best:|calibration|Error|Traceback" | tail -4
  echo "  ($name took $(( $(date +%s) - ts ))s, tune exit ${PIPESTATUS[0]})"
done
echo "######## DONE -> $DEPLODOCK_PRIOR_FILE ########"
