#!/usr/bin/env bash
# Regression: run sky130 + ics55 gcd flows against current bin/iEDA, verify GDS.

export HOME=/home/lxq
export MAMBA_ROOT_PREFIX=/home/lxq/AiEDA/micromamba
eval "$(/home/lxq/.local/bin/micromamba shell hook -s bash)"
micromamba activate ieda-build
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export FOUNDRY_ROOT=/home/lxq/AiEDA/Foundary
unset PDK_DIR
ROOT=/home/lxq/AiEDA/iEDA
LOGDIR=$ROOT/scripts/integration/regress_logs
mkdir -p "$LOGDIR"
BIN=$ROOT/bin/iEDA
echo "=== iEDA binary ==="; ls -la "$BIN"

run_flow () {
  local name=$1 dir=$2 gds=$3
  echo; echo "############################################################"
  echo "### FLOW: $name   ($dir)"
  echo "############################################################"
  cd "$ROOT/scripts/design/$dir" || return 2
  rm -f result/*.gds result/*.gds2 2>/dev/null
  ln -sf "$BIN" ./iEDA
  local t0=$SECONDS
  bash run_iEDA.sh > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  local dt=$((SECONDS-t0))
  echo "$name run_iEDA.sh exit=$rc  elapsed=${dt}s"
  echo "--- final GDS ($gds) ---"
  ls -la result/$gds 2>&1
  if ls result/$gds >/dev/null 2>&1; then echo "$name: GDS_OK"; else echo "$name: GDS_MISSING"; echo "--- last 25 log lines ---"; tail -25 "$LOGDIR/$name.log"; fi
}

run_flow sky130 sky130_gcd "final_design.gds2"
run_flow ics55  ics55_gcd  "*.gds*"
echo; echo "=== REGRESSION DONE ==="
