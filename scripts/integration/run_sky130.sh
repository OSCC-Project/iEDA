#!/usr/bin/env bash
export HOME=/home/lxq
export MAMBA_ROOT_PREFIX=/home/lxq/AiEDA/micromamba
eval "$(/home/lxq/.local/bin/micromamba shell hook -s bash)"
micromamba activate ieda-build
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd /home/lxq/AiEDA/iEDA/scripts/design/sky130_gcd
# make ./iEDA resolve to the built binary
ln -sf /home/lxq/AiEDA/iEDA/bin/iEDA ./iEDA
echo "=== sky130 gcd full flow (netlist->GDS) ==="
bash run_iEDA.sh
echo "RUN_IEDA_EXIT=$?"
echo "=== result artifacts ==="
ls -la result/ 2>&1 | head -30
