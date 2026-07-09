#!/usr/bin/env bash
export HOME=/home/lxq
export MAMBA_ROOT_PREFIX=/home/lxq/AiEDA/micromamba
eval "$(/home/lxq/.local/bin/micromamba shell hook -s bash)"
micromamba activate ieda-build
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PDK_DIR=/home/lxq/AiEDA/ics55-pdk
cd /home/lxq/AiEDA/iEDA/scripts/design/ics55_gcd
ln -sf /home/lxq/AiEDA/iEDA/bin/iEDA ./iEDA
echo "=== ics55 gcd flow (PDK_DIR=$PDK_DIR) ==="
echo "=== netlist input? ==="; ls -la result/verilog/*.v 2>&1 | head -2
bash run_iEDA.sh
echo "ICS55_RUN_EXIT=$?"
echo "=== artifacts ==="; ls -la result/*.gds* result/*.def 2>&1 | head -15
