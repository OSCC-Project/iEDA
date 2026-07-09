#!/usr/bin/env bash
export HOME=/home/lxq
export MAMBA_ROOT_PREFIX=/home/lxq/AiEDA/micromamba
eval "$(/home/lxq/.local/bin/micromamba shell hook -s bash)"
micromamba activate ieda-build
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export FOUNDRY_ROOT=/home/lxq/AiEDA/Foundary
unset PDK_DIR
cd /home/lxq/AiEDA/iEDA/scripts/design/sky130_gcd
ln -sf /home/lxq/AiEDA/iEDA/bin/iEDA ./iEDA
rm -f result/iPNP_result.def result/iPNP_result.v
echo "=== sky130 flow + iPNP + iPW ==="
bash run_iEDA.sh
echo "RUN_EXIT=$?"
