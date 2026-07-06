#!/usr/bin/env bash
export HOME=/home/lxq
export MAMBA_ROOT_PREFIX=/home/lxq/AiEDA/micromamba
eval "$(/home/lxq/.local/bin/micromamba shell hook -s bash)"; micromamba activate ecc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PDK_DIR=/home/lxq/AiEDA/ics55-pdk
W=/home/lxq/AiEDA/iEDA/scripts/design/ics55_gcd; cd $W
export WORKSPACE=$W RESULT_DIR=$W/result IEDA_CONFIG_DIR=$W/iEDA_config TCL_SCRIPT_DIR=$W/script IEDA_TCL_SCRIPT_DIR=$W/script SDC_FILE=$W/default.sdc
export TECH_LEF="${PDK_DIR}/prtech/techLEF/N551P6M_ieda.lef"
export LEF_STDCELL="${PDK_DIR}/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/lef/ics55_LLSC_H7CR_ieda.lef ${PDK_DIR}/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/lef/ics55_LLSC_H7CL_ieda.lef"
export LIB_STDCELL="${PDK_DIR}/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib ${PDK_DIR}/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/liberty/ics55_LLSC_H7CR_ss_rcworst_1p08_125_nldm.lib"
ln -sf /home/lxq/AiEDA/iEDA/bin/iEDA ./iEDA
echo "=== ics55 STA + Power ==="; ./iEDA -script /home/lxq/AiEDA/ics55_sta_power.tcl 2>&1 | tail -6
echo "=== ics55 DRC ==="; ./iEDA -script /home/lxq/AiEDA/ics55_drc.tcl 2>&1 | tail -6
echo "=== outputs ==="; ls -la result/sta/ result/report/drc.rpt 2>&1 | head
