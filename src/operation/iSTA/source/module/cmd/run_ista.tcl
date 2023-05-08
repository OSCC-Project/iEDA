set_design_workspace "/var/lib/jenkins/ysyx/"

read_verilog 1214/asic_top.v

set LIB_FILES { \
  1101/lib/S011HD1P1024X64M4B0_SS_1.08_125.lib  \
  1101/lib/S011HD1P128X21M2B0_SS_1.08_125.lib  \
  1101/lib/S011HD1P256X8M4B0_SS_1.08_125.lib \
  1101/lib/S011HD1P512X19M4B0_SS_1.08_125.lib \
  1101/lib/S011HD1P512X73M2B0_SS_1.08_125.lib \
  1101/lib/S011HDSP4096X64M8B0_SS_1.08_125.lib \
  1101/lib/S013PLLFN_v1.5.1_typ.lib \
  1101/lib/SP013D3WP_V1p7_typ.lib \
  1101/lib/SP013D3WP_V1p7_typ1.lib \
  1101/lib/scc011ums_hd_hvt_ss_v1p08_125c_ccs.lib \
  1101/lib/scc011ums_hd_lvt_ss_v1p08_125c_ccs.lib \
  1101/lib/scc011ums_hd_rvt_ss_v1p08_125c_ccs.lib \
}

foreach LIB_FILE $LIB_FILES { \
    read_liberty $LIB_FILE    \
}

link_design asic_top

read_sdc 1214/asic_top.sdc
read_spef 1214/asic_top.spef

report_timing

