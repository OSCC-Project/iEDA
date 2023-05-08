# Marco place 
placeInst -inst_name u0_rcg/u0_pll -llx 280410 -lly 3621630 -orient MX -cellmaster S013PLLFN
placeInst -inst_name u0_rcg/u1_pll -llx 280410 -lly 2783000 -orient MX -cellmaster S013PLLFN
placeInst -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer11 -llx 3592620 -lly 3951710 -orient MX -cellmaster S011HD1P_X64Y4D32_BW
placeInst -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer12 -llx 2984150 -lly 3951710 -orient MX -cellmaster S011HD1P_X64Y4D32_BW
placeInst -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer21 -llx 3592620 -lly 3528950 -orient R0 -cellmaster S011HD1P_X64Y4D32_BW
placeInst -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer22 -llx 2984150 -lly 3528950 -orient R0 -cellmaster S011HD1P_X64Y4D32_BW
placeInst -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw/ram -llx 3334100 -lly 513930 -orient MX -cellmaster S011HD1P_X32Y2D128_BW
placeInst -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw_1/ram -llx 2386895 -lly 513930 -orient MX -cellmaster S011HD1P_X32Y2D128_BW
placeInst -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw_2/ram -llx 1439690 -lly 513930 -orient MX -cellmaster S011HD1P_X32Y2D128_BW
placeInst -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw_3/ram -llx 492485 -lly 513930 -orient MX -cellmaster S011HD1P_X32Y2D128_BW
placeInst -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw/ram -llx 3334100 -lly 1096670 -orient MX -cellmaster S011HD1P_X32Y2D128_BW
placeInst -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw_1/ram -llx 3334100 -lly 1679410 -orient MX -cellmaster S011HD1P_X32Y2D128_BW
placeInst -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw_2/ram -llx 492485 -lly 1096670 -orient MX -cellmaster S011HD1P_X32Y2D128_BW
placeInst -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw_3/ram -llx 492485 -lly 1679410 -orient MX -cellmaster S011HD1P_X32Y2D128_BW

# blockage
addPlacementBlk -box "159000 2680000 880500 4341000"
addPlacementHalo -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer11 -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer12 -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer21 -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer22 -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw/ram -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw_1/ram -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw_2/ram -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw_3/ram -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw/ram -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw_1/ram -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw_2/ram -distance "5000 5000 5000 5000"
addPlacementHalo -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw_3/ram -distance "5000 5000 5000 5000"

addRoutingHalo -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer11 -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer12 -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer21 -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_vga_ctrl/vga/buffer22 -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw/ram -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw_1/ram -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw_2/ram -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_ysyx_210539/dcache/Ram_bw_3/ram -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw/ram -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw_1/ram -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw_2/ram -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"
addRoutingHalo -inst_name u0_soc_top/u0_ysyx_210539/icache/Ram_bw_3/ram -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "5000 5000 5000 5000"

addRoutingBlk -layer "METAL1 METAL2 METAL3 METAL4 METAl5 METAl6 METAL7 METAL8 ALPA" -box "159900 2680000 880500 4341500" 
# addRoutingHalo -inst_name all -layer "METAL1 METAL2 METAL3 METAL4 METAL5 METAL6" -distance "6000 6000 6000 6000"

