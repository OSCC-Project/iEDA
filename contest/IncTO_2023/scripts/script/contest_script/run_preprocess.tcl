#===========================================================
##   init flow config from sky130 gcd scripts
#===========================================================
flow_init -config ./../../../scripts/design/sky130_gcd/iEDA_config/flow_config.json

#===========================================================
##   read db config from sky130 gcd scripts
#===========================================================
db_init -config ./../../../scripts/design/sky130_gcd/iEDA_config/db_default_config.json

#===========================================================
##   reset data path
#===========================================================
source ./script/DB_script/db_path_setting.tcl

#===========================================================
##   reset lib
#===========================================================
source ./script/DB_script/db_init_lib.tcl

#===========================================================
##   reset sdc
#===========================================================
source ./script/DB_script/db_init_sdc.tcl

#===========================================================
##   read lef
#===========================================================
source ./script/DB_script/db_init_lef.tcl

#===========================================================
##   read def
#===========================================================
#def_init -path /opt/benchmark/sky130/def/asic_top_1025.def
#def_init -path /home/guofan/contest/Contest_data/contest_1/asic_top.def
def_init -path /home/guofan/contest/Contest_data/contest_1/result/output/output.def

#===========================================================
##   run contest
#===========================================================
run_preprocess -gcell_x 22920 -gcell_y 29730 -guide_output ./result/output/output.guide

#===========================================================
##   def & netlist
#===========================================================
def_save -path ./result/output/output.def

#===========================================================
##   save netlist 
#===========================================================
netlist_save -path ./result/output/output.v -exclude_cell_names {}

#===========================================================
##   report db summary
#===========================================================
report_db -path "./result/report/contest.rpt"

#===========================================================
##   Exit 
#===========================================================
flow_exit
