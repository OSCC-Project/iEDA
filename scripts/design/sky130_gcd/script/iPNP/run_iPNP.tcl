#===========================================================
##   init flow config
#===========================================================
flow_init -config /home/sujianrong/iEDA/scripts/design/sky130_gcd/iEDA_config/flow_config.json

#===========================================================
##   read db config
#===========================================================
db_init -config /home/sujianrong/iEDA/scripts/design/sky130_gcd/iEDA_config/db_default_config.json

#===========================================================
##   read lef
#===========================================================
source /home/sujianrong/iEDA/scripts/design/sky130_gcd/script/DB_script/db_init_lef.tcl

#===========================================================
##   read lib
#===========================================================
source /home/sujianrong/iEDA/scripts/design/sky130_gcd/script/DB_script/db_init_lib.tcl

#===========================================================
##   read sdc
#===========================================================
source /home/sujianrong/iEDA/scripts/design/sky130_gcd/script/DB_script/db_init_sdc.tcl

#===========================================================
##   read def (use DEF from PNP config)
#===========================================================
def_init -path /home/sujianrong/iEDA/src/operation/iPNP/data/test/aes_no_pwr.def

#===========================================================
##   run PNP
#===========================================================
run_pnp -config /home/sujianrong/iEDA/scripts/design/sky130_gcd/iEDA_config/pnp_default_config.json

#===========================================================
##   def & netlist
#===========================================================
def_save -path /home/sujianrong/iEDA/scripts/design/sky130_gcd/result/pnp/iPNP_result.def

#===========================================================
##   save netlist 
#===========================================================
netlist_save -path /home/sujianrong/iEDA/scripts/design/sky130_gcd/result/pnp/iPNP_result.v -exclude_cell_names {}

#===========================================================
##   Exit 
#===========================================================
flow_exit
