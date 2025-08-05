# add_pdn_io -net_name VDD   -direction INOUT -is_power 1 
# add_pdn_io -net_name VSS   -direction INOUT -is_power 0

global_net_connect -net_name VDD     -instance_pin_name VPWR      -is_power 1
global_net_connect -net_name VDD     -instance_pin_name VPB       -is_power 1
global_net_connect -net_name VDD     -instance_pin_name vdd       -is_power 1
global_net_connect -net_name VSS     -instance_pin_name VGND      -is_power 0
global_net_connect -net_name VSS     -instance_pin_name VNB       -is_power 0
global_net_connect -net_name VSS     -instance_pin_name gnd       -is_power 0

create_grid -layer_name met1 -net_name_power VDD -net_name_ground VSS -width 0.48 

create_stripe -layer_name met4 -net_name_power VDD -net_name_ground VSS -width 1.60 -pitch 27.14 -offset 13.57
create_stripe -layer_name met5 -net_name_power VDD -net_name_ground VSS -width 1.60 -pitch 27.20 -offset 13.60

set connect1 "met1 met4"
set connect2 "met4 met5"

connect_two_layer -layers [concat $connect1 $connect2]

