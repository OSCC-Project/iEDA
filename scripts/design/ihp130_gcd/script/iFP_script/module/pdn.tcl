add_pdn_io -net_name VDD   -direction INOUT -is_power 1 
add_pdn_io -net_name VSS   -direction INOUT -is_power 0

global_net_connect  -net_name VDD  -instance_pin_name VDD    -is_power 1
global_net_connect  -net_name VDD  -instance_pin_name VDDPE  -is_power 1
global_net_connect  -net_name VDD  -instance_pin_name VDDCE  -is_power 1
global_net_connect  -net_name VDD  -instance_pin_name vdd    -is_power 1
global_net_connect  -net_name VSS  -instance_pin_name VSS    -is_power 0
global_net_connect  -net_name VSS  -instance_pin_name VSSE   -is_power 0
global_net_connect  -net_name VSS  -instance_pin_name vss    -is_power 0

create_grid -layer_name "Metal1" -net_name_power VDD -net_name_ground VSS -width 0.44 -offset 0

create_stripe -layer_name "Metal5"    -net_name_power VDD -net_name_ground VSS -width 2.700 -pitch 75.6 -offset 13.600
create_stripe -layer_name "TopMetal1" -net_name_power VDD -net_name_ground VSS -width 1.800 -pitch 75.6 -offset 13.570

set connect1 "Metal1 Metal5"
set connect2 "Metal5 TopMetal1"

connect_two_layer -layers [concat $connect1 $connect2]

