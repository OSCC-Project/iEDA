add_pdn_io -net_name VDD   -direction INOUT -is_power 1 
add_pdn_io -net_name VSS   -direction INOUT -is_power 0

global_net_connect -net_name VDD     -instance_pin_name VPWR      -is_power 1
global_net_connect -net_name VDD     -instance_pin_name VPB       -is_power 1
global_net_connect -net_name VDD     -instance_pin_name vdd       -is_power 1
global_net_connect -net_name VSS     -instance_pin_name VGND      -is_power 0
global_net_connect -net_name VSS     -instance_pin_name VNB       -is_power 0
global_net_connect -net_name VSS     -instance_pin_name gnd       -is_power 0


