add_pdn_io -net_name VDD   -direction INOUT -is_power 1 
add_pdn_io -net_name VDDIO   -direction INOUT -is_power 1 
add_pdn_io -net_name VSS   -direction INOUT -is_power 0
add_pdn_io -net_name VSSIO   -direction INOUT -is_power 0

global_net_connect  -net_name VDD  -instance_pin_name VDD1    -is_power 1
global_net_connect  -net_name VDD  -instance_pin_name VDD  -is_power 1
global_net_connect  -net_name VDD  -instance_pin_name VNW  -is_power 1
global_net_connect  -net_name VSS  -instance_pin_name VSS1    -is_power 0
global_net_connect  -net_name VSS  -instance_pin_name VSS   -is_power 0
global_net_connect  -net_name VSS  -instance_pin_name VPW    -is_power 0
global_net_connect  -net_name VDDIO  -instance_pin_name VDDIO    -is_power 1
global_net_connect  -net_name VSSIO  -instance_pin_name VSSIO    -is_power 0

