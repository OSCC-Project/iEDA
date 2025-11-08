create_grid -layer_name "MET1" -net_name_power VDD -net_name_ground VSS -width 0.16 -offset 0
create_stripe -layer_name "MET4"  -net_name_power VDD -net_name_ground VSS -width 1   -pitch 16 -offset 0.5
create_stripe -layer_name "MET5"  -net_name_power VDD -net_name_ground VSS -width 1   -pitch 16 -offset 0.5

connect_two_layer -layers "MET1 MET4"
connect_two_layer -layers "MET4 MET5"
