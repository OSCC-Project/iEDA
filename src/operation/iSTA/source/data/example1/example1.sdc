create_clock -period 50 -name clk1 [get_ports clk1]
create_clock -period 10 -name clk2 [get_ports clk2]
create_clock -period 5 -name clk3 [get_ports clk3]

set_input_delay 10 -clock [get_clocks clk1] [get_ports in1]
set_input_delay 2 -clock [get_clocks clk2] [get_ports in2]

set_output_delay 1 -clock [get_clocks clk3] [get_ports out]