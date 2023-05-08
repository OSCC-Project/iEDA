create_clock -period 50 -name tau2015_clk tau2015_clk

set_input_transition 0.080 -min -rise inp1 -clock tau2015_clk
set_input_transition 0.085 -min -fall inp1 -clock tau2015_clk
set_input_transition 0.090 -max -rise inp1 -clock tau2015_clk
set_input_transition 0.095 -max -fall inp1 -clock tau2015_clk
set_input_transition 0.100 -min -rise inp2 -clock tau2015_clk
set_input_transition 0.110 -min -fall inp2 -clock tau2015_clk
set_input_transition 0.120 -max -rise inp2 -clock tau2015_clk
set_input_transition 0.130 -max -fall inp2 -clock tau2015_clk
set_input_transition 0.110 -min -rise tau2015_clk -clock tau2015_clk
set_input_transition 0.110 -min -fall tau2015_clk -clock tau2015_clk
set_input_transition 0.110 -max -rise tau2015_clk -clock tau2015_clk
set_input_transition 0.110 -max -fall tau2015_clk -clock tau2015_clk
set_load 4 out

set_input_delay 0 -min -rise inp1 -clock tau2015_clk
set_input_delay 0 -min -fall inp1 -clock tau2015_clk
set_input_delay 5 -max -rise inp1 -clock tau2015_clk
set_input_delay 5 -max -fall inp1 -clock tau2015_clk
set_input_delay 0 -min -rise inp2 -clock tau2015_clk
set_input_delay 0 -min -fall inp2 -clock tau2015_clk
set_input_delay 1 -max -rise inp2 -clock tau2015_clk
set_input_delay 1 -max -fall inp2 -clock tau2015_clk

set_output_delay 1 -min -rise out -clock tau2015_clk
set_output_delay 1 -min -fall out -clock tau2015_clk
set_output_delay 6 -max -rise out -clock tau2015_clk
set_output_delay 6 -max -fall out -clock tau2015_clk



