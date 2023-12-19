create_clock -name CLK_osc_in -period 1 -waveform { 0 0.5 } [get_ports {osc_in}]
create_clock -name CLK_chiplink_rx_clk -period 40 -waveform { 0 20 } [get_ports {chiplink_rx_clk}]

create_generated_clock -name CLK_chiplink_tx_clk -source [get_ports {osc_in}]  -divide_by 1 [get_ports {chiplink_tx_clk}] 
set_clock_groups -asynchronous -name CLK_osc_in_1 -group [get_clocks {CLK_osc_in CLK_chiplink_tx_clk}] -group [get_clocks {CLK_chiplink_rx_clk}]

set_max_fanout  32 [current_design]

