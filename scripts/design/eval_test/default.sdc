set clk_name  core_clock
set clk_port_name clk
set clk_expect_freq_mhz 200
set clk_period [expr 1000.0 / $clk_expect_freq_mhz]
set clk_io_pct 0.2

set clk_port [get_ports $clk_port_name]

create_clock -name $clk_name -period $clk_period  $clk_port
