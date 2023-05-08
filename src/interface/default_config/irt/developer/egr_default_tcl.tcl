set TECH_LEF_PATH ""
set LEF_PATH ""
set DEF_PATH ""

tech_lef_init -path $TECH_LEF_PATH
lef_init -path $LEF_PATH
def_init -path $DEF_PATH

run_egr -temp_directory_path "<temp_directory_path>" \
        -congestion_cell_x_pitch 15 \
        -congestion_cell_y_pitch 15 \
        -bottom_routing_layer "" \
        -top_routing_layer "" \
        -report_lower_remain_num -5 \
        -report_upper_remain_num 0 \
        -accuracy 2
