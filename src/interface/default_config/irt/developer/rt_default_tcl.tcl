set TECH_LEF_PATH ""
set LEF_PATH ""
set DEF_PATH ""

tech_lef_init -path $TECH_LEF_PATH
lef_init -path $LEF_PATH
def_init -path $DEF_PATH

init_rt -temp_directory_path "<temp_directory_path>" \
        -log_level 0 \
        -thread_number 8 \
        -bottom_routing_layer "" \
        -top_routing_layer "" \
        -layer_utilization_ratio "" \
        -enable_output_gds_files 0 \
        -resource_allocate_initial_penalty 100 \
        -resource_allocate_penalty_drop_rate 0.8 \
        -resource_allocate_outer_iter_num 10 \
        -resource_allocate_inner_iter_num 10

run_rt -flow "pa ra gr ta dr vr"

destroy_rt