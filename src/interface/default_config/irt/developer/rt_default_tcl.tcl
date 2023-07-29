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
        -enable_output_gds_files 0 \
        -enable_idrc_interfaces 0 \
        -pa_max_iter_num 1 \
        -ra_initial_penalty 100 \
        -ra_penalty_drop_rate 0.8 \
        -ra_outer_max_iter_num 10 \
        -ra_inner_max_iter_num 10 \
        -gr_max_iter_num 1 \
        -ta_model_max_iter_num 1 \
        -ta_panel_max_iter_num 1 \
        -dr_model_max_iter_num 1 \
        -dr_box_max_iter_num 1 \
        -vr_max_iter_num 1


run_rt -flow "pa ra gr ta dr vr"

destroy_rt