#[cfg(test)]
pub mod test_fun_mod;
mod tests {
    use super::*;

    #[test]
    fn test_rule_identifier(){
        test_fun_mod::rule_identifier();
    }

    #[test]
    fn test_rule_sdf_version_pair(){
        test_fun_mod::rule_sdf_version_pair();
    }

    #[test]
    fn test_rule_design_name_pair(){
        test_fun_mod::rule_design_name_pair();
    }

    #[test]
    fn test_rule_date_pair(){
        test_fun_mod::rule_date_pair();
    }

    #[test]
    fn test_rule_vendor_pair(){
        test_fun_mod::rule_vendor_pair();
    }

    #[test]
    fn test_rule_program_name_pair(){
        test_fun_mod::rule_program_name_pair();
    }

    #[test]
    fn test_rule_version_pair(){
        test_fun_mod::rule_version_pair();
    }

    #[test]
    fn test_rule_divider_pair(){
        test_fun_mod::rule_divider_pair();
    }
    
    #[test]
    fn test_rule_voltage_pair(){
        test_fun_mod::rule_voltage_pair();
    }

    #[test]
    fn test_rule_process_pair(){
        test_fun_mod::rule_process_pair();
    }

    #[test]
    fn test_rule_temperature_pair(){
        test_fun_mod::rule_temperature_pair();
    }

    #[test]
    fn test_rule_time_scale_pair(){
        test_fun_mod::rule_time_scale_pair();
    }

    #[test]
    fn test_rule_number(){
        test_fun_mod::rule_number();
    }

    #[test]
    fn test_rule_temperature(){
        test_fun_mod::rule_temperature();
    }

    #[test]
    fn test_rule_sdf_header(){
        test_fun_mod::rule_sdf_header();
    }

    #[test]
    fn test_rule_celltype(){
        test_fun_mod::rule_celltype();
    }

    #[test]
    fn test_rule_cell_instance(){
        test_fun_mod::rule_cell_instance();
    }

    #[test]
    fn test_rule_input_output_path(){
        test_fun_mod::rule_input_output_path();
    }

    #[test]
    fn test_rule_path_pulse_percent(){
        test_fun_mod::rule_path_pulse_percent();
    }

    #[test]
    fn test_rule_scalar_port(){
        test_fun_mod::rule_scalar_port();
    }

    #[test]
    fn test_rule_bus_port(){
        test_fun_mod::rule_bus_port();
    }

    #[test]
    fn test_rule_port_instance(){
        test_fun_mod::rule_port_instance();
    }

    #[test]
    fn test_rule_port_spec(){
        test_fun_mod::rule_port_spec();
    }

    #[test]
    fn test_rule_iopath_part1(){
        test_fun_mod::rule_iopath_part1();
    }

    #[test]
    fn test_rule_rvalue(){
        test_fun_mod::rule_rvalue();
    }

    #[test]
    fn test_rule_delval(){
        test_fun_mod::rule_delval();
    }

    #[test]
    fn test_rule_delval_list(){
        test_fun_mod::rule_delval_list();
    }

    #[test]
    fn test_rule_iopath_part2_part(){
        test_fun_mod::rule_iopath_part2_part();
    }

    #[test]
    fn test_rule_iopath_part2(){
        test_fun_mod::rule_iopath_part2();
    }

    #[test]
    fn test_rule_iopath_item(){
        test_fun_mod::rule_iopath_item();
    }

    #[test]
    fn test_rule_cond_item(){
        test_fun_mod::rule_cond_item();
    }

    #[test]
    fn test_rule_condelse_item(){
        test_fun_mod::rule_condelse_item();
    }

    #[test]
    fn test_rule_port_item(){
        test_fun_mod::rule_port_item();
    }

    #[test]
    fn test_rule_interconnect_item(){
        test_fun_mod::rule_interconnect_item();
    }

    #[test]
    fn test_rule_device_item(){
        test_fun_mod::rule_device_item();
    }

    #[test]
    fn test_rule_del_def(){
        test_fun_mod::rule_del_def();
    }

    #[test]
    fn test_rule_absolute_increment(){
        test_fun_mod::rule_absolute_increment();
    }

    #[test]
    fn test_rule_deltype(){
        test_fun_mod::rule_deltype();
    }

    #[test]
    fn test_rule_del_spec(){
        test_fun_mod::rule_del_spec();
    }

    #[test]
    fn test_rule_concat_expression(){
        test_fun_mod::rule_concat_expression();
    }

    #[test]
    fn test_rule_simple_expression_type1(){
        test_fun_mod::rule_simple_expression_type1();
    }

    #[test]
    fn test_rule_simple_expression_type2(){
        test_fun_mod::rule_simple_expression_type2();
    }

    #[test]
    fn test_rule_simple_expression_type3(){
        test_fun_mod::rule_simple_expression_type3();
    }

    #[test]
    fn test_rule_simple_expression_type4(){
        test_fun_mod::rule_simple_expression_type4();
    }

    #[test]
    fn test_rule_simple_expression_type5(){
        test_fun_mod::rule_simple_expression_type5();
    }

    #[test]
    fn test_rule_simple_expression_type6(){
        test_fun_mod::rule_simple_expression_type6();
    }

    #[test]
    fn test_rule_simple_expression_type7(){
        test_fun_mod::rule_simple_expression_type7();
    }

    #[test]
    fn test_rule_simple_expression_type8(){
        test_fun_mod::rule_simple_expression_type8();
    }

    #[test]
    fn test_rule_simple_expression_type9(){
        test_fun_mod::rule_simple_expression_type9();
    }

    #[test]
    fn test_rule_simple_expression(){
        test_fun_mod::rule_simple_expression();
    }

    #[test]
    fn test_rule_conditional_port_expr_type1(){
        test_fun_mod::rule_conditional_port_expr_type1();
    }

    #[test]
    fn test_rule_conditional_port_expr_type2(){
        test_fun_mod::rule_conditional_port_expr_type2();
    }

    #[test]
    fn test_rule_conditional_port_expr_type3(){
        test_fun_mod::rule_conditional_port_expr_type3();
    }

    #[test]
    fn test_rule_conditional_port_expr_type4(){
        test_fun_mod::rule_conditional_port_expr_type4();
    }

    #[test]
    fn test_rule_conditional_port_expr(){
        test_fun_mod::rule_conditional_port_expr();
    }

    #[test]
    fn test_rule_setup_hold_recovery_removal_item(){
        test_fun_mod::rule_setup_hold_recovery_removal_item();
    }

    #[test]
    fn test_rule_setuphold_item1_recrem_item1_nochange_item(){
        test_fun_mod::rule_setuphold_item1_recrem_item1_nochange_item();
    }

    #[test]
    fn test_rule_setuphold_item2_recrem_item2_item(){
        test_fun_mod::rule_setuphold_item2_recrem_item2_item();
    }

    #[test]
    fn test_rule_skew_item(){
        test_fun_mod::rule_skew_item();
    }

    #[test]
    fn test_rule_width_period_item(){
        test_fun_mod::rule_width_period_item();
    }

    #[test]
    fn test_rule_sccond(){
        test_fun_mod::rule_sccond();
    }

    #[test]
    fn test_rule_port_tchk_type1(){
        test_fun_mod::rule_port_tchk_type1();
    }

    #[test]
    fn test_rule_port_tchk_type2(){
        test_fun_mod::rule_port_tchk_type2();
    }
    
    #[test]
    fn test_rule_port_tchk(){
        test_fun_mod::rule_port_tchk();
    }

    #[test]
    fn test_rule_timing_check_condition_type1(){
        test_fun_mod::rule_timing_check_condition_type1();
    }

    #[test]
    fn test_rule_timing_check_condition_type2(){
        test_fun_mod::rule_timing_check_condition_type2();
    }

    #[test]
    fn test_rule_timing_check_condition_type3(){
        test_fun_mod::rule_timing_check_condition_type3();
    }

    #[test]
    fn test_rule_timing_check_condition(){
        test_fun_mod::rule_timing_check_condition();
    }

    #[test]
    fn test_rule_scalar_node(){
        test_fun_mod::rule_scalar_node();
    }

    #[test]
    fn test_rule_scalar_net(){
        test_fun_mod::rule_scalar_net();
    }

    #[test]
    fn test_rule_tchk_def(){
        test_fun_mod::rule_tchk_def();
    }

    #[test]
    fn test_rule_tc_spec(){
        test_fun_mod::rule_tc_spec();
    }

    #[test]
    fn test_rule_name(){
        test_fun_mod::rule_name();
    }

    #[test]
    fn test_rule_exception(){
        test_fun_mod::rule_exception();
    }

    #[test]
    fn test_rule_constraint_path(){
        test_fun_mod::rule_constraint_path();
    }

    #[test]
    fn test_rule_path_constraint_item(){
        test_fun_mod::rule_path_constraint_item();
    }

    #[test]
    fn test_rule_period_constraint_item(){
        test_fun_mod::rule_period_constraint_item();
    }

    #[test]
    fn test_rule_sum_item(){
        test_fun_mod::rule_sum_item();
    }

    #[test]
    fn test_rule_diff_item(){
        test_fun_mod::rule_diff_item();
    }

    #[test]
    fn test_rule_skew_constraint_item(){
        test_fun_mod::rule_skew_constraint_item();
    }

    #[test]
    fn test_rule_cns_def(){
        test_fun_mod::rule_cns_def();
    }

    #[test]
    fn test_rule_arrival_departure_item(){
        test_fun_mod::rule_arrival_departure_item();
    }

    #[test]
    fn test_rule_slack_item(){
        test_fun_mod::rule_slack_item();
    }

    #[test]
    fn test_rule_waveform_item(){
        test_fun_mod::rule_waveform_item();
    }

    #[test]
    fn test_rule_tenv_def(){
        test_fun_mod::rule_tenv_def();
    }

    #[test]
    fn test_rule_te_def(){
        test_fun_mod::rule_te_def();
    }

    #[test]
    fn test_rule_te_spec(){
        test_fun_mod::rule_te_spec();
    }

    #[test]
    fn test_rule_pos_neg_pair_posedge(){
        test_fun_mod::rule_pos_neg_pair_posedge();
    }

    #[test]
    fn test_test_rule_pos_neg_pair_negedge(){
        test_fun_mod::rule_pos_neg_pair_negedge();
    }

    #[test]
    fn test_rule_pos_pair(){
        test_fun_mod::rule_pos_pair();
    }

    #[test]
    fn test_rule_neg_pair(){
        test_fun_mod::rule_neg_pair();
    }

    #[test]
    fn test_rule_edge_list_type1(){
        test_fun_mod::rule_edge_list_type1();
    }

    #[test]
    fn test_rule_edge_list_type2(){
        test_fun_mod::rule_edge_list_type2();
    }

    #[test]
    fn test_rule_edge_list(){
        test_fun_mod::rule_edge_list();
    }

    #[test]
    fn test_rule_timing_spec(){
        test_fun_mod::rule_timing_spec();
    }

    #[test]
    fn test_rule_cell(){
        test_fun_mod::rule_cell();
    }

    #[test]
    fn test_rule_delay_file(){
        test_fun_mod::rule_delay_file();
    }
}

