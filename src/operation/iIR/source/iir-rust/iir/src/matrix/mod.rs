pub mod ir_inst_power;
pub mod ir_rc;

pub fn read_matrix_raw_data(inst_power: &str, power_net_spef: &str) {
    ir_rc::read_rc_data_from_spef(power_net_spef);
}
