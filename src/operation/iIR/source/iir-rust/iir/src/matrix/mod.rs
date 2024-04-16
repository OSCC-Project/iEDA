pub mod ir_inst_power;
pub mod ir_rc;

use log;

extern crate nalgebra as na;
use na::Vector;

pub fn build_matrix_from_raw_data(inst_power_path: &str, power_net_spef: &str) {
    let rc_data = ir_rc::read_rc_data_from_spef(power_net_spef);

    for (net_name, one_net_data) in rc_data.get_power_nets_data() {
        log::info!("construct power net {} matrix start", net_name);
        let conductance_matrix = ir_rc::build_conductance_matrix(one_net_data);
        let current_vector =
            ir_inst_power::build_instance_current_vector(inst_power_path, one_net_data);
        log::info!("construct power net {} matrix finish", net_name);
    }
}
