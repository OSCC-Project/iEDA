mod app;
mod matrix;

fn main() {
    app::init_ir();

    log::info!("start iIR");
    let spef_file_path = "/home/taosimin/T28/spef/asic_top.spef_vdd_vss_1212.rcworst.0c.spef";
    let _instance_power_path = "/home/shaozheqing/iEDA/bin/report_instance.csv";

    matrix::ir_rc::read_rc_data_from_spef(spef_file_path);
}
