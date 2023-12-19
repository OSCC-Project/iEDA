mod matrix;
fn main() {
    println!("start iIR!");

    let spef_file_path = "/home/taosimin/T28/spef/asic_top.spef_vdd_vss_1212.rcworst.0c.spef";
    matrix::ir_rc::read_rc_data_from_spef(spef_file_path);
}
