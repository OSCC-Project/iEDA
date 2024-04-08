mod matrix;

use std::io::Write;
use env_logger;
use log;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{timestamp}][{file_name}:{line_no}][{thread_id:?}][{level}] {args}",
                timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                file_name = record.file_static().unwrap(),
                line_no = record.line().unwrap(),
                thread_id = std::thread::current().id(),
                level = record.level(),
                args = record.args(),
            )
        })
        .init();

    log::info!("start iIR");
    let spef_file_path = "/home/taosimin/T28/spef/asic_top.spef_vdd_vss_1212.rcworst.0c.spef";
    let instance_power_path = "/home/shaozheqing/iEDA/bin/report_instance.csv";
    matrix::build_matrix_from_raw_data(instance_power_path, spef_file_path);
}
