mod verilog_parser;
use std::time::Instant;

fn main() {
    let start_time = Instant::now();

    let verilog_file_str =
        // "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/example1.v";
    "/home/taosimin/T28/ieda_1208/asic_top_1208.syn.v";

    let top_module_name = "asic_top";
    let mut verilog_file = verilog_parser::parse_verilog_file(verilog_file_str, top_module_name);
    // let verilog_modules = verilog_file.get_verilog_modules();
    let top_verilog_module_option = verilog_file.get_module(top_module_name);

    let top_module = top_verilog_module_option.unwrap().borrow();
    // println!("{:#?}", top_module);
    let inst_stmt = top_module.find_inst_stmt(
        "u0_soc_top/u0_ysyx_210720/coretop/ysyx_210720_ICache/dataArrayWay0",
        "TS5N28HPCPLVTA64X128M2FW",
    );
    // if let Some(inst_stmt_value) = inst_stmt {
    //     println!("{:#?}", inst_stmt_value);
    // }

    // println!("Number of verilog modules: {}", verilog_modules.len());
    let end_time = Instant::now();
    let elapsed_time = end_time.duration_since(start_time);
    let elapsed_s = elapsed_time.as_secs();

    println!("Program execution time (seconds): {} s", elapsed_s);
}
