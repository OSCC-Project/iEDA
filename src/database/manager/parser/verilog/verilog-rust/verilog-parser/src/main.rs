mod verilog_parser;
use std::time::Instant;

fn main() {
    let start_time = Instant::now();

    let verilog_file_str =
        "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/example1.v";
        // "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/asic_top_DC_downsize.v";
        // "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/asic_top_flatten.v";
    // let verilog_modules = verilog_parser::parse_verilog_file(verilog_file_str);
    
    // for verilog_module in &verilog_modules {
    //     println!("{:#?}", verilog_module);
    // }
    // println!("Number of verilog modules: {}", &verilog_modules.unwrap().len());
    let end_time = Instant::now();
    let elapsed_time = end_time.duration_since(start_time);
    let elapsed_s = elapsed_time.as_secs();

    println!("Program execution time (milliseconds): {} s", elapsed_s);
}
