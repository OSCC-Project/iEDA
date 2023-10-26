mod verilog_parser;

fn main() {
    let verilog_file_str =
        // "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/example1.v";
        // "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/asic_top_DC_downsize.v";
        "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/asic_top_flatten.v";
    let verilog_module = verilog_parser::parse_verilog_file(verilog_file_str);
    println!("{:#?}", verilog_module);
}
