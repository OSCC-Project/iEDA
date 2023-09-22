mod liberty_parser;

fn main() {
    let lib_file_str =
        "/home/taosimin/iEDA/src/database/manager/parser/liberty/lib-rust/liberty-parser/example/example1_slow.lib";
    let lib_file = liberty_parser::parse_lib_file(lib_file_str);
}
