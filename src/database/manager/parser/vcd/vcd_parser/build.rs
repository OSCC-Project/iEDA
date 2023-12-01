fn main() {
    let config = cbindgen::Config::from_file("cbindgen.toml")
        .expect("Failed to read cbindgen configuration file");

    cbindgen::Builder::new()
        .with_config(config)
        .with_crate("./")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("dist/vcd_parser.h");
}
