fn main() {
    let source_files = vec!["src/spef_parser/mod.rs", "src/spef_parser/spef_data.rs"];
    cxx_build::bridges(source_files)
        .file("src/spef_parser.cc")
        .flag_if_supported("-std=c++14")
        .compile("spef_parser");

    // println!("cargo:rerun-if-changed=src/main.rs");
    // println!("cargo:rerun-if-changed=src/blobstore.cc");
    // println!("cargo:rerun-if-changed=include/blobstore.h");
}
