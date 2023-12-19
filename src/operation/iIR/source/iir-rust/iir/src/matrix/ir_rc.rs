use spef_parser::spef_parser;

/// read rc data from spef file.
pub fn read_rc_data_from_spef(spef_file_path: &str) {
    let spef_data = spef_parser::parse_spef_file(spef_file_path);
    let spef_data_nets = spef_data.get_nets();

    for spef_net in spef_data_nets {
        // println!("{:?}", spef_net);
    }
}
