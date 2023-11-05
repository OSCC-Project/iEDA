use std::ffi::c_void;
use std::os::raw::c_char;

use crate::spef_parser::parse_spef_file;

#[repr(C)]
enum ConnectionType {
    INTERNAL,
    EXTERNAL,
    UNITIALIZED,
}

#[repr(C)]
enum ConnectionDirection {
    INPUT,
    OUTPUT,
    INOUT,
    UNITIALIZED,
}

#[repr(C)]
struct HeaderItem {
    key: String,
    value: String,
}

#[repr(C)]
struct NameMapItem {
    index: usize,
    name: String,
}

#[repr(C)]
struct PortItem {
    name: String,
    direction: ConnectionDirection,
    coordinates: [f64; 2],
}

#[repr(C)]
struct ConnItem {
    conn_type: ConnectionType,
    conn_direction: ConnectionDirection,
    pin_name: String,
    driving_cell: String,
    load: f64,
    layer: usize,
    coordinates: [f64; 2],
    ll_coordinate: [f64; 2],
    ur_coordinate: [f64; 2],
}

#[repr(C)]
struct CapItem {
    pin_port: [String; 2],
    cap_val: f64,
}

#[repr(C)]
struct ResItem {
    pin_port: [String; 2],
    res: f64,
}

#[repr(C)]
struct NetItem {
    name: String,
    lcap: f64,
    conns: Vec<ConnItem>,
    caps: Vec<CapItem>,
    ress: Vec<ResItem>,
}
// Shared structure between cpp and rust, it uses the types in the `exter "Rust" section`
#[repr(C)]
struct SpefFile {
    name: String,
    header: Vec<HeaderItem>,
    name_vector: Vec<NameMapItem>,
    ports: Vec<PortItem>,
    nets: Vec<NetItem>,
}

#[no_mangle]
pub extern "C" fn rust_parser_spef(spef_path: *const c_char) -> *mut c_void {
    let c_str = unsafe { std::ffi::CStr::from_ptr(spef_path) };
    let r_str = c_str.to_string_lossy().into_owned();
    println!("r str {}", r_str);

    let spef_data = parse_spef_file(&r_str);

    let spef_data_pointer = Box::new(spef_data);
    let raw_pointer = Box::into_raw(spef_data_pointer);
    raw_pointer as *mut c_void
}
