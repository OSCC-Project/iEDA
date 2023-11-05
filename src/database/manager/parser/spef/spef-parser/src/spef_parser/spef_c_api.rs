use std::ffi::c_void;
use std::os::raw::c_char;

use crate::spef_parser::parse_spef_file;

#[repr(C)]
pub struct RustVec {
    data: *mut c_void,
    len: usize,
    cap: usize,
    type_size: usize,
}

fn rust_vec_to_c_array<T>(vec: &Vec<T>) -> RustVec {
    RustVec {
        data: vec.as_ptr() as *mut c_void,
        len: vec.len(),
        cap: vec.capacity(),
        type_size: std::mem::size_of::<T>(),
    }
}

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
    pin_name: *mut c_char,
    driving_cell: *mut c_char,
    load: f64,
    layer: usize,
    coordinates: [f64; 2],
    ll_coordinate: [f64; 2],
    ur_coordinate: [f64; 2],
}

#[repr(C)]
struct CapItem {
    pin_port: [*mut c_char; 2],
    cap_val: f64,
}

#[repr(C)]
struct ResItem {
    pin_port: [*mut c_char; 2],
    res: f64,
}

#[repr(C)]
struct NetItem {
    name: *mut c_char,
    lcap: f64,
    conns: RustVec,
    caps: RustVec,
    ress: RustVec,
}
// Shared structure between cpp and rust, it uses the types in the `exter "Rust" section`
#[repr(C)]
struct SpefFile {
    name: *mut c_char,
    header: RustVec,
    name_map: RustVec,
    ports: RustVec,
    nets: RustVec,
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
