use std::ffi::c_double;
use std::ffi::c_void;
use std::os::raw::c_char;

use std::ffi::CString;
use std::ops::Deref;
use std::ops::DerefMut;

use crate::spef_parser::parse_spef_file;
use crate::spef_parser::spef_data;

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

pub fn string_to_c_char(s: &str) -> *mut c_char {
    let cs = CString::new(s).unwrap();
    cs.into_raw()
}

#[no_mangle]
pub extern "C" fn free_c_char(s: *mut c_char) {
    unsafe {
        let _ = CString::from_raw(s);
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
struct RustHeaderItem {
    key: *mut c_char,
    value: String,
}

#[repr(C)]
struct RustNameMapItem {
    index: usize,
    name: *mut c_char,
}

#[repr(C)]
struct RustPortItem {
    name: *mut c_char,
    direction: ConnectionDirection,
    coordinates: [f64; 2],
}

#[repr(C)]
struct RustConnItem {
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
struct RustCapItem {
    pin_port: [*mut c_char; 2],
    cap_val: f64,
}

#[repr(C)]
struct RustResItem {
    pin_port: [*mut c_char; 2],
    res: f64,
}

#[repr(C)]
struct RustNetItem {
    name: *mut c_char,
    lcap: c_double,
    conns: RustVec,
    caps: RustVec,
    ress: RustVec,
}
// Shared structure between cpp and rust, it uses the types in the `exter "Rust" section`
#[repr(C)]
struct RustSpefFile {
    file_name: *mut c_char,
    header: RustVec,
    name_map: RustVec,
    ports: RustVec,
    nets: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_parser_spef(spef_path: *const c_char) -> *mut c_void {
    unsafe {
        let c_str = unsafe { std::ffi::CStr::from_ptr(spef_path) };
        let r_str = c_str.to_string_lossy().into_owned();
        println!("r str {}", r_str);

        let spef_data = parse_spef_file(&r_str);

        let spef_data_pointer = Box::new(spef_data);
        let raw_pointer = Box::into_raw(spef_data_pointer);
        raw_pointer as *mut c_void
    }
}

#[no_mangle]
pub extern "C" fn rust_covert_spef_file(c_spef_data: *mut spef_data::SpefExchange) -> *mut c_void {
    unsafe {
        let file_name = string_to_c_char(&(*c_spef_data).file_name);
        let header = rust_vec_to_c_array(&(*c_spef_data).header);
        let name_map = rust_vec_to_c_array(&(*c_spef_data).namemap);
        let ports = rust_vec_to_c_array(&(*c_spef_data).ports);
        let nets = rust_vec_to_c_array(&(*c_spef_data).nets);

        let rust_spef_data = RustSpefFile { file_name, header, name_map, ports, nets };

        let spef_data_pointer = Box::new(rust_spef_data);
        let raw_pointer = Box::into_raw(spef_data_pointer);
        raw_pointer as *mut c_void
    }
}

#[no_mangle]
pub extern "C" fn rust_convert_spef_net(c_spef_net: *mut spef_data::SpefNet) -> *mut c_void {
    unsafe {
        let name = string_to_c_char(&(*c_spef_net).name);
        let lcap: c_double = (*c_spef_net).lcap;
        let conns = rust_vec_to_c_array(&(*c_spef_net).connection);
        let caps = rust_vec_to_c_array(&(*c_spef_net).caps);
        let ress = rust_vec_to_c_array(&(*c_spef_net).ress);

        let rust_spef_net = RustNetItem { name, lcap, conns, caps, ress };

        let spef_net_pointer = Box::new(rust_spef_net);
        let raw_pointer = Box::into_raw(spef_net_pointer);
        raw_pointer as *mut c_void
    }
}
