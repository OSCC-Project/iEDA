use std::ffi::c_double;
use std::ffi::c_void;
use std::os::raw::c_char;

use std::ffi::CString;
use std::ops::Deref;
use std::ops::DerefMut;

use crate::spef_parser::parse_spef_file;
use crate::spef_parser::spef_data;

#[repr(C)]
struct RustPair<T> {
    first: T,
    second: T,
}

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
    direction: spef_data::ConnectionDirection,
    coordinate: [f64; 2],
}

#[repr(C)]
struct RustConnItem {
    conn_type: spef_data::ConnectionType,
    conn_direction: spef_data::ConnectionDirection,
    name: *mut c_char,
    driving_cell: *mut c_char,
    load: c_double,
    layer: u32,
    coordinate: RustPair<c_double>,
    ll_coordinate: RustPair<c_double>,
    ur_coordinate: RustPair<c_double>,
}

#[repr(C)]
struct RustResCapItem {
    node1: *mut c_char,
    node2: *mut c_char,
    res_cap: f64,
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

#[no_mangle]
pub extern "C" fn rust_convert_spef_conn(c_spef_net: *mut spef_data::SpefConnEntry) -> *mut c_void {
    unsafe {
        let conn_type = (*c_spef_net).conn_type;
        let conn_direction = (*c_spef_net).conn_direction;
        let name = string_to_c_char(&(*c_spef_net).name);
        let driving_cell = string_to_c_char(&(*c_spef_net).driving_cell);
        let load = (*c_spef_net).load;
        let layer = (*c_spef_net).layer;

        let coordinate = RustPair { first: (*c_spef_net).coordinate.0, second: (*c_spef_net).coordinate.1 };

        let ll_coordinate = RustPair { first: (*c_spef_net).ll_coordinate.0, second: (*c_spef_net).ll_coordinate.1 };

        let ur_coordinate = RustPair { first: (*c_spef_net).ur_coordinate.0, second: (*c_spef_net).ur_coordinate.1 };

        let rust_spef_conn = RustConnItem {
            conn_type,
            conn_direction,
            name,
            driving_cell,
            load,
            layer,
            coordinate,
            ll_coordinate,
            ur_coordinate,
        };

        let spef_conn_pointer = Box::new(rust_spef_conn);
        let raw_pointer = Box::into_raw(spef_conn_pointer);
        raw_pointer as *mut c_void
    }
}

#[no_mangle]
pub extern "C" fn rust_convert_spef_net_cap_res(c_spef_net_cap_res: *mut spef_data::SpefResCap) -> *mut c_void {
    unsafe {
        let node1 = string_to_c_char(&(*c_spef_net_cap_res).node1);
        let node2 = string_to_c_char(&(*c_spef_net_cap_res).node2);
        let res_cap = (*c_spef_net_cap_res).res_or_cap;

        let rust_spef_net_cap_res = RustResCapItem { node1, node2, res_cap };

        let spef_net_cap_res_pointer = Box::new(rust_spef_net_cap_res);
        let raw_pointer = Box::into_raw(spef_net_cap_res_pointer);
        raw_pointer as *mut c_void
    }
}
