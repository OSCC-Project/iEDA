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
    ports: RustVec,
    nets: RustVec,
}

pub fn split_spef_index_str(index_name: &str) -> (&str, &str) {
    let v: Vec<&str> = index_name.split(":").collect();
    let index_str = v.first().unwrap();
    let node_str = v.last().unwrap();
    if v.len() == 2 {
        (&index_str[1..], *node_str)
    } else {
        (&index_str[1..], "")
    }
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
        let ports = rust_vec_to_c_array(&(*c_spef_data).ports);
        let nets = rust_vec_to_c_array(&(*c_spef_data).nets);

        let rust_spef_data = RustSpefFile { file_name, header, ports, nets };

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

#[no_mangle]
pub extern "C" fn rust_expand_name(c_spef_data: *mut spef_data::SpefExchange, index: usize) -> *mut c_char {
    unsafe {
        let name = (*c_spef_data).name_map.get(&index).unwrap();
        string_to_c_char(name)
    }
}

#[no_mangle]
pub extern "C" fn rust_expand_all_name(c_spef_data: *mut spef_data::SpefExchange) {
    unsafe {
        let expand_name = |name: &str, spef_data: &mut spef_data::SpefExchange| -> String {
            let split_names = split_spef_index_str(&name);
            let index = split_names.0.parse::<usize>().unwrap();
            let node1_map_name = spef_data.name_map.get(&index).unwrap();
            let remove_slash_name: String = node1_map_name.chars().filter(|&c| c != '\\').collect();
            if !split_names.1.is_empty() {
                let expand_node1_name = remove_slash_name + ":" + split_names.1;
                return expand_node1_name;
            }
            remove_slash_name
        };

        for spef_net in &mut (*c_spef_data).nets {
            let net_name = &spef_net.name;
            let index = net_name[1..].parse::<usize>().unwrap();
            let expand_net_name = (*c_spef_data).name_map.get(&index).unwrap();
            let remove_slash_net_name = expand_net_name.chars().filter(|&c| c != '\\').collect();
            spef_net.name = remove_slash_net_name;

            for spef_conn in &mut spef_net.connection {
                let conn_name = &spef_conn.name;
                let expand_conn_name = expand_name(conn_name, &mut (*c_spef_data));
                spef_conn.set_name(expand_conn_name);
            }

            for spef_cap in &mut spef_net.caps {
                let node1_name = &spef_cap.node1;
                let expand_node1_name = expand_name(node1_name, &mut (*c_spef_data));
                spef_cap.node1 = expand_node1_name;

                let node2_name = &spef_cap.node2;
                if !node2_name.is_empty() {
                    let expand_node2_name = expand_name(node2_name, &mut (*c_spef_data));
                    spef_cap.node2 = expand_node2_name;
                }
            }

            for spef_res in &mut spef_net.ress {
                let node1_name = &spef_res.node1;
                let expand_node1_name = expand_name(node1_name, &mut (*c_spef_data));
                spef_res.node1 = expand_node1_name;

                let node2_name = &spef_res.node2;
                if !node2_name.is_empty() {
                    let expand_node2_name = expand_name(node2_name, &mut (*c_spef_data));
                    spef_res.node2 = expand_node2_name;
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn rust_get_spef_cap_unit(c_spef_data: *mut spef_data::SpefExchange) -> *mut c_char {
    unsafe {
        let mut unit_str: &str = "";
        for entry in &(*c_spef_data).header {
            if entry.header_key == "*C_UNIT" {
                unit_str = &entry.header_value;
                break;
            }
        }
        string_to_c_char(unit_str)
    }
}

#[no_mangle]
pub extern "C" fn rust_get_spef_res_unit(c_spef_data: *mut spef_data::SpefExchange) -> *mut c_char {
    unsafe {
        let mut unit_str: &str = "";
        for entry in &(*c_spef_data).header {
            if entry.header_key == "*R_UNIT" {
                unit_str = &entry.header_value;
                break;
            }
        }
        string_to_c_char(unit_str)
    }
}
