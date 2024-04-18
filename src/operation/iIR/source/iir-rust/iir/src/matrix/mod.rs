pub mod ir_inst_power;
pub mod ir_rc;

use log;

use sprs::{TriMat, TriMatI};
use std::ffi::c_double;
use std::ffi::c_void;
use std::ffi::CString;
use std::os::raw::c_char;

use self::ir_rc::RCData;

/// RC matrix used for C interface.
#[repr(C)]
pub struct RustMatrix {
    // val at (row,col)
    data: f64,
    row: usize,
    col: usize,
}

/// RC vector used for C interface.
#[repr(C)]
pub struct RustVector {
    // val at position
    data: f64,
    index: usize,
}

/// Rust vec to C vec
#[repr(C)]
pub struct RustVec {
    data: *mut c_void,
    len: usize,
    cap: usize,
    type_size: usize,
}

/// One Net conductance matrix data.
struct IRNetConductanceData {
    net_name: String,
    conductance_matrix: Vec<RustMatrix>,
}

/// One net conductance matrix data for C.
#[repr(C)]
pub struct RustNetConductanceData {
    net_name: *const c_char,
    node_num: usize,
    g_matrix_vec: RustVec,
    ir_net_raw_ptr: *const c_void,
}

/// One net equation data.
#[repr(C)]
pub struct RustNetEquationData {
    net_name: *const c_char,
    g_matrix_vec: RustVec,
    j_vec: RustVec,
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

/// Rust convert rc matrix to C matrix.
fn rust_convert_rc_matrix(rc_matrix: &TriMatI<f64, usize>) -> Vec<RustMatrix> {
    let mut rust_matrix_vec = vec![];

    for (val, (row, col)) in rc_matrix.triplet_iter() {
        rust_matrix_vec.push(RustMatrix { row, col, data: *val });
    }

    rust_matrix_vec
}

/// c str to rust string.
pub fn c_str_to_r_str(str: *const c_char) -> String {
    let c_str = unsafe { std::ffi::CStr::from_ptr(str) };
    let r_str = c_str.to_string_lossy().into_owned();
    r_str
}

#[no_mangle]
pub extern "C" fn read_spef(c_power_net_spef: *const c_char) -> *const c_void {
    let power_net_spef = c_str_to_r_str(c_power_net_spef);
    let rc_data = ir_rc::read_rc_data_from_spef(&power_net_spef);
    let mv_rc_data = Box::new(rc_data);
    Box::into_raw(mv_rc_data) as *const c_void
}

#[no_mangle]
pub extern "C" fn build_one_net_conductance_matrix_data(
    c_rc_data: *const c_void,
    c_net_name: *const c_char,
) -> RustNetConductanceData {
    let rc_data = unsafe { &*(c_rc_data as *const RCData) };

    let one_net_name = c_str_to_r_str(c_net_name);
    let one_net_rc_data = rc_data.get_one_net_data(&one_net_name);

    let conductance_matrix_triplet = ir_rc::build_conductance_matrix(one_net_rc_data);
    let rust_matrix = rust_convert_rc_matrix(&conductance_matrix_triplet);
    let rust_matrix_vec = rust_vec_to_c_array(&rust_matrix);

    let one_net_conductance_data =
        Box::from(IRNetConductanceData { net_name: one_net_name, conductance_matrix: rust_matrix });

    // Need free the memory after use.
    let ir_net_raw_ptr = Box::into_raw(one_net_conductance_data);

    // The C image of rust data.
    let rust_one_net_conductance_data = RustNetConductanceData {
        net_name: c_net_name,
        node_num: conductance_matrix_triplet.shape().0,
        g_matrix_vec: rust_matrix_vec,
        ir_net_raw_ptr: ir_net_raw_ptr as *const c_void,
    };
    rust_one_net_conductance_data
}

/// Build RC matrix and current vector data.
#[no_mangle]
pub extern "C" fn build_matrix_from_raw_data(
    c_inst_power_path: *const c_char,
    c_power_net_spef: *const c_char,
) -> RustVec {
    // Firstly, read spef.
    let power_net_spef = c_str_to_r_str(c_power_net_spef);
    let inst_power_path = c_str_to_r_str(c_inst_power_path);

    let rc_data = ir_rc::read_rc_data_from_spef(&power_net_spef);
    let instance_power_data = ir_inst_power::read_instance_pwr_csv(&inst_power_path).expect("error reading instance power csv file");

    let mut net_matrix_data: Vec<RustNetEquationData> = Vec::new();
    // Secondly, construct matrix data.
    for (net_name, one_net_data) in rc_data.get_power_nets_data() {
        log::info!("construct power net {} matrix start", net_name);

        // Build rc matrix
        let conductance_matrix_triplet = ir_rc::build_conductance_matrix(one_net_data);
        let rust_matrix = rust_convert_rc_matrix(&conductance_matrix_triplet);

        // Read instance power data.
        let current_vector_result = ir_inst_power::build_instance_current_vector(&instance_power_data, one_net_data);
        if let Ok(current_vector) = current_vector_result {
            // Construct net matrix(rc matrix and current vector) data.
            let mut current_vec: Vec<RustVector> = Vec::new();
            for (index, val) in current_vector {
                current_vec.push(RustVector { data: val, index });
            }
            let rust_matrix_vec = rust_vec_to_c_array(&rust_matrix);
            let rust_current_vec = rust_vec_to_c_array(&current_vec);
            net_matrix_data.push(RustNetEquationData {
                net_name: string_to_c_char(net_name),
                g_matrix_vec: rust_matrix_vec,
                j_vec: rust_current_vec,
            });
        } else {
            panic!("current vector is none");
        }
        log::info!("construct power net {} matrix finish", net_name);
    }

    // Finaly, return data to C.
    let net_matrix_data_vec = rust_vec_to_c_array(&net_matrix_data);
    net_matrix_data_vec
}

#[cfg(test)]
mod tests {
    use super::ir_rc;
    use crate::matrix::{
        rust_convert_rc_matrix, rust_vec_to_c_array, string_to_c_char, RustNetConductanceData, RustNetEquationData,
        RustVector,
    };

    #[test]
    fn test_build_matrix() {
        let spef_file_path = "/home/taosimin/T28/spef/asic_top.spef_vdd_vss_1212.rcworst.0c.spef";

        let rc_data = ir_rc::read_rc_data_from_spef(spef_file_path);

        for (net_name, one_net_data) in rc_data.get_power_nets_data() {
            log::info!("construct power net {} matrix start", net_name);

            // Build rc matrix
            let conductance_matrix_triplet = ir_rc::build_conductance_matrix(one_net_data);
            let rust_matrix = rust_convert_rc_matrix(&conductance_matrix_triplet);

            let mut current_vec: Vec<RustVector> = Vec::new();

            let rust_matrix_vec = rust_vec_to_c_array(&rust_matrix);
            let rust_current_vec = rust_vec_to_c_array(&current_vec);

            let mut net_matrix_data: Vec<RustNetConductanceData> = Vec::new();
            let mut one_net_rc_net = RustNetEquationData {
                net_name: string_to_c_char(net_name),
                g_matrix_vec: rust_matrix_vec,
                j_vec: rust_current_vec,
            };
            net_matrix_data.push(one_net_rc_net);
        }
    }
}
