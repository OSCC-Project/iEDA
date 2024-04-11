pub mod ir_inst_power;
pub mod ir_rc;

use log;

use sprs::{TriMat, TriMatI};
use std::ffi::c_double;
use std::ffi::c_void;
use std::os::raw::c_char;

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

/// One net rc matrix data.
#[repr(C)]
pub struct RustNetRCData {
    net_name: String,
    rc_matrix: RustVec,
    current_vec: RustVec,
}

fn rust_vec_to_c_array<T>(vec: &Vec<T>) -> RustVec {
    RustVec {
        data: vec.as_ptr() as *mut c_void,
        len: vec.len(),
        cap: vec.capacity(),
        type_size: std::mem::size_of::<T>(),
    }
}

/// Rust convert rc matrix to C matrix.
fn rust_convert_rc_matrix(rc_matrix: &TriMatI<f64, usize>) -> Vec<RustMatrix> {
    let mut rust_matrix_vec = vec![];

    for (val, (row, col)) in rc_matrix.triplet_iter() {
        rust_matrix_vec.push(RustMatrix {
            row,
            col,
            data: *val,
        });
    }

    rust_matrix_vec
}

/// c str to rust string.
pub fn c_str_to_r_str(str: *const c_char) -> String {
    let c_str = unsafe { std::ffi::CStr::from_ptr(str) };
    let r_str = c_str.to_string_lossy().into_owned();
    r_str
}

/// Build RC matrix and current vector data.
#[no_mangle]
pub extern "C" fn build_matrix_from_raw_data(
    c_inst_power_path: *const c_char,
    c_power_net_spef: *const c_char,
) -> RustVec {
    // Firstly, read spef.
    let power_net_spef= c_str_to_r_str(c_power_net_spef);
    let inst_power_path = c_str_to_r_str(c_inst_power_path);

    let rc_data = ir_rc::read_rc_data_from_spef(&power_net_spef);

    let mut net_matrix_data: Vec<RustNetRCData> = Vec::new();
    // Secondly, construct matrix data.
    for (net_name, one_net_data) in rc_data.get_power_nets_data() {
        log::info!("construct power net {} matrix start", net_name);

        // Build rc matrix
        let conductance_matrix_triplet = ir_rc::build_conductance_matrix(one_net_data);
        let rust_matrix = rust_convert_rc_matrix(&conductance_matrix_triplet);

        // Read instance power data.
        let current_vector_result =
            ir_inst_power::build_instance_current_vector(&inst_power_path, one_net_data);
        if let Ok(current_vector) = current_vector_result {
            // Construct net matrix(rc matrix and current vector) data.
            let mut current_vec: Vec<RustVector> = Vec::new();
            for (index, val) in current_vector {
                current_vec.push(RustVector { data: val, index });
            }
            let rust_matrix_vec = rust_vec_to_c_array(&rust_matrix);
            let rust_current_vec = rust_vec_to_c_array(&current_vec);
            net_matrix_data.push(RustNetRCData {
                net_name: net_name.clone(),
                rc_matrix: rust_matrix_vec,
                current_vec: rust_current_vec,
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
