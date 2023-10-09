use crate::liberty_parser::liberty_data;

use std::ffi::CString;
use std::os::raw::c_char;
use std::os::raw::c_void;

#[repr(C)]
pub struct RustVec {
    data: *mut c_void,
    len: usize,
    cap: usize,
}

#[no_mangle]
pub extern "C" fn rust_vec_new() -> RustVec {
    let vec: Vec<u8> = Vec::new();
    RustVec { data: vec.as_ptr() as *mut c_void, len: vec.len(), cap: vec.capacity() }
}

#[no_mangle]
pub extern "C" fn rust_vec_len(vec: &RustVec) -> usize {
    vec.len
}

// More functions to manipulate the Vec...

#[no_mangle]
pub extern "C" fn string_to_c_char(s: String) -> *mut c_char {
    let cs = CString::new(s).unwrap();
    cs.into_raw()
}

#[no_mangle]
pub extern "C" fn free_c_char(s: *mut c_char) {
    unsafe {
        CString::from_raw(s);
    }
}

#[repr(C)]
pub struct RustLibertyGroupStmt {
    file_name: *mut c_char,
    line_no: u32,
    group_name: *mut c_char,
    attri_values: RustVec,
    stmts: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_convert_group_stmt(group_stmt: *mut liberty_data::LibertyGroupStmt) {}

#[repr(C)]
pub struct RustLibertySimpleAttrStmt {
    file_name: *mut c_char,
    line_no: u32,
    attri_name: *mut c_char,
    attri_value: *mut c_void,
}

#[no_mangle]
pub extern "C" fn rust_convert_simple_attribute_stmt(group_stmt: *mut liberty_data::LibertySimpleAttrStmt) {}

#[repr(C)]
pub struct RustLibertyComplexAttrStmt {
    file_name: *mut c_char,
    line_no: u32,
    attri_name: *mut c_char,
    attri_values: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_convert_complex_attribute_stmt(group_stmt: *mut liberty_data::LibertyComplexAttrStmt) {}
