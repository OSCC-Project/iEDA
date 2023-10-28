use crate::vcd_parser::parse_vcd_file;
use crate::vcd_parser::vcd_data;

use std::borrow::BorrowMut;
use std::ffi::CString;
use std::ops::Deref;
use std::ops::DerefMut;
use std::os::raw::*;

use super::vcd_data::VCDScope;

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

#[no_mangle]
pub extern "C" fn rust_vec_len(vec: &RustVec) -> usize {
    vec.len
}

fn string_to_c_char(s: &str) -> *mut c_char {
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
pub struct RustVCDScope {
    name: *mut c_char,
    parent_scope: *mut c_void,
    children_scope: RustVec,
}

#[repr(C)]
pub struct RustVCDFile {
    start_time: c_longlong,
    end_time: c_longlong,
    time_resolution: c_uint,
    date: *mut c_char,
    version: *mut c_char,
    comment: *mut c_char,
    scope_root: *mut c_void,
}

#[no_mangle]
pub extern "C" fn rust_convert_vcd_scope(
    c_vcd_scope: *const vcd_data::VCDScope,
) -> *mut RustVCDScope {
    unsafe {
        let scope_name = (*c_vcd_scope).get_name();
        let name = string_to_c_char(scope_name);
        let parent_scope = (*c_vcd_scope)
            .get_parent_scope()
            .clone()
            .unwrap()
            .deref()
            .as_ptr();

        let the_scope_children_scope = (*c_vcd_scope).get_children_scopes();
        let children_scope = rust_vec_to_c_array(the_scope_children_scope);

        let vcd_scope = RustVCDScope {
            name,
            parent_scope: parent_scope as *mut c_void,
            children_scope,
        };
        let vcd_scope_pointer = Box::new(vcd_scope);
        let raw_pointer = Box::into_raw(vcd_scope_pointer);
        raw_pointer
    }
}
#[no_mangle]
pub extern "C" fn rust_convert_vcd_file(c_vcd_file: *mut vcd_data::VCDFile) -> *mut RustVCDFile {
    unsafe {
        let start_time = (*c_vcd_file).get_start_time();
        let end_time = (*c_vcd_file).get_end_time();
        let time_resolution = (*c_vcd_file).get_time_resolution();
        let vcd_date = (*c_vcd_file).get_date();
        let date = string_to_c_char(vcd_date);

        let vcd_version = (*c_vcd_file).get_version();
        let version = string_to_c_char(vcd_version);

        let vcd_comment = (*c_vcd_file).get_comment();
        let comment = string_to_c_char(vcd_comment);

        let vcd_root_scope = (*c_vcd_file).get_root_scope();
        let root_scope = vcd_root_scope.as_ref().unwrap().as_ref();

        let void_ptr = root_scope as *const VCDScope;

        let scope_root = rust_convert_vcd_scope(void_ptr);

        let vcd_file = RustVCDFile {
            start_time,
            end_time,
            time_resolution,
            date,
            version,
            comment,
            scope_root: scope_root as *mut c_void,
        };

        let vcd_file_pointer = Box::new(vcd_file);
        let raw_pointer = Box::into_raw(vcd_file_pointer);
        raw_pointer
    }
}

#[no_mangle]
pub extern "C" fn rust_parse_vcd(lib_path: *const c_char) -> *mut c_void {
    let c_str = unsafe { std::ffi::CStr::from_ptr(lib_path) };
    let r_str = c_str.to_string_lossy().into_owned();
    println!("r str {}", r_str);

    let vcd_file = parse_vcd_file(&r_str);

    let vcd_file_pointer = Box::new(vcd_file.unwrap());

    let raw_pointer = Box::into_raw(vcd_file_pointer);
    raw_pointer as *mut c_void
}
