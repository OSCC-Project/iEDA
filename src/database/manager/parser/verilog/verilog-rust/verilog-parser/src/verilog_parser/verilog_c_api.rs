use crate::verilog_parser::verilog_data;

use std::ffi::CString;
use std::ops::Deref;
use std::ops::DerefMut;
use std::os::raw::*;

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

// More functions to manipulate the Vec...

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
pub struct RustVerilogModule {
    line_no: usize,
    module_name: *mut c_char,
    port_list: RustVec,
    module_stmts: RustVec,
} 

#[no_mangle]
pub extern "C" fn rust_convert_raw_verilog_module(verilog_module: *mut verilog_data::VerilogModule)
-> *mut RustVerilogModule {
    unsafe {
        // get the value in verilog_data::VerilogModule.
        let line_no = (*verilog_module).get_stmt().get_line_no();
        let module_name_str = (*verilog_module).get_module_name();
        let port_list_rust_vec = (*verilog_module).get_port_list();
        let module_stmts_rust_vec = (*verilog_module).get_module_stmts();

        // convert str, vec.
        let module_name = string_to_c_char(module_name_str);
        let port_list = rust_vec_to_c_array(port_list_rust_vec);
        let module_stmts = rust_vec_to_c_array(module_stmts_rust_vec);

        let verilog_module = RustVerilogModule {line_no,module_name,port_list,module_stmts};
        let verilog_module_pointer = Box::new(verilog_module);
        let raw_pointer = Box::into_raw(verilog_module_pointer);
        raw_pointer
    }
}



