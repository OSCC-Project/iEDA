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

#[repr(C)]
pub struct RustVerilogDcls {
    line_no: usize,
    verilog_dcls: RustVec,
} 

#[no_mangle]
pub extern "C" fn rust_convert_verilog_dcls(c_verilog_dcls_struct: *mut c_void)
-> *mut RustVerilogDcls {
    unsafe {
        let mut verilog_stmt = unsafe { &mut *(c_verilog_dcls_struct as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };
        let verilog_dcls_struct = (*verilog_stmt).as_any().downcast_ref::<verilog_data::VerilogDcls>().unwrap();
        let line_no = (*verilog_dcls_struct).get_stmt().get_line_no();
        let verilog_dcls_rust_vec = (*verilog_dcls_struct).get_verilog_dcls();
        let verilog_dcls= rust_vec_to_c_array(verilog_dcls_rust_vec);
        let rust_verilog_dcls = RustVerilogDcls { line_no, verilog_dcls };
        let rust_verilog_dcls_pointer = Box::new(rust_verilog_dcls);
        let raw_pointer = Box::into_raw(rust_verilog_dcls_pointer);
        raw_pointer
    }
}

#[repr(C)]
pub struct RustVerilogInst {
    line_no: usize,
    inst_name: *mut c_char,
    cell_name: *mut c_char,
    port_connections: RustVec,  
} 

#[no_mangle]
pub extern "C" fn rust_convert_verilog_inst(c_verilog_inst: *mut c_void)
-> *mut RustVerilogInst {
    unsafe {
        let mut verilog_stmt = unsafe { &mut *(c_verilog_inst as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };
        let verilog_inst = (*verilog_stmt).as_any().downcast_ref::<verilog_data::VerilogInst>().unwrap();

        // get value in verilog_inst.
        let line_no = (*verilog_inst).get_stmt().get_line_no();
        let inst_name_str = (*verilog_inst).get_inst_name();
        let cell_name_str = (*verilog_inst).get_cell_name();
        let port_connections_rust_vec = (*verilog_inst).get_port_connections();

        // convert str,vec.
        let inst_name = string_to_c_char(inst_name_str);
        let cell_name = string_to_c_char(cell_name_str);
        let port_connections = rust_vec_to_c_array(port_connections_rust_vec);

        let rust_verilog_inst = RustVerilogInst { line_no, inst_name,cell_name, port_connections};
        let rust_verilog_inst_pointer = Box::new(rust_verilog_inst);
        let raw_pointer = Box::into_raw(rust_verilog_inst_pointer);
        raw_pointer
    }
}

#[repr(C)]
pub struct RustVerilogPortRefPortConnect {
    port_id: *const c_void,
    net_expr: *mut c_void, 
} 

#[no_mangle]
pub extern "C" fn rust_convert_verilog_port_ref_port_connect(c_port_connect: *mut verilog_data::VerilogPortRefPortConnect) -> *mut RustVerilogPortRefPortConnect {
    unsafe {
        let port_id = (*c_port_connect).get_port_id();
        let net_expr = (*c_port_connect).get_net_expr();

        let c_port_id = &*port_id as *const _ as *const c_void;
        let c_net_expr = 
            if net_expr.is_some() { net_expr.as_deref().unwrap() as *const _ as *mut c_void } else { std::ptr::null_mut() };

        let port_connect =
        RustVerilogPortRefPortConnect { port_id:c_port_id, net_expr: c_net_expr as *mut c_void };
        let port_connect_pointer = Box::new(port_connect);
        Box::into_raw(port_connect_pointer)

    }
}


#[no_mangle]
pub extern "C" fn rust_is_module_inst_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let mut verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    unsafe { (*verilog_stmt).is_module_inst_stmt() }
}

#[no_mangle]
pub extern "C" fn rust_is_module_assign_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let mut verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    unsafe { (*verilog_stmt).is_module_assign_stmt() }
}

#[no_mangle]
pub extern "C" fn rust_is_verilog_dcl_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let mut verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    unsafe { (*verilog_stmt).is_verilog_dcl_stmt() }
}

#[no_mangle]
pub extern "C" fn rust_is_verilog_dcls_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let mut verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    unsafe { (*verilog_stmt).is_verilog_dcls_stmt() }
}

#[no_mangle]
pub extern "C" fn rust_is_module_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let mut verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    unsafe { (*verilog_stmt).is_module_stmt() }
}



