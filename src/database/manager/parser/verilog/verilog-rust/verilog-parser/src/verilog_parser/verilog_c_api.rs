use crate::verilog_parser::verilog_data;

use std::cell::RefCell;
use std::ffi::CStr;
use std::ffi::CString;
use std::ops::Deref;
use std::os::raw::*;
use std::rc::Rc;

#[repr(C)]
pub struct RustVec {
    data: *mut c_void,
    len: usize,
    cap: usize,
    type_size: usize,
}

pub fn rust_vec_to_c_array<T>(vec: &Vec<T>) -> RustVec {
    RustVec {
        data: vec.as_ptr() as *mut c_void,
        len: vec.len(),
        cap: vec.capacity(),
        type_size: std::mem::size_of::<T>(),
    }
}

#[no_mangle]
pub extern "C" fn verilog_rust_vec_len(vec: &RustVec) -> usize {
    vec.len
}

// More functions to manipulate the Vec...

fn string_to_c_char(s: &str) -> *mut c_char {
    let cs = CString::new(s).unwrap();
    cs.into_raw()
}

#[no_mangle]
pub extern "C" fn verilog_free_c_char(s: *mut c_char) {
    unsafe {
        let _ = CString::from_raw(s);
    }
}

#[repr(C)]
pub struct CRange {
    has_value: bool,
    start: i32,
    end: i32,
}

#[repr(C)]
pub struct RustVerilogID {
    id: *mut c_char,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_id(c_verilog_virtual_base_id: *mut c_void) -> *mut RustVerilogID {
    let virtual_base_id =
        unsafe { &mut *(c_verilog_virtual_base_id as *mut Box<dyn verilog_data::VerilogVirtualBaseID>) };

    let rust_name = (*virtual_base_id).get_name();
    let name = string_to_c_char(rust_name);

    let rust_verilog_id = RustVerilogID { id: name };

    let rust_verilog_id_pointer = Box::new(rust_verilog_id);
    
    Box::into_raw(rust_verilog_id_pointer)
}

#[no_mangle]
pub extern "C" fn rust_is_id(c_verilog_virtual_base_id: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyAttrValue
    let virtual_base_id =
        unsafe { &mut *(c_verilog_virtual_base_id as *mut Box<dyn verilog_data::VerilogVirtualBaseID>) };

    (*virtual_base_id).is_id()
}

#[repr(C)]
pub struct RustVerilogIndexID {
    id: *mut c_char,
    base_id: *mut c_char,
    index: i32,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_index_id(c_verilog_virtual_base_id: *mut c_void) -> *mut RustVerilogIndexID {
    let virtual_base_id =
        unsafe { &mut *(c_verilog_virtual_base_id as *mut Box<dyn verilog_data::VerilogVirtualBaseID>) };

    let rust_name = (*virtual_base_id).get_name();
    let name = string_to_c_char(rust_name);

    let mut base_id = std::ptr::null_mut();
    let mut index = 0;

    if let Some(open_bracket_index) = rust_name.find('[') {
        if let Some(close_bracket_index) = rust_name.find(']') {
            if let Ok(idx) = rust_name[open_bracket_index + 1..close_bracket_index].parse::<i32>() {
                base_id = string_to_c_char(&rust_name[..open_bracket_index]);
                index = idx;
            }
        }
    }

    let rust_verilog_index_id = RustVerilogIndexID { id: name, base_id, index };

    let rust_verilog_index_id_pointer = Box::new(rust_verilog_index_id);
    
    Box::into_raw(rust_verilog_index_id_pointer)
}

#[no_mangle]
pub extern "C" fn rust_is_bus_index_id(c_verilog_virtual_base_id: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyAttrValue
    let virtual_base_id =
        unsafe { &mut *(c_verilog_virtual_base_id as *mut Box<dyn verilog_data::VerilogVirtualBaseID>) };

    (*virtual_base_id).is_bus_index_id()
}

#[repr(C)]
pub struct RustVerilogSliceID {
    id: *mut c_char,
    base_id: *mut c_char,
    range_base: i32,
    range_max: i32,
}

#[no_mangle]
pub extern "C" fn rust_get_index_name(verilog_slice_id: *mut RustVerilogSliceID, index: usize) -> *const c_char {
    let base_id = unsafe { CStr::from_ptr((*verilog_slice_id).base_id).to_str().unwrap() };
    let result = format!("{}[{}]", base_id, index);
    let result_cstring = CString::new(result).unwrap();
    result_cstring.into_raw()
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_slice_id(c_verilog_virtual_base_id: *mut c_void) -> *mut RustVerilogSliceID {
    let virtual_base_id =
        unsafe { &mut *(c_verilog_virtual_base_id as *mut Box<dyn verilog_data::VerilogVirtualBaseID>) };

    let rust_name = (*virtual_base_id).get_name();
    let name = string_to_c_char(rust_name);

    let mut base_id = std::ptr::null_mut();
    let mut range_max = 0;
    let mut range_base = 0;

    // Extracting the base_id, range_max, and range_base from the name.
    // need test range_base/range_max is obtained correctly.
    if let Some(open_bracket_index) = rust_name.find('[') {
        if let Some(colon_index) = rust_name.find(':') {
            if let Some(close_bracket_index) = rust_name.find(']') {
                base_id = string_to_c_char(&rust_name[..open_bracket_index]);
                let range_from: i32 = rust_name[open_bracket_index + 1..colon_index].parse().unwrap();
                let range_to: i32 = rust_name[colon_index + 1..close_bracket_index].parse().unwrap();
                (range_base, range_max) =
                    if range_from < range_to { (range_from, range_to) } else { (range_to, range_from) };
            }
        }
    }

    let rust_verilog_slice_id = RustVerilogSliceID { id: name, base_id, range_base, range_max };

    let rust_verilog_slice_id_pointer = Box::new(rust_verilog_slice_id);
    
    Box::into_raw(rust_verilog_slice_id_pointer)
}

#[no_mangle]
pub extern "C" fn rust_is_bus_slice_id(c_verilog_virtual_base_id: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyAttrValue
    let virtual_base_id =
        unsafe { &mut *(c_verilog_virtual_base_id as *mut Box<dyn verilog_data::VerilogVirtualBaseID>) };

    (*virtual_base_id).is_bus_slice_id()
}

#[repr(C)]
pub struct RustVerilogNetIDExpr {
    line_no: usize,
    verilog_id: *const c_void,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_net_id_expr(c_verilog_net_id_expr: *mut c_void) -> *mut RustVerilogNetIDExpr {
    let virtual_base_id =
        unsafe { &mut *(c_verilog_net_id_expr as *mut Box<dyn verilog_data::VerilogVirtualBaseNetExpr>) };
    let verilog_net_id_expr = (*virtual_base_id).as_any().downcast_ref::<verilog_data::VerilogNetIDExpr>().unwrap();

    let line_no = (*verilog_net_id_expr).get_net_expr().get_line_no();
    let verilog_id_box = (*verilog_net_id_expr).get_verilog_id();
    let verilog_id = verilog_id_box as *const _ as *const c_void;

    let rust_verilog_net_id_expr = RustVerilogNetIDExpr { line_no, verilog_id };

    let rust_verilog_net_id_expr_pointer = Box::new(rust_verilog_net_id_expr);
    
    Box::into_raw(rust_verilog_net_id_expr_pointer)
}

#[repr(C)]
pub struct RustVerilogNetConcatExpr {
    line_no: usize,
    verilog_id_concat: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_net_concat_expr(
    c_verilog_net_concat_expr: *mut c_void,
) -> *mut RustVerilogNetConcatExpr {
    let virtual_base_id =
        unsafe { &mut *(c_verilog_net_concat_expr as *mut Box<dyn verilog_data::VerilogVirtualBaseNetExpr>) };
    let verilog_net_concat_expr =
        (*virtual_base_id).as_any().downcast_ref::<verilog_data::VerilogNetConcatExpr>().unwrap();

    let line_no = (*verilog_net_concat_expr).get_net_expr().get_line_no();
    let verilog_id_concat_rust_vec = (*verilog_net_concat_expr).get_verilog_id_concat();
    let verilog_id_concat = rust_vec_to_c_array(verilog_id_concat_rust_vec);

    let rust_verilog_net_concat_expr = RustVerilogNetConcatExpr { line_no, verilog_id_concat };

    let rust_verilog_net_concat_expr_pointer = Box::new(rust_verilog_net_concat_expr);
    
    Box::into_raw(rust_verilog_net_concat_expr_pointer)
}

#[repr(C)]
pub struct RustVerilogConstantExpr {
    line_no: usize,
    verilog_id: *const c_void,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_constant_expr(
    c_verilog_constant_expr: *mut c_void,
) -> *mut RustVerilogConstantExpr {
    let virtual_base_id =
        unsafe { &mut *(c_verilog_constant_expr as *mut Box<dyn verilog_data::VerilogVirtualBaseNetExpr>) };
    let verilog_constant_expr =
        (*virtual_base_id).as_any().downcast_ref::<verilog_data::VerilogConstantExpr>().unwrap();

    let line_no = (*verilog_constant_expr).get_net_expr().get_line_no();
    let verilog_id_box = (*verilog_constant_expr).get_verilog_id();
    let verilog_id = verilog_id_box as *const _ as *const c_void;

    let rust_verilog_constant_expr = RustVerilogConstantExpr { line_no, verilog_id };

    let rust_verilog_constant_expr_pointer = Box::new(rust_verilog_constant_expr);
    
    Box::into_raw(rust_verilog_constant_expr_pointer)
}

#[no_mangle]
pub extern "C" fn rust_is_id_expr(c_verilog_virtual_base_net_expr: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let verilog_virtual_base_net_expr =
        unsafe { &mut *(c_verilog_virtual_base_net_expr as *mut Box<dyn verilog_data::VerilogVirtualBaseNetExpr>) };

    (*verilog_virtual_base_net_expr).is_id_expr()
}

#[no_mangle]
pub extern "C" fn rust_is_concat_expr(c_verilog_virtual_base_net_expr: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let verilog_virtual_base_net_expr =
        unsafe { &mut *(c_verilog_virtual_base_net_expr as *mut Box<dyn verilog_data::VerilogVirtualBaseNetExpr>) };

    (*verilog_virtual_base_net_expr).is_concat_expr()
}

#[no_mangle]
pub extern "C" fn rust_is_constant(c_verilog_virtual_base_net_expr: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let verilog_virtual_base_net_expr =
        unsafe { &mut *(c_verilog_virtual_base_net_expr as *mut Box<dyn verilog_data::VerilogVirtualBaseNetExpr>) };

    (*verilog_virtual_base_net_expr).is_constant()
}

#[repr(C)]
pub struct RustVerilogModule {
    line_no: usize,
    module_name: *mut c_char,
    port_list: RustVec,
    module_stmts: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_convert_raw_verilog_module(
    verilog_module: *mut verilog_data::VerilogModule,
) -> *mut RustVerilogModule {
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

        let rust_verilog_module = RustVerilogModule { line_no, module_name, port_list, module_stmts };
        let rust_verilog_module_pointer = Box::new(rust_verilog_module);
        
        Box::into_raw(rust_verilog_module_pointer)
    }
}

#[repr(C)]
pub struct RustVerilogDcl {
    line_no: usize,
    dcl_type: verilog_data::DclType,
    dcl_name: *mut c_char,
    range: CRange,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_dcl(c_verilog_dcl_struct: *mut c_void) -> *mut RustVerilogDcl {
    let verilog_dcl_struct = unsafe { &mut *(c_verilog_dcl_struct as *mut Box<verilog_data::VerilogDcl>) };
    let line_no = (*verilog_dcl_struct).get_stmt().get_line_no();
    let dcl_type = (*verilog_dcl_struct).get_dcl_type();
    let dcl_name_str = (*verilog_dcl_struct).get_dcl_name();
    let dcl_name = string_to_c_char(dcl_name_str);
    let rust_range = (*verilog_dcl_struct).get_range();
    let range = match rust_range {
        Some((start, end)) => CRange { has_value: true, start: *start, end: *end },
        None => CRange { has_value: false, start: 0, end: 0 },
    };

    let rust_verilog_dcl = RustVerilogDcl { line_no, dcl_type, dcl_name, range };
    let rust_verilog_dcl_pointer = Box::new(rust_verilog_dcl);
    
    Box::into_raw(rust_verilog_dcl_pointer)
}

#[repr(C)]
pub struct RustVerilogDcls {
    line_no: usize,
    verilog_dcls: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_dcls(c_verilog_dcls_struct: *mut c_void) -> *mut RustVerilogDcls {
    let verilog_stmt = unsafe { &mut *(c_verilog_dcls_struct as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };
    let verilog_dcls_struct = (*verilog_stmt).as_any().downcast_ref::<verilog_data::VerilogDcls>().unwrap();
    let line_no = (*verilog_dcls_struct).get_stmt().get_line_no();
    let verilog_dcls_rust_vec = (*verilog_dcls_struct).get_verilog_dcls();
    let verilog_dcls = rust_vec_to_c_array(verilog_dcls_rust_vec);
    let rust_verilog_dcls = RustVerilogDcls { line_no, verilog_dcls };
    let rust_verilog_dcls_pointer = Box::new(rust_verilog_dcls);
    
    Box::into_raw(rust_verilog_dcls_pointer)
}

#[repr(C)]
pub struct RustVerilogInst {
    line_no: usize,
    inst_name: *mut c_char,
    cell_name: *mut c_char,
    port_connections: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_inst(c_verilog_inst: *mut c_void) -> *mut RustVerilogInst {
    let verilog_stmt = unsafe { &mut *(c_verilog_inst as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };
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

    let rust_verilog_inst = RustVerilogInst { line_no, inst_name, cell_name, port_connections };
    let rust_verilog_inst_pointer = Box::new(rust_verilog_inst);
    
    Box::into_raw(rust_verilog_inst_pointer)
}

#[repr(C)]
pub struct RustVerilogAssign {
    line_no: usize,
    left_net_expr: *const c_void,
    right_net_expr: *const c_void,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_assign(c_verilog_assign: *mut c_void) -> *mut RustVerilogAssign {
    let verilog_stmt = unsafe { &mut *(c_verilog_assign as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };
    let verilog_assign = (*verilog_stmt).as_any().downcast_ref::<verilog_data::VerilogAssign>().unwrap();

    // get value in verilog_assign.
    let line_no = (*verilog_assign).get_stmt().get_line_no();
    let left_net_expr_box = (*verilog_assign).get_left_net_expr();
    let right_net_expr_box = (*verilog_assign).get_right_net_expr();

    let c_left_net_expr = left_net_expr_box as *const _ as *const c_void;
    let c_right_net_expr = right_net_expr_box as *const _ as *const c_void;

    let rust_verilog_assign =
        RustVerilogAssign { line_no, left_net_expr: c_left_net_expr, right_net_expr: c_right_net_expr };

    let rust_verilog_assign_pointer = Box::new(rust_verilog_assign);
    
    Box::into_raw(rust_verilog_assign_pointer)
}

#[repr(C)]
pub struct RustVerilogPortRefPortConnect {
    port_id: *const c_void,
    net_expr: *mut c_void,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_port_ref_port_connect(
    c_port_connect: *mut c_void,
) -> *mut RustVerilogPortRefPortConnect {
    let c_port_connect = unsafe { &mut *(c_port_connect as *mut Box<verilog_data::VerilogPortRefPortConnect>) };
    let port_id = (*c_port_connect).get_port_id();

    let net_expr = (*c_port_connect).get_net_expr();

    let c_port_id = port_id as *const _ as *const c_void;
    let c_net_expr =
        if let Some(net_expr_box) = net_expr { net_expr_box as *const _ as *mut c_void } else { std::ptr::null_mut() };

    let rust_port_connect = RustVerilogPortRefPortConnect { port_id: c_port_id, net_expr: c_net_expr as *mut c_void };
    let rust_port_connect_pointer = Box::new(rust_port_connect);
    Box::into_raw(rust_port_connect_pointer)
}

#[no_mangle]
pub extern "C" fn rust_is_module_inst_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    (*verilog_stmt).is_module_inst_stmt()
}

#[no_mangle]
pub extern "C" fn rust_is_module_assign_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    (*verilog_stmt).is_module_assign_stmt()
}

#[no_mangle]
pub extern "C" fn rust_is_verilog_dcl_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    (*verilog_stmt).is_verilog_dcl_stmt()
}

#[no_mangle]
pub extern "C" fn rust_is_verilog_dcls_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    (*verilog_stmt).is_verilog_dcls_stmt()
}

#[no_mangle]
pub extern "C" fn rust_is_module_stmt(c_verilog_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let verilog_stmt = unsafe { &mut *(c_verilog_stmt as *mut Box<dyn verilog_data::VerilogVirtualBaseStmt>) };

    (*verilog_stmt).is_module_stmt()
}

#[repr(C)]
pub struct RustVerilogFile {
    verilog_modules: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_convert_verilog_file(c_verilog_file: *const verilog_data::VerilogFile) -> *mut RustVerilogFile {
    unsafe {
        let verilog_modules_rust_vec = (*c_verilog_file).get_verilog_modules();

        let verilog_modules = rust_vec_to_c_array(verilog_modules_rust_vec);

        let rust_verilog_file = RustVerilogFile { verilog_modules };
        let rust_verilog_file_pointer = Box::new(rust_verilog_file);
        
        Box::into_raw(rust_verilog_file_pointer)
    }
}

#[no_mangle]
pub extern "C" fn rust_convert_rc_ref_cell_module(
    c_data: *const Rc<RefCell<verilog_data::VerilogModule>>,
) -> *mut c_void {
    unsafe {
        let module_ptr = (*c_data).deref().as_ptr();
        module_ptr as *mut c_void
    }
}
