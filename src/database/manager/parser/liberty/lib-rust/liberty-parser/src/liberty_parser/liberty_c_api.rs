use crate::liberty_parser::liberty_data;

use std::ffi::CString;
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
pub extern "C" fn lib_rust_vec_len(vec: &RustVec) -> usize {
    vec.len
}

// More functions to manipulate the Vec...

pub fn string_to_c_char(s: &str) -> *mut c_char {
    let cs = CString::new(s).unwrap();
    cs.into_raw()
}

#[no_mangle]
pub extern "C" fn lib_free_c_char(s: *mut c_char) {
    unsafe {
        let _ = CString::from_raw(s);
    }
}

#[repr(C)]
pub struct RustStingView {
    data: *const u8,
    len: usize,
}
#[no_mangle]
pub extern "C" fn test_string_to_view() -> RustStingView {
    let s = "abc";
    RustStingView { data: (s.as_ptr()), len: (s.len()) }
}

#[repr(C)]
pub struct RustLibertyStringValue {
    value: *mut c_char,
}

#[no_mangle]
pub extern "C" fn rust_convert_string_value(string_value: *mut c_void) -> *mut RustLibertyStringValue {
    let attribute_value = unsafe { &mut *(string_value as *mut Box<dyn liberty_data::LibertyAttrValue>) };

    let rust_value = (*attribute_value).get_string_value();
    let value = string_to_c_char(rust_value);

    let lib_value = RustLibertyStringValue { value };

    let lib_value_pointer = Box::new(lib_value);

    Box::into_raw(lib_value_pointer)
}

/// strint value converted value should be release by the API.
#[no_mangle]
pub extern "C" fn rust_free_string_value(c_string_value: *mut RustLibertyStringValue) {
    unsafe {
        lib_free_c_char((*c_string_value).value);
        let _: Box<RustLibertyStringValue> = Box::from_raw(c_string_value);
    }
}

#[repr(C)]
pub struct RustLibertyFloatValue {
    value: c_double,
}

#[no_mangle]
pub extern "C" fn rust_convert_float_value(float_value: *mut c_void) -> *mut RustLibertyFloatValue {
    let attribute_value = unsafe { &mut *(float_value as *mut Box<dyn liberty_data::LibertyAttrValue>) };
    let value = attribute_value.get_float_value();

    let lib_value = RustLibertyFloatValue { value };

    let lib_value_pointer = Box::new(lib_value);

    Box::into_raw(lib_value_pointer)
}

#[no_mangle]
pub extern "C" fn rust_free_float_value(c_float_value: *mut RustLibertyFloatValue) {
    unsafe {
        let _: Box<RustLibertyFloatValue> = Box::from_raw(c_float_value);
    }
}

#[no_mangle]
pub extern "C" fn rust_is_float_value(c_attribute_value: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyAttrValue
    let attribute_value = unsafe { &mut *(c_attribute_value as *mut Box<dyn liberty_data::LibertyAttrValue>) };

    (*attribute_value).is_float()
}

#[no_mangle]
pub extern "C" fn rust_is_string_value(c_attribute_value: *mut c_void) -> bool {
    let attribute_value = unsafe { &mut *(c_attribute_value as *mut Box<dyn liberty_data::LibertyAttrValue>) };

    (*attribute_value).is_string()
}

#[repr(C)]
pub struct RustLibertyGroupStmt {
    file_name: *mut c_char,
    line_no: usize,
    group_name: *mut c_char,
    attri_values: RustVec,
    stmts: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_convert_raw_group_stmt(
    group_stmt: *mut liberty_data::LibertyGroupStmt,
) -> *mut RustLibertyGroupStmt {
    unsafe {
        let file_name_str = (*group_stmt).get_attri().get_file_name();
        let file_name = string_to_c_char(file_name_str);
        let line_no = (*group_stmt).get_attri().get_line_no();
        let group_name_str = (*group_stmt).get_group_name();
        let group_name = string_to_c_char(group_name_str);
        let attri_values_rust_vec = (*group_stmt).get_attri_values();
        let stmts_rust_vec = (*group_stmt).get_stmts();

        let attri_values = rust_vec_to_c_array(attri_values_rust_vec);
        let stmts = rust_vec_to_c_array(stmts_rust_vec);

        let lib_group_stmt = RustLibertyGroupStmt { file_name, line_no, group_name, attri_values, stmts };
        let lib_group_stmt_pointer = Box::new(lib_group_stmt);

        Box::into_raw(lib_group_stmt_pointer)
    }
}

#[no_mangle]
pub extern "C" fn rust_free_group_stmt(c_group_stmt: *mut RustLibertyGroupStmt) {
    unsafe {
        lib_free_c_char((*c_group_stmt).file_name);
        lib_free_c_char((*c_group_stmt).group_name);

        let _: Box<RustLibertyGroupStmt> = Box::from_raw(c_group_stmt);
    }
}

/// The API differenct form rust_convert_raw_group_stmt is one use thin pointer, this use fat pointer.
#[no_mangle]
pub extern "C" fn rust_convert_group_stmt(
    c_group_stmt: *mut liberty_data::LibertyGroupStmt,
) -> *mut RustLibertyGroupStmt {
    let lib_stmt = unsafe { &mut *(c_group_stmt as *mut Box<dyn liberty_data::LibertyStmt>) };
    let group_stmt = (*lib_stmt).as_any().downcast_ref::<liberty_data::LibertyGroupStmt>().unwrap();

    let file_name_str = (*group_stmt).get_attri().get_file_name();
    let file_name = string_to_c_char(file_name_str);
    let line_no = (*group_stmt).get_attri().get_line_no();
    let group_name_str = (*group_stmt).get_group_name();
    let group_name = string_to_c_char(group_name_str);
    let attri_values_rust_vec = (*group_stmt).get_attri_values();
    let stmts_rust_vec = (*group_stmt).get_stmts();

    let attri_values = rust_vec_to_c_array(attri_values_rust_vec);
    let stmts = rust_vec_to_c_array(stmts_rust_vec);

    let lib_group_stmt = RustLibertyGroupStmt { file_name, line_no, group_name, attri_values, stmts };
    let lib_group_stmt_pointer = Box::new(lib_group_stmt);

    Box::into_raw(lib_group_stmt_pointer)
}

#[repr(C)]
pub struct RustLibertySimpleAttrStmt {
    file_name: *mut c_char,
    line_no: usize,
    attri_name: *mut c_char,
    attri_value: *const c_void,
}

#[no_mangle]
pub extern "C" fn rust_convert_simple_attribute_stmt(
    c_simple_attri_stmt: *mut c_void,
) -> *mut RustLibertySimpleAttrStmt {
    let lib_stmt = unsafe { &mut *(c_simple_attri_stmt as *mut Box<dyn liberty_data::LibertyStmt>) };
    let simple_attri_stmt = (*lib_stmt).as_any().downcast_ref::<liberty_data::LibertySimpleAttrStmt>().unwrap();

    let file_name_str = (*simple_attri_stmt).get_attri().get_file_name();
    let file_name = string_to_c_char(file_name_str);
    let line_no = (*simple_attri_stmt).get_attri().get_line_no();

    let attri_name_str = (*simple_attri_stmt).get_attri_name();
    let attri_name = string_to_c_char(attri_name_str);

    let attri_value_box = (*simple_attri_stmt).get_attri_value();
    let attri_value = attri_value_box as *const _ as *const c_void;

    let lib_simple_attri_stmt = RustLibertySimpleAttrStmt { file_name, line_no, attri_name, attri_value };

    let lib_simple_attri_stmt_pointer = Box::new(lib_simple_attri_stmt);

    Box::into_raw(lib_simple_attri_stmt_pointer)
}

#[no_mangle]
pub extern "C" fn rust_free_simple_attribute_stmt(c_simple_attri_stmt: *mut RustLibertySimpleAttrStmt) {
    unsafe {
        lib_free_c_char((*c_simple_attri_stmt).file_name);
        lib_free_c_char((*c_simple_attri_stmt).attri_name);

        let _: Box<RustLibertySimpleAttrStmt> = Box::from_raw(c_simple_attri_stmt);
    }
}

#[repr(C)]
pub struct RustLibertyComplexAttrStmt {
    file_name: *mut c_char,
    line_no: usize,
    attri_name: *mut c_char,
    attri_values: RustVec,
}

#[no_mangle]
pub extern "C" fn rust_convert_complex_attribute_stmt(
    c_complex_attri_stmt: *mut c_void,
) -> *mut RustLibertyComplexAttrStmt {
    let lib_stmt = unsafe { &mut *(c_complex_attri_stmt as *mut Box<dyn liberty_data::LibertyStmt>) };
    let complex_attri_stmt = (*lib_stmt).as_any().downcast_ref::<liberty_data::LibertyComplexAttrStmt>().unwrap();

    let file_name_str = (*complex_attri_stmt).get_attri().get_file_name();
    let file_name = string_to_c_char(file_name_str);
    let line_no = (*complex_attri_stmt).get_attri().get_line_no();

    let attri_name_str = (*complex_attri_stmt).get_attri_name();
    let attri_name = string_to_c_char(attri_name_str);

    let attri_values_rust_vec = (*complex_attri_stmt).get_attri_values();
    let attri_values = rust_vec_to_c_array(attri_values_rust_vec);

    let lib_complex_attri_stmt = RustLibertyComplexAttrStmt { file_name, line_no, attri_name, attri_values };

    let lib_complex_attri_stmt_pointer = Box::new(lib_complex_attri_stmt);

    Box::into_raw(lib_complex_attri_stmt_pointer)
}

#[no_mangle]
pub extern "C" fn rust_free_complex_attribute_stmt(c_complex_attri_stmt: *mut RustLibertyComplexAttrStmt) {
    unsafe {
        lib_free_c_char((*c_complex_attri_stmt).file_name);
        lib_free_c_char((*c_complex_attri_stmt).attri_name);

        let _: Box<RustLibertyComplexAttrStmt> = Box::from_raw(c_complex_attri_stmt);
    }
}

#[no_mangle]
pub extern "C" fn rust_is_simple_attri_stmt(c_lib_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let lib_stmt = unsafe { &mut *(c_lib_stmt as *mut Box<dyn liberty_data::LibertyStmt>) };

    (*lib_stmt).is_simple_attr_stmt()
}

#[no_mangle]
pub extern "C" fn rust_is_complex_attri_stmt(c_lib_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let lib_stmt = unsafe { &mut *(c_lib_stmt as *mut Box<dyn liberty_data::LibertyStmt>) };

    (*lib_stmt).is_complex_attr_stmt()
}

#[no_mangle]
pub extern "C" fn rust_is_attri_stmt(c_lib_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let lib_stmt = unsafe { &mut *(c_lib_stmt as *mut Box<dyn liberty_data::LibertyStmt>) };

    (*lib_stmt).is_attr_stmt()
}

#[no_mangle]
pub extern "C" fn rust_is_group_stmt(c_lib_stmt: *mut c_void) -> bool {
    // Casting c_void pointer to *mut dyn LibertyStmt
    let lib_stmt = unsafe { &mut *(c_lib_stmt as *mut Box<dyn liberty_data::LibertyStmt>) };

    (*lib_stmt).is_group_stmt()
}
