//! The liberty datastructure include:
//! 1) liberty attribute value : float value, string value.
//! 2) liberty statement :
//!     simple attribute statement,
//!     complex attribute statement,
//!     group statement.
//!

use std::fmt::Debug;

pub trait LibertyAttrValue: Debug {
    fn is_string(&self) -> bool {
        false
    }
    fn is_float(&self) -> bool {
        false
    }

    fn get_float_value(&self) -> f64 {
        panic!("This is unknown value.");
    }
    fn get_string_value(&self) -> &str {
        panic!("This is unknown value.");
    }
}

/// liberty float value.
/// # Examples
/// 1.7460
#[derive(Debug)]
pub struct LibertyFloatValue {
    pub(crate) value: f64,
}

impl LibertyAttrValue for LibertyFloatValue {
    fn is_float(&self) -> bool {
        true
    }

    fn get_float_value(&self) -> f64 {
        self.value
    }
}

/// liberty string value.
/// # Examples
/// "0.0010,0.0020,0.0030,0.0040,0.0050,0.0060,0.0070"
#[derive(Debug)]
pub struct LibertyStringValue {
    pub(crate) value: String,
}

impl LibertyAttrValue for LibertyStringValue {
    fn is_string(&self) -> bool {
        true
    }

    fn get_string_value(&self) -> &str {
        &self.value
    }
    
    fn get_float_value(&self) -> f64 {
        // Try to parse string as float, fallback to panic if not possible
        self.value.parse::<f64>().unwrap_or_else(|_| {
            panic!("This is unknown value.");
        })
    }
}

/// liberty stmt.
pub trait LibertyStmt: Debug {
    fn is_simple_attr_stmt(&self) -> bool {
        false
    }
    fn is_complex_attr_stmt(&self) -> bool {
        false
    }
    fn is_attr_stmt(&self) -> bool {
        false
    }
    fn is_group_stmt(&self) -> bool {
        false
    }
    fn as_any(&self) -> &dyn std::any::Any;
}

/// liberty attribute stmt.
#[derive(Debug)]
pub struct LibertyAttrStmt {
    file_name: String,
    line_no: usize,
}

impl LibertyAttrStmt {
    fn new(file_name: &str, line_no: usize) -> LibertyAttrStmt {
        LibertyAttrStmt { file_name: file_name.to_string(), line_no }
    }

    pub fn get_file_name(&self) -> &str {
        &self.file_name
    }

    pub fn get_line_no(&self) -> usize {
        self.line_no
    }
}

/// The simple attribute statement.
/// # Example
/// capacitance : 1.774000e-01;
#[derive(Debug)]
pub struct LibertySimpleAttrStmt {
    attri: LibertyAttrStmt,
    attri_name: String,
    attri_value: Box<dyn LibertyAttrValue>,
}

impl LibertySimpleAttrStmt {
    pub fn new(
        file_name: &str,
        line_no: usize,
        attri_name: &str,
        attri_value: Box<dyn LibertyAttrValue>,
    ) -> LibertySimpleAttrStmt {
        LibertySimpleAttrStmt {
            attri: LibertyAttrStmt::new(file_name, line_no),
            attri_name: attri_name.to_string(),
            attri_value,
        }
    }

    pub fn get_attri(&self) -> &LibertyAttrStmt {
        &self.attri
    }

    pub fn get_attri_name(&self) -> &str {
        &self.attri_name
    }

    pub fn get_attri_value(&self) -> &Box<dyn LibertyAttrValue> {
        &self.attri_value
    }
}

impl LibertyStmt for LibertySimpleAttrStmt {
    fn is_simple_attr_stmt(&self) -> bool {
        true
    }
    fn is_attr_stmt(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The complex attribute statement.
/// # Example
/// index_1 ("0.0010,0.0020,0.0030");
#[derive(Debug)]
pub struct LibertyComplexAttrStmt {
    attri: LibertyAttrStmt,
    attri_name: String,
    attri_values: Vec<Box<dyn LibertyAttrValue>>,
}

impl LibertyComplexAttrStmt {
    pub fn new(
        file_name: &str,
        line_no: usize,
        attri_name: &str,
        attri_values: Vec<Box<dyn LibertyAttrValue>>,
    ) -> LibertyComplexAttrStmt {
        LibertyComplexAttrStmt {
            attri: LibertyAttrStmt::new(file_name, line_no),
            attri_name: attri_name.to_string(),
            attri_values,
        }
    }

    pub fn get_attri(&self) -> &LibertyAttrStmt {
        &self.attri
    }

    pub fn get_attri_name(&self) -> &str {
        &self.attri_name
    }

    pub fn get_attri_values(&self) -> &Vec<Box<dyn LibertyAttrValue>> {
        &self.attri_values
    }
}

impl LibertyStmt for LibertyComplexAttrStmt {
    fn is_complex_attr_stmt(&self) -> bool {
        true
    }
    fn is_attr_stmt(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The group statement.
/// # Example
///
/// wire_load("5K_hvratio_1_1") {
/// capacitance : 1.774000e-01;
/// resistance : 3.571429e-03;
/// slope : 5.000000;
/// fanout_length( 1, 1.7460 );
/// fanout_length( 2, 3.9394 );
/// fanout_length( 3, 6.4626 );
/// fanout_length( 4, 9.2201 );
/// fanout_length( 5, 11.9123 );
/// fanout_length( 6, 14.8358 );
/// fanout_length( 7, 18.6155 );
/// fanout_length( 8, 22.6727 );
/// fanout_length( 9, 25.4842 );
/// fanout_length( 11, 27.0320 );
/// }
#[derive(Debug)]
pub struct LibertyGroupStmt {
    attri: LibertyAttrStmt,
    group_name: String,
    attri_values: Vec<Box<dyn LibertyAttrValue>>,
    stmts: Vec<Box<dyn LibertyStmt>>,
}

impl LibertyGroupStmt {
    pub fn new(
        file_name: &str,
        line_no: usize,
        group_name: &str,
        attri_values: Vec<Box<dyn LibertyAttrValue>>,
        stmts: Vec<Box<dyn LibertyStmt>>,
    ) -> LibertyGroupStmt {
        LibertyGroupStmt {
            attri: LibertyAttrStmt::new(file_name, line_no),
            group_name: group_name.to_string(),
            attri_values,
            stmts,
        }
    }

    pub fn get_group_name(&self) -> &str {
        &self.group_name
    }

    pub fn get_attri(&self) -> &LibertyAttrStmt {
        &self.attri
    }
    #[allow(dead_code)]
    pub fn get_attri_name(&self) -> &str {
        self.attri_values.first().unwrap().get_string_value()
    }

    pub fn get_attri_values(&self) -> &Vec<Box<dyn LibertyAttrValue>> {
        &self.attri_values
    }

    pub fn get_stmts(&self) -> &Vec<Box<dyn LibertyStmt>> {
        &self.stmts
    }
}

impl LibertyStmt for LibertyGroupStmt {
    fn is_group_stmt(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub enum LibertyParserData {
    GroupStmt(LibertyGroupStmt),
    ComplexStmt(LibertyComplexAttrStmt),
    SimpleStmt(LibertySimpleAttrStmt),
    String(LibertyStringValue),
    Float(LibertyFloatValue),
}
