//! The liberty datastructure include:
//! 1) liberty attribute value : float value, string value.
//! 2) liberty statement :
//!     simple attribute statement,
//!     complex attribute statement,
//!     group statement.
//!

trait LibertyAttrValue {
    fn is_string(&self) -> u32 {
        0
    }
    fn is_float(&self) -> u32 {
        0
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
struct LibertyFloatValue {
    value: f64,
}

impl LibertyAttrValue for LibertyFloatValue {
    fn is_float(&self) -> u32 {
        1
    }

    fn get_float_value(&self) -> f64 {
        self.value
    }
}

/// liberty string value.
/// # Examples
/// "0.0010,0.0020,0.0030,0.0040,0.0050,0.0060,0.0070"
struct LibertyStringValue {
    value: String,
}

impl LibertyAttrValue for LibertyStringValue {
    fn is_string(&self) -> u32 {
        1
    }

    fn get_string_value(&self) -> &str {
        &self.value
    }
}

/// liberty stmt.
trait LibertyStmt {
    fn is_simple_attr_stmt(&self) -> u32 {
        0
    }
    fn is_complex_attr_stmt(&self) -> u32 {
        0
    }
    fn is_attr_stmt(&self) -> u32 {
        0
    }
    fn is_group_stmt(&self) -> u32 {
        0
    }
}

/// liberty attribute stmt.
struct LibertyAttrStmt {
    file_name: String,
    line_no: u32,
}

impl LibertyAttrStmt {
    fn new(file_name: &str, line_no: u32) -> LibertyAttrStmt {
        LibertyAttrStmt { file_name: file_name.to_string(), line_no: line_no }
    }
}

/// The simple attribute statement.
/// # Example
/// capacitance : 1.774000e-01;
struct LibertySimpleAttrStmt {
    attri: LibertyAttrStmt,
    attri_name: String,
    attri_value: Box<dyn LibertyAttrValue>,
}

impl LibertySimpleAttrStmt {
    fn new(
        file_name: &str,
        line_no: u32,
        attri_name: &str,
        attri_value: Box<dyn LibertyAttrValue>,
    ) -> LibertySimpleAttrStmt {
        LibertySimpleAttrStmt {
            attri: LibertyAttrStmt::new(file_name, line_no),
            attri_name: attri_name.to_string(),
            attri_value: attri_value,
        }
    }
}

/// The complex attribute statement.
/// # Example
/// index_1 ("0.0010,0.0020,0.0030");
struct LibertyComplexAttrStmt {
    attri: LibertyAttrStmt,
    attri_name: String,
    attri_values: Vec<Box<dyn LibertyAttrValue>>,
}

impl LibertyComplexAttrStmt {
    fn new(
        file_name: &str,
        line_no: u32,
        attri_name: &str,
        attri_values: Vec<Box<dyn LibertyAttrValue>>,
    ) -> LibertyComplexAttrStmt {
        LibertyComplexAttrStmt {
            attri: LibertyAttrStmt::new(file_name, line_no),
            attri_name: attri_name.to_string(),
            attri_values: attri_values,
        }
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
struct LibertyGroupStmt {
    attri: LibertyAttrStmt,
    group_name: String,
    attri_values: Vec<Box<dyn LibertyAttrValue>>,
    stmts: Vec<Box<LibertyAttrStmt>>,
}

impl LibertyGroupStmt {
    fn new(
        file_name: &str,
        line_no: u32,
        group_name: &str,
        attri_values: Vec<Box<dyn LibertyAttrValue>>,
        stmts: Vec<Box<LibertyAttrStmt>>,
    ) -> LibertyGroupStmt {
        LibertyGroupStmt {
            attri: LibertyAttrStmt::new(file_name, line_no),
            group_name: group_name.to_string(),
            attri_values: attri_values,
            stmts: stmts,
        }
    }
}
