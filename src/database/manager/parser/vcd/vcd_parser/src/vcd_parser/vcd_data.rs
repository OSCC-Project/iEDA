//! The VCD file data structure include :
//! 1) VCDBit
//! 2) VCDValue
//! 3) VCDVariableType
//! 4) VCDSignal
//! 5) VCDScope
//! 6) VCDFile

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

/// VCD signal bit value.
pub enum VCDBit {
    BitZero(u8),
    BitOne(u8),
    BitX(u8),
    BitZ(u8),
}

/// VCD value type.
pub enum VCDValue {
    BitScalar(VCDBit),
    BitVector(Vec<VCDBit>),
    BitReal(f64),
}

/// VCD signal value, include time and value
pub struct VCDTimeAndValue {
    time: u64,
    value: VCDValue,
}

/// VCD variable type of vcd signal
pub enum VCDVariableType {
    VarEvent,
    VarInteger,
    VarParameter,
    VarReal,
    VarRealTime,
    VarReg,
    VarSupply0,
    VarSupply1,
    VarTime,
    VarTri,
    VarTriAnd,
    VarTriOr,
    VarTriReg,
    VarTri0,
    VarTri1,
    VarWAnd,
    VarWire,
    VarWor,
}

/// VCD signal
pub struct VCDSignal<'a> {
    hash: String,
    reference_name: String,
    signal_size: u32,
    var_type: VCDVariableType,
    left_index: i32,
    right_index: i32,
    scope: &'a VCDScope<'a>,
}

/// VCD Scope type
pub enum VCDScopeType {
    ScopeBegin,
    ScopeFork,
    ScopeFunction,
    ScopeModule,
    ScopeTask,
    ScopeRoot,
}

/// VCD Scope
pub struct VCDScope<'a> {
    name: String,
    scope_type: VCDScopeType,
    parent_scope: Option<Rc<RefCell<VCDScope<'a>>>>,
    children_scopes: Vec<Rc<RefCell<VCDScope<'a>>>>,
    scope_signals: Vec<Rc<VCDSignal<'a>>>,
}

impl<'a> VCDScope<'a> {
    pub fn new(name: String) -> Self {
        Self {
            name,
            scope_type: VCDScopeType::ScopeModule,
            parent_scope: None,
            children_scopes: Default::default(),
            scope_signals: Default::default(),
        }
    }

    pub fn set_scope_type(&mut self, type_str: &str) {
        match type_str {
            "module" => self.scope_type = VCDScopeType::ScopeModule,
            "begin" => self.scope_type = VCDScopeType::ScopeBegin,
            "fork" => self.scope_type = VCDScopeType::ScopeFork,
            "function" => self.scope_type = VCDScopeType::ScopeFunction,
            "task" => self.scope_type = VCDScopeType::ScopeTask,
            _ => panic!("unknown scope type {}", type_str),
        }
    }

    pub fn set_parent_scope(&mut self, parent_scope: Rc<RefCell<VCDScope<'a>>>) {
        self.parent_scope = Some(parent_scope);
    }

    pub fn add_child_scope(&mut self, child_scope: Rc<RefCell<VCDScope<'a>>>) {
        self.children_scopes.push(child_scope);
    }
}

/// VCD time unit
pub enum VCDTimeUnit {
    UnitSecond,
    UnitMS,
    UnitNS,
    UnitPS,
    UnitFS,
}

/// VCD File
pub struct VCDFile<'a> {
    start_time: i64,
    end_time: i64,
    time_unit: VCDTimeUnit,
    time_resolution: u32,
    date: String,
    version: String,
    comment: String,
    scope_root: Option<Arc<VCDScope<'a>>>,
    signal_values: HashMap<String, VecDeque<Box<VCDTimeAndValue>>>,
}

impl<'a> VCDFile<'a> {
    pub fn new() -> Self {
        Self {
            start_time: 0,
            end_time: 0,
            time_unit: VCDTimeUnit::UnitNS,
            time_resolution: 0,
            date: Default::default(),
            version: Default::default(),
            comment: Default::default(),
            scope_root: Default::default(),
            signal_values: Default::default(),
        }
    }

    pub fn set_date(&mut self, date_text: String) {
        self.date = date_text;
    }

    pub fn set_time_unit(&mut self, time_unit: &str) {
        match time_unit {
            "s" => self.time_unit = VCDTimeUnit::UnitSecond,
            "ms" => self.time_unit = VCDTimeUnit::UnitMS,
            "ns" => self.time_unit = VCDTimeUnit::UnitNS,
            "ps" => self.time_unit = VCDTimeUnit::UnitPS,
            "fs" => self.time_unit = VCDTimeUnit::UnitFS,
            _ => panic!("unknown time unit {}", time_unit),
        }
    }

    pub fn set_time_resolution(&mut self, time_resolution: u32) {
        self.time_resolution = time_resolution;
    }

    pub fn set_comment(&mut self, comment: String) {
        self.comment = comment;
    }
}

/// VCD File parser
pub struct VCDFileParser<'a> {
    vcd_file: VCDFile<'a>,
    scope_stack: VecDeque<Rc<RefCell<VCDScope<'a>>>>,
}

impl<'a> VCDFileParser<'a> {
    pub fn new() -> Self {
        Self {
            vcd_file: VCDFile::new(),
            scope_stack: VecDeque::new(),
        }
    }

    pub fn get_vcd_file(&mut self) -> &mut VCDFile<'a> {
        return &mut self.vcd_file;
    }

    pub fn get_vcd_file_imute(self) -> VCDFile<'a> {
        return self.vcd_file;
    }

    pub fn get_scope_stack(&mut self) -> &mut VecDeque<Rc<RefCell<VCDScope<'a>>>> {
        return &mut self.scope_stack;
    }
}

/// unified vcd data structure
pub enum VCDParserData<'a> {
    FileData(VCDFile<'a>),
}
