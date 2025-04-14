//! The VCD file data structure include :
//! 1) VCDBit
//! 2) VCDValue
//! 3) VCDVariableType
//! 4) VCDSignal
//! 5) VCDScope
//! 6) VCDFile

use crate::vcd_parser::vcd_calc_tc_sp;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

/// VCD signal bit value.
#[derive(Eq, Hash, PartialEq, Copy, Clone)]
pub enum VCDBit {
    BitZero,
    BitOne,
    BitX,
    BitZ,
}

/// VCD value type.
#[derive(PartialEq)]
pub enum VCDValue {
    BitScalar(VCDBit),
    BitVector(Vec<VCDBit>),
    BitReal(f64),
}

/// VCD signal value, include time and value

#[derive(PartialEq)]
pub struct VCDTimeAndValue {
    pub time: i64,
    pub value: VCDValue,
}

impl VCDValue {
    pub fn get_bit_scalar(&self) -> VCDBit {
        match self {
            VCDValue::BitScalar(bit) => *bit,
            _ => panic!("Not a BitScalar"),
        }
    }

    pub fn get_vector_bit(&self, index: usize) -> VCDBit {

        match self {
            VCDValue::BitVector(bit_vec) => {
                let default_bit_value = VCDBit::BitZero;
                let bit_value = if index < bit_vec.len() {
                    bit_vec[index as usize]
                } else {
                    default_bit_value
                };

                bit_value
            }
            _ => panic!("Not a BitVector"),
        }

        
    }
}

/// VCD variable type of vcd signal
#[derive(Copy, Clone)]
#[repr(C)]
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
    Default,
}

/// VCD signal
pub struct VCDSignal {
    /// hash or called ref name.
    hash: String,
    name: String,
    bus_index: Option<(i32, i32)>,
    signal_size: u32,
    signal_type: VCDVariableType,
    scope: Option<Rc<RefCell<VCDScope>>>,
}

impl VCDSignal {
    pub fn new() -> Self {
        Self {
            hash: Default::default(),
            name: Default::default(),
            bus_index: None,
            signal_size: 1,
            signal_type: VCDVariableType::Default,
            scope: None,
        }
    }

    pub fn set_hash(&mut self, hash: String) {
        self.hash = hash;
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn set_bus_index(&mut self, left_index: i32, right_index: i32) {
        let bus_index = (left_index, right_index);
        self.bus_index = Some(bus_index);
    }

    pub fn set_signal_size(&mut self, signal_size: u32) {
        self.signal_size = signal_size;
    }

    pub fn set_signal_type(&mut self, type_str: &str) {
        match type_str {
            "event" => self.signal_type = VCDVariableType::VarEvent,
            "integer" => self.signal_type = VCDVariableType::VarInteger,
            "parameter" => self.signal_type = VCDVariableType::VarParameter,
            "real" => self.signal_type = VCDVariableType::VarReal,
            "realtime" => self.signal_type = VCDVariableType::VarRealTime,
            "reg" => self.signal_type = VCDVariableType::VarReg,
            "supply0" => self.signal_type = VCDVariableType::VarSupply0,
            "supply1" => self.signal_type = VCDVariableType::VarSupply1,
            "time" => self.signal_type = VCDVariableType::VarTime,
            "tri" => self.signal_type = VCDVariableType::VarTri,
            "triand" => self.signal_type = VCDVariableType::VarTriAnd,
            "trior" => self.signal_type = VCDVariableType::VarTriOr,
            "trireg" => self.signal_type = VCDVariableType::VarTriReg,
            "tri0" => self.signal_type = VCDVariableType::VarTri0,
            "tri1" => self.signal_type = VCDVariableType::VarTri1,
            "wand" => self.signal_type = VCDVariableType::VarWAnd,
            "wire" => self.signal_type = VCDVariableType::VarWire,
            "wor" => self.signal_type = VCDVariableType::VarWor,
            _ => panic!("unknown signal type {}", type_str),
        }
    }

    pub fn set_scope(&mut self, parent_scope: Rc<RefCell<VCDScope>>) {
        self.scope = Some(parent_scope);
    }

    pub fn get_hash(&self) -> &str {
        &self.hash
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_bus_index(&self) -> &Option<(i32, i32)> {
        &self.bus_index
    }

    pub fn get_signal_size(&self) -> u32 {
        self.signal_size
    }

    pub fn get_signal_type(&self) -> &VCDVariableType {
        &self.signal_type
    }

    pub fn get_scope(&self) -> &Option<Rc<RefCell<VCDScope>>> {
        &self.scope
    }
}

/// VCD Scope type
pub enum VCDScopeType {
    ScopeBegin,
    ScopeFork,
    ScopeFunction,
    ScopeModule,
    ScopeTask,
    // ScopeRoot,
}

/// VCD Scope
pub struct VCDScope {
    name: String,
    scope_type: VCDScopeType,
    parent_scope: Option<Rc<RefCell<VCDScope>>>,
    children_scopes: Vec<Rc<RefCell<VCDScope>>>,
    scope_signals: Vec<Rc<VCDSignal>>,
}

impl VCDScope {
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

    pub fn add_child_scope(&mut self, child_scope: Rc<RefCell<VCDScope>>) {
        self.children_scopes.push(child_scope);
    }

    pub fn add_scope_signal(&mut self, vcd_signal: VCDSignal) {
        self.scope_signals.push(vcd_signal.into());
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_parent_scope(&self) -> &Option<Rc<RefCell<VCDScope>>> {
        &self.parent_scope
    }

    pub fn set_parent_scope(&mut self, parent_scope: Rc<RefCell<VCDScope>>) {
        self.parent_scope = Some(parent_scope);
    }

    pub fn get_children_scopes(&self) -> &Vec<Rc<RefCell<VCDScope>>> {
        &self.children_scopes
    }

    pub fn get_scope_signals(&self) -> &Vec<Rc<VCDSignal>> {
        &self.scope_signals
    }
}

/// VCD time unit
#[derive(Copy, Clone)]
pub enum VCDTimeUnit {
    UnitSecond,
    UnitMS,
    UnitUS,
    UnitNS,
    UnitPS,
    UnitFS,
}

/// VCD File
pub struct VCDFile {
    start_time: i64,
    end_time: i64,
    time_resolution: u32,
    time_unit: VCDTimeUnit,
    date: String,
    version: String,
    comment: String,
    root_scope: Option<Rc<RefCell<VCDScope>>>,
    signal_values: HashMap<String, VecDeque<Box<VCDTimeAndValue>>>,
    signal_tc_vec: Vec<vcd_calc_tc_sp::SignalTC>,
    signal_duration_vec: Vec<vcd_calc_tc_sp::SignalDuration>,
}

impl VCDFile {
    pub fn new() -> Self {
        Self {
            start_time: 0,
            end_time: 0,
            time_unit: VCDTimeUnit::UnitNS,
            time_resolution: 0,
            date: Default::default(),
            version: Default::default(),
            comment: Default::default(),
            root_scope: Default::default(),
            signal_values: Default::default(),
            signal_tc_vec: Default::default(),
            signal_duration_vec: Default::default(),
        }
    }

    pub fn get_start_time(&mut self) -> i64 {
        self.start_time
    }
    pub fn _set_start_time(&mut self, start_time: i64) {
        self.start_time = start_time;
    }

    pub fn get_end_time(&self) -> i64 {
        self.end_time
    }
    pub fn set_end_time(&mut self, end_time: i64) {
        self.end_time = end_time;
    }

    pub fn get_time_resolution(&mut self) -> u32 {
        self.time_resolution
    }

    pub fn get_time_unit(&mut self) -> &VCDTimeUnit {
        &self.time_unit
    }

    pub fn get_date(&mut self) -> &str {
        &self.date
    }

    pub fn get_version(&mut self) -> &str {
        &self.version
    }

    pub fn get_comment(&mut self) -> &str {
        &self.comment
    }

    pub fn get_root_scope(&self) -> &Option<Rc<RefCell<VCDScope>>> {
        &self.root_scope
    }

    pub fn get_signal_values(&self) -> &HashMap<String, VecDeque<Box<VCDTimeAndValue>>> {
        &self.signal_values
    }

    pub fn set_root_scope(&mut self, root_scope: Rc<RefCell<VCDScope>>) {
        self.root_scope = Some(root_scope);
    }

    pub fn set_date(&mut self, date_text: String) {
        self.date = date_text;
    }

    #[allow(dead_code)]
    pub fn set_version(&mut self, version_text: String) {
        self.version = version_text;
    }

    pub fn set_time_unit(&mut self, time_unit: &str) {
        match time_unit {
            "s" => self.time_unit = VCDTimeUnit::UnitSecond,
            "ms" => self.time_unit = VCDTimeUnit::UnitMS,
            "us" => self.time_unit = VCDTimeUnit::UnitUS,
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

    pub fn add_signal_value(&mut self, signal_hash: String, signal_value: Box<VCDTimeAndValue>) {
        if self.signal_values.contains_key(&signal_hash) {
            let signal_vec = self.signal_values.get_mut(&signal_hash).unwrap();
            signal_vec.push_back(signal_value);
        } else {
            let mut signal_vec = VecDeque::<Box<VCDTimeAndValue>>::new();
            signal_vec.push_back(signal_value);
            self.signal_values.insert(signal_hash, signal_vec);
        }
    }

    pub fn set_signal_tc_vec(&mut self, signal_tc_vec: Vec<vcd_calc_tc_sp::SignalTC>) {
        self.signal_tc_vec = signal_tc_vec;
    }
    pub fn get_signal_tc_vec(&self) -> &Vec<vcd_calc_tc_sp::SignalTC> {
        return &self.signal_tc_vec;
    }

    pub fn set_signal_duration_vec(
        &mut self,
        signal_duration_vec: Vec<vcd_calc_tc_sp::SignalDuration>,
    ) {
        self.signal_duration_vec = signal_duration_vec;
    }

    pub fn get_signal_duration_vec(&self) -> &Vec<vcd_calc_tc_sp::SignalDuration> {
        return &self.signal_duration_vec;
    }
}

/// VCD File parser
pub struct VCDFileParser {
    vcd_file: VCDFile,
    scope_stack: VecDeque<Rc<RefCell<VCDScope>>>,
    current_time: i64,
}

impl VCDFileParser {
    pub fn new() -> Self {
        Self {
            vcd_file: VCDFile::new(),
            scope_stack: VecDeque::new(),
            current_time: 0,
        }
    }

    pub fn get_vcd_file(&mut self) -> &mut VCDFile {
        return &mut self.vcd_file;
    }

    pub fn get_vcd_file_imute(self) -> VCDFile {
        return self.vcd_file;
    }

    pub fn get_scope_stack(&mut self) -> &mut VecDeque<Rc<RefCell<VCDScope>>> {
        return &mut self.scope_stack;
    }
    pub fn is_scope_empty(&self) -> bool {
        return self.scope_stack.is_empty();
    }

    pub fn set_current_time(&mut self, current_time: i64) {
        self.current_time = current_time;
        self.vcd_file.set_end_time(current_time);
    }

    pub fn get_current_time(&self) -> i64 {
        return self.current_time;
    }

    pub fn set_root_scope(&mut self, root_scope: Rc<RefCell<VCDScope>>) {
        self.scope_stack.push_back(root_scope.clone());
        self.vcd_file.set_root_scope(root_scope);
    }
}
