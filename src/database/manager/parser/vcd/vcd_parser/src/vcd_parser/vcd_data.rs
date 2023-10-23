//! The VCD file data structure include :
//! 1) VCDBit
//! 2) VCDValue
//! 3) VCDVariableType
//! 4) VCDSignal
//! 5) VCDScope
//! 6) VCDFile

use std::collections::{HashMap, VecDeque};

/// VCD signal bit value.
enum VCDBit {
    BitZero(u8),
    BitOne(u8),
    BitX(u8),
    BitZ(u8),
}

/// VCD value type.
enum VCDValue {
    BitScalar(VCDBit),
    BitVector(Vec<VCDBit>),
    BitReal(f64),
}

/// VCD signal value, include time and value
struct VCDTimeAndValue {
    time: u64,
    value: VCDValue,
}

/// VCD variable type of vcd signal
enum VCDVariableType {
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
struct VCDSignal<'a> {
    hash: String,
    reference_name: String,
    signal_size: u32,
    var_type: VCDVariableType,
    left_index: i32,
    right_index: i32,
    scope: &'a VCDScope<'a>,
}

/// VCD Scope type
enum VCDScopeType {
    ScopeBegin,
    ScopeFork,
    ScopeFunction,
    ScopeModule,
    ScopeTask,
    ScopeRoot,
}

/// VCD Scope
struct VCDScope<'a> {
    name: String,
    scope_type: VCDScopeType,
    parent_scope: &'a VCDScope<'a>,
    children_scopes: Vec<Box<VCDScope<'a>>>,
    scope_signals: Vec<Box<VCDSignal<'a>>>,
}

/// VCD time unit
enum VCDTimeUnit {
    UnitSecond,
    UnitMS,
    UnitNS,
    UnitPS,
    UnitFS,
}

/// VCD File
struct VCDFile<'a> {
    start_time: i64,
    end_time: i64,
    time_unit: VCDTimeUnit,
    time_resulution: u32,
    date: String,
    version: String,
    comment: String,
    scope_root: Box<VCDScope<'a>>,
    signal_values: HashMap<String, VecDeque<Box<VCDTimeAndValue>>>,
}
