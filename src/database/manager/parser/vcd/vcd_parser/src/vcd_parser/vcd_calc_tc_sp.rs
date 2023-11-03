use crate::vcd_parser::vcd_data;

use super::vcd_data::VCDBit;
use super::vcd_data::VCDFile;
use super::vcd_data::VCDScope;
use super::vcd_data::VCDSignal;
use super::vcd_data::VCDTimeAndValue;
use super::vcd_data::VCDValue;

use core::panic;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use threadpool::ThreadPool;

#[derive(Clone)]
pub struct SignalTC {
    signal_name: String,
    signal_tc: u64,
}

impl SignalTC {
    pub fn new(signal_name: String) -> Self {
        Self {
            signal_name,
            signal_tc: 0,
        }
    }

    pub fn incr_tc(&mut self) {
        self.signal_tc += 1;
    }
}

#[derive(Clone)]
pub struct SignalDuration {
    signal_name: String,
    bit_0_duration: u64,
    bit_1_duration: u64,
    bit_x_duration: u64,
    bit_z_duration: u64,
}

impl SignalDuration {
    pub fn new(signal_name: String) -> Self {
        Self {
            signal_name,
            bit_0_duration: 0,
            bit_1_duration: 0,
            bit_x_duration: 0,
            bit_z_duration: 0,
        }
    }
}

pub trait VcdCounter {
    fn is_trasition(
        &self,
        pre_time_value: &VCDTimeAndValue,
        cur_time_value: &VCDTimeAndValue,
        bus_index: Option<i32>,
    ) -> bool {
        if pre_time_value.time == cur_time_value.time {
            return false;
        }

        let pre_bit_value = match &pre_time_value.value {
            VCDValue::BitScalar(bit) => Ok(bit),
            VCDValue::BitVector(vec) => {
                if let Some(index) = bus_index {
                    Ok(&vec[index as usize])
                } else {
                    Err("No bus index provided")
                }
            }
            _ => Err("Unmatched value"),
        }
        .unwrap();

        let cur_bit_value = match &cur_time_value.value {
            VCDValue::BitScalar(bit) => Ok(bit),
            VCDValue::BitVector(vec) => {
                if let Some(index) = bus_index {
                    Ok(&vec[index as usize])
                } else {
                    Err("No bus index provided")
                }
            }
            _ => Err("Unmatched value"),
        }
        .unwrap();

        if *pre_bit_value != VCDBit::BitX
            && *cur_bit_value != VCDBit::BitX
            && *pre_bit_value != *cur_bit_value
        {
            return true;
        }

        false
    }

    fn get_duration(pre_time_value: &VCDTimeAndValue, cur_time_value: &VCDTimeAndValue) -> i64 {
        let pre_time = pre_time_value.time;
        let cur_time = cur_time_value.time;

        cur_time - pre_time
    }

    fn update_duration(signal_duration: &mut SignalDuration, bit_value: VCDBit, duration: u64) {
        match bit_value {
            VCDBit::BitZero => signal_duration.bit_0_duration += duration,
            VCDBit::BitOne => signal_duration.bit_1_duration += duration,
            VCDBit::BitX => signal_duration.bit_x_duration += duration,
            VCDBit::BitZ => signal_duration.bit_z_duration += duration,
        }
    }
}

pub struct VcdScalarCounter<'a> {
    top_vcd_scope: &'a VCDScope,
    vcd_file: &'a VCDFile,
    signal: &'a VCDSignal,
    pub signal_tc_vec: &'a mut Vec<SignalTC>,
    pub signal_duration_vec: &'a mut Vec<SignalDuration>,
}
impl<'a> VcdCounter for VcdScalarCounter<'a> {}

impl<'a> VcdScalarCounter<'a> {
    pub fn new(
        top_vcd_scope: &'a VCDScope,
        vcd_file: &'a VCDFile,
        signal: &'a VCDSignal,
        signal_tc_vec: &'a mut Vec<SignalTC>,
        signal_duration_vec: &'a mut Vec<SignalDuration>,
    ) -> Self {
        Self {
            top_vcd_scope: top_vcd_scope,
            vcd_file,
            signal,
            signal_tc_vec,
            signal_duration_vec,
        }
    }

    pub fn count_tc_and_glitch(&mut self) -> SignalTC {
        let signal_hash = self.signal.get_hash();
        let signal_time_values = self.vcd_file.get_signal_values().get(signal_hash);

        let signal_name = self.signal.get_name().to_string();

        let mut signal_toggle = SignalTC::new(signal_name);

        // count the toggle, if current signal value is rise transition or fall
        // transition, count add one.
        let mut prev_time_signal_value: Option<&vcd_data::VCDTimeAndValue> = None;

        if let Some(signal_time_values) = signal_time_values.as_deref() {
            for signal_time_value in signal_time_values {
                let signal_time_value = signal_time_value.as_ref();
                if let Some(prev) = prev_time_signal_value {
                    if self.is_trasition(prev, signal_time_value, None) {
                        // is transition, incr tc.
                        signal_toggle.incr_tc();
                    } else {
                        // TODO glitch
                    }
                }
                prev_time_signal_value = Some(signal_time_value);
            }
        }
        signal_toggle
    }

    pub fn count_duration(&mut self) {}

    pub fn run(&mut self) {}
}

pub struct FindScopeClosure {
    pub closure: Box<dyn Fn(&Rc<RefCell<VCDScope>>, &str) -> Option<Rc<RefCell<VCDScope>>>>,
}

impl FindScopeClosure {
    pub fn new() -> Self {
        let closure = Box::new(
            move |parent_scope: &Rc<RefCell<VCDScope>>, top_instance_name: &str| {
                let children_scopes = parent_scope
                    .as_ref()
                    .borrow_mut()
                    .get_children_scopes()
                    .clone();
                for child_scope in children_scopes.clone() {
                    if child_scope.borrow().get_name() == top_instance_name {
                        return Some(child_scope);
                    }
                }

                for child_scope in children_scopes.clone() {
                    let recursive_closure = FindScopeClosure::new();
                    if let Some(found_scope) =
                        (recursive_closure.closure)(&child_scope, top_instance_name)
                    {
                        return Some(found_scope);
                    }
                }
                None
            },
        );
        Self { closure }
    }
}

pub struct CalcTcAndSp<'a> {
    top_vcd_scope: &'a VCDScope,
    vcd_file: &'a VCDFile,
}

impl<'a> CalcTcAndSp<'a> {
    pub fn new(top_vcd_scope: &'a VCDScope, vcd_file: &'a VCDFile) -> Self {
        Self {
            top_vcd_scope,
            vcd_file,
        }
    }
    pub fn count_signal(
        &self,
        signal: &VCDSignal,
        signal_tc_vec: &mut Vec<SignalTC>,
        signal_duration_vec: &mut Vec<SignalDuration>,
    ) {
        let signal_size = signal.get_signal_size();
        if signal_size == 1 {
            // scalar signal
            let mut scalar_counter = VcdScalarCounter::new(
                self.top_vcd_scope,
                self.vcd_file,
                signal,
                signal_tc_vec,
                signal_duration_vec,
            );
            scalar_counter.run();
        } else {
            // bus signal
        }
    }

    pub fn traverse_scope_signal(
        &self,
        parent_scope: &VCDScope,
        thread_pool: &ThreadPool,
        signal_tc_vec: &mut Vec<SignalTC>,
        signal_duration_vec: &mut Vec<SignalDuration>,
    ) {
        let signals = parent_scope.get_scope_signals();

        for scope_signal in signals {
            if let vcd_data::VCDVariableType::VarWire = *scope_signal.get_signal_type() {
                /*count signal */
                let cur_signal = Arc::new(Mutex::new(scope_signal.deref()));
                // Select whether to use multithreading for count signal
                #[cfg(feature = "multithreading")]
                {
                    thread_pool.execute(move || {
                        self.count_signal(cur_signal, signal_tc_vec, signal_duration_vec);
                    });
                }

                #[cfg(not(feature = "multithreading"))]
                {
                    self.count_signal(scope_signal, signal_tc_vec, signal_duration_vec);
                }
            } else {
                continue;
            }
        }

        // View the next level of the scope
        let children_scopes = parent_scope.get_children_scopes();
        for child_scope in children_scopes {
            self.traverse_scope_signal(
                &child_scope.borrow(),
                thread_pool,
                signal_tc_vec,
                signal_duration_vec,
            );
        }
    }
}

// pub struct TraverseScopeClosure {
//     pub closure: Box<dyn Fn(&VCDScope, &ThreadPool)>,
//     // pub signal_tc_vec: Vec<SignalTC>,
//     // pub signal_duration_vec: Vec<SignalDuration>,
// }

// impl TraverseScopeClosure {
//     pub fn new(
//         signal_tc_vec: &mut Vec<SignalTC>,
//         signal_duration_vec: &mut Vec<SignalDuration>,
//     ) -> Self {
//         let closure = Box::new(|parent_scope: &VCDScope, thread_pool: &ThreadPool| {
//             let signals = parent_scope.get_scope_signals();
//             // Calculate the signal of the current layer scope
//             for scope_signal in signals {
//                 if let vcd_data::VCDVariableType::VarWire = *scope_signal.get_signal_type() {
//                     /*count signal */
//                     let cur_signal = Arc::new(Mutex::new(scope_signal.deref()));
//                     // Select whether to use multithreading for count signal
//                     #[cfg(feature = "multithreading")]
//                     {
//                         thread_pool.execute(move || {
//                             count_signal(cur_signal, signal_tc_vec, signal_duration_vec);
//                         });
//                     }

//                     #[cfg(not(feature = "multithreading"))]
//                     {
//                         count_signal(scope_signal, signal_tc_vec, signal_duration_vec);
//                     }
//                 } else {
//                     continue;
//                 }
//             }

//             // View the next level of the scope
//             let children_scopes = parent_scope.get_children_scopes();
//             for child_scope in children_scopes {
//                 let recursive_closure =
//                     TraverseScopeClosure::new(signal_tc_vec, signal_duration_vec);
//                 (recursive_closure.closure)(child_scope.borrow().deref(), thread_pool);
//             }
//         });

//         Self {
//             closure, // signal_tc_vec: signal_tc_vec.to_vec(),
//                      // signal_duration_vec: signal_duration_vec.to_vec(),
//         }
//     }
// }
