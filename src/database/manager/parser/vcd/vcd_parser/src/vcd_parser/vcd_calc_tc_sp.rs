use crate::vcd_parser::vcd_data;

use super::vcd_data::VCDScope;
use super::vcd_data::VCDSignal;

use core::panic;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use threadpool::ThreadPool;

#[derive(Clone)]
pub struct SignalTC {
    signal_name: String,
    signal_tc: u64,
}

#[derive(Clone)]
pub struct SignalDuration {
    signal_name: String,
    bit_0_duration: u64,
    bit_1_duration: u64,
    bit_x_duration: u64,
    bit_z_duration: u64,
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

fn count_signal(
    signal: &VCDSignal,
    signal_tc_vec: &mut Vec<SignalTC>,
    signal_duration_vec: &mut Vec<SignalDuration>,
) {
    let signal_size = signal.get_signal_size();
    if signal_size == 1 {
        // scalar signal
    } else {
        // bus signal
    }
}

pub fn traverse_scope_signal(
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
                    count_signal(cur_signal, signal_tc_vec, signal_duration_vec);
                });
            }

            #[cfg(not(feature = "multithreading"))]
            {
                count_signal(scope_signal, signal_tc_vec, signal_duration_vec);
            }
        } else {
            continue;
        }
    }

    // View the next level of the scope
    let children_scopes = parent_scope.get_children_scopes();
    for child_scope in children_scopes {
        traverse_scope_signal(
            &child_scope.borrow(),
            thread_pool,
            signal_tc_vec,
            signal_duration_vec,
        );
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
