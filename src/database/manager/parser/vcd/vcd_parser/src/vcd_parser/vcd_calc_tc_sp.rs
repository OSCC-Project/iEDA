use crate::vcd_parser::vcd_data;

use super::vcd_data::VCDScope;

use std::cell::RefCell;
use std::rc::Rc;

pub struct SignalTC {
    signal_name: String,
    signal_tc: u64,
}

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
    pub fn new(parent_scope: &Rc<RefCell<VCDScope>>, top_instance_name: &str) -> Self {
        let closure = Box::new(
            move |parent_scope: &Rc<RefCell<VCDScope>>, top_instance_name: &str| {
                let children_scopes = parent_scope
                    .as_ref()
                    .borrow_mut()
                    .get_children_scopes()
                    .clone();
                for child_scope in children_scopes.clone() {
                    if child_scope.clone().borrow_mut().get_name() == top_instance_name {
                        return Some(child_scope);
                    }
                }

                for child_scope in children_scopes.clone() {
                    let recursive_closure = FindScopeClosure::new(&child_scope, top_instance_name);
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
