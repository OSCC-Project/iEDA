use crate::vcd_parser::vcd_data;

use super::vcd_data::VCDBit;
use super::vcd_data::VCDFile;
use super::vcd_data::VCDScope;
use super::vcd_data::VCDSignal;
use super::vcd_data::VCDTimeAndValue;
use super::vcd_data::VCDValue;

use std::cell::RefCell;
// use std::collections::HashMap;
// use std::ops::Deref;
use std::rc::Rc;
// use std::sync::{Arc, Mutex};

use threadpool::ThreadPool;

#[derive(Clone)]
pub struct SignalTC {
    pub signal_name: String,
    pub signal_tc: u64,
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

    pub fn get_name(&self) -> &str {
        &self.signal_name
    }
}

#[derive(Clone)]
pub struct SignalDuration {
    pub signal_name: String,
    pub bit_0_duration: u64,
    pub bit_1_duration: u64,
    pub bit_x_duration: u64,
    pub bit_z_duration: u64,
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

    pub fn get_name(&self) -> &str {
        &self.signal_name
    }
}

pub trait VcdCounter {
    fn is_transition(
        &self,
        pre_time_value: &VCDTimeAndValue,
        cur_time_value: &VCDTimeAndValue,
        bus_index: Option<i32>,
    ) -> bool {
        if pre_time_value.time == cur_time_value.time {
            return false;
        }

        let default_bit_value = VCDBit::BitZero;

        let pre_bit_value = match &pre_time_value.value {
            VCDValue::BitScalar(bit) => Ok(bit),
            VCDValue::BitVector(bit_vec) => {
                let index = bus_index.unwrap() as usize;

                let bit_value = if index < bit_vec.len() {
                    &bit_vec[index as usize]
                } else {
                    &default_bit_value
                };
                Ok(bit_value)
            }
            _ => Err("Unmatched value"),
        }
        .unwrap();

        let cur_bit_value = match &cur_time_value.value {
            VCDValue::BitScalar(bit) => Ok(bit),
            VCDValue::BitVector(bit_vec) => {
                let index = bus_index.unwrap() as usize;

                let bit_value = if index < bit_vec.len() {
                    &bit_vec[index as usize]
                } else {
                    &default_bit_value
                };
                Ok(bit_value)
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

    fn get_duration(
        &self,
        pre_time_value: &VCDTimeAndValue,
        cur_time_value: &VCDTimeAndValue,
    ) -> i64 {
        let pre_time = pre_time_value.time;
        let cur_time = cur_time_value.time;

        cur_time - pre_time
    }

    fn update_duration(
        &self,
        signal_duration: &mut SignalDuration,
        bit_value: VCDBit,
        duration: u64,
    ) {
        match bit_value {
            VCDBit::BitZero => signal_duration.bit_0_duration += duration,
            VCDBit::BitOne => signal_duration.bit_1_duration += duration,
            VCDBit::BitX => signal_duration.bit_x_duration += duration,
            VCDBit::BitZ => signal_duration.bit_z_duration += duration,
        }
    }
}

pub struct VcdScalarCounter<'a> {
    vcd_file: &'a VCDFile,
    signal: &'a VCDSignal,
    pub signal_tc_vec: &'a mut Vec<SignalTC>,
    pub signal_duration_vec: &'a mut Vec<SignalDuration>,
}
impl<'a> VcdCounter for VcdScalarCounter<'a> {}

impl<'a> VcdScalarCounter<'a> {
    pub fn new(
        vcd_file: &'a VCDFile,
        signal: &'a VCDSignal,
        signal_tc_vec: &'a mut Vec<SignalTC>,
        signal_duration_vec: &'a mut Vec<SignalDuration>,
    ) -> Self {
        Self {
            vcd_file,
            signal,
            signal_tc_vec,
            signal_duration_vec,
        }
    }

    fn count_tc_and_glitch(&mut self) {
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
                    if self.is_transition(prev, signal_time_value, None) {
                        // is transition, incr tc.
                        signal_toggle.incr_tc();
                    } else {
                        // TODO glitch
                    }
                }
                prev_time_signal_value = Some(signal_time_value);
            }
        }
        self.signal_tc_vec.push(signal_toggle)
    }

    fn count_duration(&mut self) {
        let signal_hash = self.signal.get_hash();
        let signal_time_values = self.vcd_file.get_signal_values().get(signal_hash);

        //TODO set simulation time
        let simulation_end_time = self.vcd_file.get_end_time();

        let signal_name = self.signal.get_name().to_string();
        let mut annotate_signal_duration_time = SignalDuration::new(signal_name);
        // count signal t0,t1,tx,tz duration, the signal may be not start zero time,
        // need consider the start time, such as t0, we accumulate the VCD bit0 time.
        let mut prev_time_signal_value: Option<&vcd_data::VCDTimeAndValue> = None;
        if let Some(signal_time_values) = signal_time_values.as_deref() {
            for signal_time_value in signal_time_values {
                let signal_time_value = signal_time_value.as_ref();
                if let Some(prev) = prev_time_signal_value {
                    let duration = self.get_duration(prev, signal_time_value);
                    let prev_bit_value = &prev.value;

                    let one_bit_value = prev_bit_value.get_bit_scalar();
                    self.update_duration(
                        &mut annotate_signal_duration_time,
                        one_bit_value,
                        duration.try_into().unwrap(),
                    );
                }
                prev_time_signal_value = Some(signal_time_value);
            }
        }

        // for last time, the signal should steady to end.
        if let Some(signal_time_values) = signal_time_values {
            if let Some(last_time_signal_value) = signal_time_values.back() {
                let last_time = last_time_signal_value.time;
                let last_bit_value = last_time_signal_value.value.get_bit_scalar();
                let last_time_duration = simulation_end_time - last_time;
                self.update_duration(
                    &mut annotate_signal_duration_time,
                    last_bit_value,
                    last_time_duration.try_into().unwrap(),
                );
            }
        }
        self.signal_duration_vec.push(annotate_signal_duration_time);
    }

    pub fn run(&mut self) {
        self.count_tc_and_glitch();
        self.count_duration();
    }
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

pub struct FindSignalClosure {
    pub closure: Box<dyn Fn(&Rc<RefCell<VCDScope>>, &str) -> Option<Rc<VCDSignal>>>,
}

impl FindSignalClosure {
    pub fn new() -> Self {
        let closure = Box::new(move |scope: &Rc<RefCell<VCDScope>>, signal_name: &str| {
            let signals = scope.borrow().get_scope_signals().clone();
            for signal in signals.clone() {
                if signal.get_name() == signal_name {
                    return Some(signal);
                }
            }

            let children_scopes = scope.borrow().get_children_scopes().clone();
            for child_scope in children_scopes.clone() {
                if let Some(found_signal) =
                    (FindSignalClosure::new().closure)(&child_scope, signal_name)
                {
                    return Some(found_signal);
                }
            }
            None
        });
        Self { closure }
    }
}

pub struct VcdBusCounter<'a> {
    vcd_file: &'a VCDFile,
    signal: &'a VCDSignal,
    pub signal_tc_vec: &'a mut Vec<SignalTC>,
    pub signal_duration_vec: &'a mut Vec<SignalDuration>,
}
impl<'a> VcdCounter for VcdBusCounter<'a> {}

impl<'a> VcdBusCounter<'a> {
    pub fn new(
        vcd_file: &'a VCDFile,
        signal: &'a VCDSignal,
        signal_tc_vec: &'a mut Vec<SignalTC>,
        signal_duration_vec: &'a mut Vec<SignalDuration>,
    ) -> Self {
        Self {
            vcd_file,
            signal,
            signal_tc_vec,
            signal_duration_vec,
        }
    }
    pub fn count_tc_and_glitch(&mut self) {
        let signal_hash = self.signal.get_hash();
        let signal_time_values = self.vcd_file.get_signal_values().get(signal_hash);

        let signal_name = self.signal.get_name().to_string();
        // println!("signal name {}", signal_name);

        // count the toggle, if current signal value is rise transition or fall
        // transition, count add one.

        /*get bus size by lindex and rindex */
        let (lindex, rindex) = self.signal.get_bus_index().unwrap();
        let bus_size = lindex - rindex + 1;

        let mut annotate_signal_toggles: Vec<SignalTC> = Vec::new();
        for i in 0..bus_size {
            let name = format!("{}[{}]", signal_name, i);
            annotate_signal_toggles.push(SignalTC::new(name));
        }

        let mut prev_time_signal_value: Option<&vcd_data::VCDTimeAndValue> = None;

        // loop access to the bus signal
        for i in rindex..=lindex {
            let vec_i = lindex - i; // bus signal value is high bit first.
            if let Some(signal_time_values) = signal_time_values.as_deref() {
                for signal_time_value in signal_time_values {
                    let signal_time_value = signal_time_value.as_ref();
                    if let Some(prev) = prev_time_signal_value {
                        if self.is_transition(prev, signal_time_value, Some(vec_i)) {
                            // is transition, incr tc.
                            /*Each signal inside the bus needs to be recorded separately*/
                            let i_usize = i as usize; // convert i to usize
                            annotate_signal_toggles[i_usize].incr_tc();
                        } else {
                            // TODO glitch
                        }
                    }
                    prev_time_signal_value = Some(signal_time_value);
                }
            }
        }
        self.signal_tc_vec.extend(annotate_signal_toggles);
    }

    pub fn count_duration(&mut self) {
        let signal_hash = self.signal.get_hash();
        let signal_time_values: Option<&std::collections::VecDeque<Box<VCDTimeAndValue>>> =
            self.vcd_file.get_signal_values().get(signal_hash);

        //TODO set simulation time
        let simulation_end_time = self.vcd_file.get_end_time();

        let signal_name = self.signal.get_name().to_string();
        // count signal t0,t1,tx,tz duration, the signal may be not start zero time,
        // need consider the start time, such as t0, we accumulate the VCD bit0 time.

        /*get bus size by lindex and rindex */
        let (lindex, rindex) = self.signal.get_bus_index().unwrap();
        let bus_size = lindex - rindex + 1;

        let mut annotate_signal_duration_times: Vec<SignalDuration> = Vec::new();
        for i in 0..bus_size {
            let name = format!("{}[{}]", signal_name, i);
            annotate_signal_duration_times.push(SignalDuration::new(name));
        }

        let mut prev_time_signal_values: Vec<Option<&VCDTimeAndValue>> =
            vec![None; bus_size.try_into().unwrap()];

        // loop access to the bus signal
        for i in rindex..=lindex {
            let vec_i = lindex - i;
            let mut if_first_value = true;

            if let Some(signal_time_values) = signal_time_values.as_deref() {
                for signal_time_value in signal_time_values {
                    let signal_time_value = signal_time_value.as_ref();
                    if let Some(prev_time_signal_value) = prev_time_signal_values[i as usize] {
                        let prev_vector = &prev_time_signal_value.value;

                        let bit_value = signal_time_value.value.get_vector_bit(vec_i as usize);
                        let prev_bit_value = prev_vector.get_vector_bit(vec_i as usize);

                        // if is the last signal value of this bus signal or current value is different from prev
                        if bit_value != prev_bit_value
                            || signal_time_value == signal_time_values.back().unwrap().as_ref()
                        {
                            let duration =
                                self.get_duration(prev_time_signal_value, signal_time_value);

                            self.update_duration(
                                &mut annotate_signal_duration_times[i as usize],
                                prev_bit_value,
                                duration.try_into().unwrap(),
                            );

                            prev_time_signal_values[i as usize] = Some(signal_time_value);
                        }
                    }
                    if if_first_value {
                        prev_time_signal_values[i as usize] = Some(signal_time_value);
                        if_first_value = false;
                    }
                }
            }
        }

        // for last time, the signal should steady to end.
        if let Some(signal_time_values) = signal_time_values {
            if let Some(last_time_signal_value) = signal_time_values.back() {
                let last_time = last_time_signal_value.time;

                let last_time_duration = simulation_end_time - last_time;
                for i in rindex..=lindex {
                    let vec_i = lindex - i;
                    let last_bit_value =
                        last_time_signal_value.value.get_vector_bit(vec_i as usize);

                    self.update_duration(
                        &mut annotate_signal_duration_times[i as usize],
                        last_bit_value,
                        last_time_duration.try_into().unwrap(),
                    );
                }
            }
        }
        self.signal_duration_vec
            .extend(annotate_signal_duration_times);
    }

    pub fn run(&mut self) {
        self.count_tc_and_glitch();
        self.count_duration();
    }
}

pub struct CalcTcAndSp<'a> {
    vcd_file: &'a VCDFile,
}

impl<'a> CalcTcAndSp<'a> {
    pub fn new(vcd_file: &'a VCDFile) -> Self {
        Self { vcd_file }
    }
    fn count_signal(
        &self,
        signal: &VCDSignal,
        signal_tc_vec: &mut Vec<SignalTC>,
        signal_duration_vec: &mut Vec<SignalDuration>,
    ) {
        let signal_size = signal.get_signal_size();
        if signal_size == 1 {
            // scalar signal
            let mut scalar_counter =
                VcdScalarCounter::new(self.vcd_file, signal, signal_tc_vec, signal_duration_vec);
            scalar_counter.run();
        } else {
            // bus signal
            let mut bus_counter =
                VcdBusCounter::new(self.vcd_file, signal, signal_tc_vec, signal_duration_vec);
            bus_counter.run();
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
                // let cur_signal = Arc::new(Mutex::new(scope_signal.deref()));
                // Select whether to use multithreading for count signal

                // thread_pool.execute(move || {
                //     self.count_signal(cur_signal, signal_tc_vec, signal_duration_vec);
                // });

                self.count_signal(scope_signal, signal_tc_vec, signal_duration_vec);
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
