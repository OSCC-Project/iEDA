use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use csv::ReaderBuilder;
use nalgebra::DVector;

use crate::matrix::ir_inst_power;
use crate::matrix::ir_rc::RCOneNetData;
use log;
use serde::de::StdError;

#[derive(Deserialize)]
struct InstancePowerRecord {
    #[serde(rename = "Instance Name")]
    instance_name: String,
    #[serde(rename = "Nominal Voltage")]
    nominal_voltage: f64,
    #[serde(rename = "Internal Power")]
    internal_power: f64,
    #[serde(rename = "Switch Power")]
    switch_power: f64,
    #[serde(rename = "Leakage Power")]
    leakage_power: f64,
    #[serde(rename = "Total Power")]
    total_power: f64,
}

/// read instance power csv file.
pub fn read_inst_pwr_csv(file_path: &str) -> Result<Vec<InstancePowerRecord>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = csv::Reader::from_reader(file);
    let mut records = Vec::new();
    for result in reader.deserialize() {
        let record: InstancePowerRecord = result?;
        records.push(record);
    }
    Ok(records)
}

/// print instance power data.
fn print_inst_pwr_data(records: &[InstancePowerRecord]) {
    for record in records {
        println!(
        "Instance Name: {}, Nominal Voltage: {}, Internal Power: {}, Switch Power: {}, Leakage Power: {}, Total Power: {}",
        record.instance_name,
        record.nominal_voltage,
        record.internal_power,
        record.switch_power,
        record.leakage_power,
        record.total_power
      );
    }
}

/// generate instance current vector from instance power.
fn get_instance_current(instance_power_data: Vec<InstancePowerRecord>) -> HashMap<String, f64> {
    let mut instance_current_map: HashMap<String, f64> = HashMap::new();
    for record in instance_power_data {
        let current = record.total_power / record.nominal_voltage;
        instance_current_map.insert(record.instance_name, current);
    }
    instance_current_map
}

/// build instance current vector.
pub fn build_instance_current_vector(
    inst_power_path: &str,
    net_data: &RCOneNetData,
) -> Result<DVector<f64>, Box<dyn StdError + 'static>> {
    log::info!(
        "build instance current vector from {} for power net {}",
        inst_power_path,
        net_data.get_name()
    );
    let instance_power_data = ir_inst_power::read_inst_pwr_csv(inst_power_path)?;
    let instance_current_map = ir_inst_power::get_instance_current(instance_power_data);

    let mut instance_current_vec: Vec<f64> = vec![0.0; net_data.get_nodes().len()];

    for (instance_name, instance_current) in instance_current_map {
        let instance_power_pin_name = instance_name; // TODO(to taosimin) fix power pin name.
        let node_index = net_data.get_node_id(&instance_power_pin_name).unwrap();
        instance_current_vec[node_index] = instance_current;
    }

    let current_vector: DVector<f64> = DVector::from_iterator(
        instance_current_vec.len(),
        instance_current_vec.iter().cloned(),
    );

    Ok(current_vector)
}

#[cfg(test)]
mod pwr_data_tests {
    use super::*;

    #[test]
    fn read_inst_pwr_csv_test() -> Result<(), Box<dyn Error>> {
        let file_path = "/home/shaozheqing/iEDA/bin/report_instance.csv";
        let vectors = read_inst_pwr_csv(file_path)?;
        print_inst_pwr_data(&vectors);
        Ok(())
    }
}
