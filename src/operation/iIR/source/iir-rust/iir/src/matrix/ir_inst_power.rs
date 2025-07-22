use log;
use std::collections::HashMap;
use std::error::Error;

use std::fs::File;

use serde::de::StdError;
use serde::Deserialize;

use crate::matrix::ir_inst_power;
use crate::matrix::ir_rc::RCOneNetData;

use super::ir_rc::POWER_INNER_RESISTANCE;

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct InstancePowerRecord {
    #[serde(rename = "Instance Name")]
    pub instance_name: String,
    #[serde(rename = "Nominal Voltage")]
    pub nominal_voltage: f64,
    #[serde(rename = "Internal Power")]
    pub internal_power: f64,
    #[serde(rename = "Switch Power")]
    pub switch_power: f64,
    #[serde(rename = "Leakage Power")]
    pub leakage_power: f64,
    #[serde(rename = "Total Power")]
    pub total_power: f64,
}

/// Read instance power csv file.
pub fn read_instance_pwr_csv(file_path: &str) -> Result<Vec<InstancePowerRecord>, Box<dyn Error>> {
    let mut records = Vec::new();
    if let Ok(file) = File::open(file_path) {
        let mut reader = csv::Reader::from_reader(file);

        for result in reader.deserialize() {
            let record: InstancePowerRecord = result.expect("error read csv");
            records.push(record);
        }
    }
    Ok(records)
}

/// Print instance power data.
#[allow(dead_code)]
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
fn get_instance_current(instance_power_data: &Vec<InstancePowerRecord>) -> HashMap<String, f64> {
    let mut instance_current_map: HashMap<String, f64> = HashMap::new();
    for record in instance_power_data {
        let current = record.total_power / record.nominal_voltage;
        instance_current_map.insert(record.instance_name.clone(), current);
    }
    instance_current_map
}

/// Build instance current vector.
pub fn build_instance_current_vector(
    inst_power_data: &Vec<InstancePowerRecord>,
    net_data: &RCOneNetData,
) -> Result<HashMap<usize, f64>, Box<dyn StdError + 'static>> {
    let pg_net_name = net_data.get_name();
    log::info!("build instance current vector for power net {}", pg_net_name);
    let instance_current_map = ir_inst_power::get_instance_current(inst_power_data);

    let mut instance_current_data: HashMap<usize, f64> = HashMap::new();

    let nominal_voltage = inst_power_data[0].nominal_voltage;

    for (instance_name, instance_current) in instance_current_map {
        let mut instance_power_pin_name = instance_name; // TODO(to taosimin) fix power pin name.
        instance_power_pin_name += ":";
        instance_power_pin_name += pg_net_name;
        let node_index = net_data.get_node_id(&instance_power_pin_name).unwrap_or_else(|| {
            log::error!("node {} not found in net {}", instance_power_pin_name, pg_net_name);
            usize::MAX
        });
        
        if node_index != usize::MAX {
            instance_current_data.insert(node_index, instance_current);
        }
    }

    let nodes = net_data.get_nodes();
    for node in nodes.borrow().iter() {
        if node.get_is_bump() {
            let node_name = node.get_node_name();
            let node_index = net_data.get_node_id(node_name).unwrap();
            // bump current value is opposite of the instance value, so we use negative value instead.
            if pg_net_name.contains("VDD") {
                let current_val: f64 = -nominal_voltage / POWER_INNER_RESISTANCE;
                instance_current_data.insert(node_index, current_val);
            } else if pg_net_name.contains("VSS") {
                let current_val: f64 = 0.0; // VSS is ground, so current is 0.
                instance_current_data.insert(node_index, current_val);
            } else {
                panic!("unknown power net name {}", pg_net_name);
            }
        }
    }

    Ok(instance_current_data)
}

#[cfg(test)]
mod pwr_data_tests {
    use super::*;

    #[test]
    fn read_inst_pwr_csv_test() -> Result<(), Box<dyn Error>> {
        let file_path = "/home/shaozheqing/iEDA/bin/report_instance.csv";
        let vectors = read_instance_pwr_csv(file_path)?;
        print_inst_pwr_data(&vectors);
        Ok(())
    }
}
