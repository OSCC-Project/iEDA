use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use csv::ReaderBuilder;
use nalgebra::DVector;

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

fn read_vcd_csv(file_path: &str) -> Result<Vec<InstancePowerRecord>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = csv::Reader::from_reader(file);
    let mut records = Vec::new();
    for result in reader.deserialize() {
        let record: InstancePowerRecord = result?;
        records.push(record);
    }
    Ok(records)
}

fn print_vcd_data(records: &[InstancePowerRecord]) {
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

#[cfg(test)]
mod vcd_data_tests {
    use super::*;

    #[test]
    fn read_vcd_csv_test() -> Result<(), Box<dyn Error>> {
        let file_path = "/home/shaozheqing/iEDA/bin/report_instance.csv";
        let vectors = read_vcd_csv(file_path)?;
        print_vcd_data(&vectors);
        Ok(())
    }
}
