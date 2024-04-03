use spef_parser::spef_parser;
use std::collections::HashMap;

extern crate nalgebra as na;
use na::{DMatrix};

/// RC node of the spef network.
pub struct RCNode {
    name: String,
    /// whether power bump.
    is_bump: bool,
    is_inst_pin: bool,
}

impl RCNode {
    pub fn new(name: String) -> RCNode {
        RCNode {
            name,
            is_bump: false,
            is_inst_pin: false,
        }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn set_is_bump(&mut self) {
        self.is_bump = true;
    }

    pub fn set_is_inst_pin(&mut self) {
        self.is_inst_pin = true;
    }
}

/// RC resistance.
#[derive(Default)]
pub struct RCResistance {
    pub from_node_id: usize,
    pub to_node_id: usize,
    pub resistance: f64,
}

/// One power net rc data.
#[derive(Default)]
pub struct RCOneNetData {
    name: String,
    node_name_to_node_id: HashMap<String, usize>,
    nodes: Vec<RCNode>,
    resistances: Vec<RCResistance>,
}

impl RCOneNetData {
    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn add_node(&mut self, one_node: RCNode) -> usize {
        let node_id = self.nodes.len();
        self.node_name_to_node_id
            .insert(String::from(one_node.get_name()), node_id);
        self.nodes.push(one_node);
        node_id
    }

    pub fn get_nodes(&self) -> &Vec<RCNode> {
        &self.nodes
    }
    pub fn get_node_id(&self, node_name:&String) -> Option<usize> {
        self.node_name_to_node_id.get(node_name).cloned()
    }

    pub fn get_resistances(&self) -> &Vec<RCResistance> {
        &self.resistances
    }
}

/// all power net rc data.
#[derive(Default)]
pub struct RCData {
    power_nets_data: HashMap<String, RCOneNetData>,
}

impl RCData {
    pub fn add_one_net_data(&mut self, one_net_data: RCOneNetData) {
        self.power_nets_data
            .insert(String::from(one_net_data.get_name()), one_net_data);
    }
    pub fn get_power_nets_data(&self) -> &HashMap<String, RCOneNetData> {
        &self.power_nets_data
    }
}

/// read rc data from spef file.
pub fn read_rc_data_from_spef(spef_file_path: &str) -> RCData {
    let spef_file = spef_parser::parse_spef_file(spef_file_path);
    let spef_data_nets = spef_file.get_nets();

    let mut rc_data = RCData::default();

    // from the spef connection build bump port node and inst pin node.
    for spef_net in spef_data_nets {
        // println!("{:?}", spef_net);
        let mut one_net_data = RCOneNetData::default();

        // build the power bump and inst pin node.
        for conn_entry in spef_net.get_conns() {
            let conn_type = conn_entry.get_conn_type();
            let pin_port_name = conn_entry.get_pin_port_name();
            match conn_type {
                spef_parser::spef_data::ConnectionType::EXTERNAL => {
                    // bump port
                    let mut rc_node = RCNode::new(String::from(pin_port_name));
                    rc_node.set_is_bump();

                    one_net_data.add_node(rc_node);
                }
                spef_parser::spef_data::ConnectionType::INTERNAL => {
                    // inst pin
                    let mut rc_node = RCNode::new(String::from(pin_port_name));
                    rc_node.set_is_inst_pin();

                    one_net_data.add_node(rc_node);
                }
                _ => println!("TODO"),
            }
        }

        // build the internal PDN node.
        for one_resistance in spef_net.get_ress() {
            let node1_name: &str = &one_resistance.node1;
            let node2_name: &str = &one_resistance.node2;
            let resistance_val = one_resistance.res_or_cap;

            let rc_node1 = RCNode::new(String::from(node1_name));
            let node1_id = one_net_data.add_node(rc_node1);

            let rc_node2 = RCNode::new(String::from(node2_name));
            let node2_id = one_net_data.add_node(rc_node2);

            let mut rc_resistance = RCResistance::default();
            rc_resistance.from_node_id = node1_id;
            rc_resistance.to_node_id = node2_id;
            rc_resistance.resistance = resistance_val;
        }

        rc_data.add_one_net_data(one_net_data);
    }

    rc_data
}

/// build conductance matrix from one net rc data.
pub fn build_conductance_matrix(rc_one_net_data: &RCOneNetData) -> DMatrix<f64> {
    let nodes = rc_one_net_data.get_nodes();
    let resistances = rc_one_net_data.get_resistances();

    let matrix_size = nodes.len();
    let mut arr = vec![vec![0.0; matrix_size]; matrix_size];

    //TODO process the
    for rc_resistance in resistances {
        let node1_id = rc_resistance.from_node_id;
        let node2_id = rc_resistance.to_node_id;
        let resistance_val = rc_resistance.resistance;

        arr[node1_id][node2_id] = -1.0 / resistance_val;
        arr[node2_id][node1_id] = -1.0 / resistance_val;
        arr[node1_id][node1_id] += 1.0 / resistance_val;
        arr[node2_id][node2_id] += 1.0 / resistance_val;
    }

    let matrix: DMatrix<f64> = DMatrix::from_row_slice(
        arr.len(),
        arr[0].len(),
        arr.iter()
            .flatten()
            .map(|&x| x)
            .collect::<Vec<_>>()
            .as_slice(),
    );

    matrix
}




extern crate quickcheck;

use quickcheck::{TestResult};

use super::*;

#[test]
fn test_build_conductance_matrix() {
    let one_net_data = RCOneNetData {
        name: "test_net".to_string(),
        node_name_to_node_id: HashMap::new(),
        nodes: vec![
            RCNode::new("node1".to_string()),
            RCNode::new("node2".to_string()),
        ],
        resistances: vec![
            RCResistance {
                from_node_id: 0,
                to_node_id: 1,
                resistance: 1.0,
            }
        ],
    };

    let matrix = build_conductance_matrix(&one_net_data);
    let expected_matrix = DMatrix::from_row_slice(
        2,
        2,
        &[
            1.0, -1.0,
            -1.0, 1.0,
        ],
    );

    assert_eq!(matrix, expected_matrix);
}