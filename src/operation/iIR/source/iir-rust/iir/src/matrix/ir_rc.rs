use log;
use spef::spef_parser;
use sprs::TriMat;
use sprs::TriMatI;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;

use super::c_str_to_r_str;
use super::RustIRPGNetlist;

pub const POWER_INNER_RESISTANCE: f64 = 1e-3;
pub const RC_COEFF: f64 = 3.0; // RC coefficient, used to scale the resistance value from SPEF.

/// RC node of the spef network.
pub struct RCNode {
    name: String,
    cap: f64,          // The node capacitance
    is_bump: bool,     // Whether power bump.
    is_inst_pin: bool, // Whether instance pin.
}

impl RCNode {
    pub fn new(name: String) -> RCNode {
        RCNode { name, cap: 0.0, is_bump: false, is_inst_pin: false }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_node_name(&self) -> &String {
        &self.name
    }
    #[allow(dead_code)]
    pub fn get_cap(&self) -> f64 {
        self.cap
    }
    pub fn set_cap(&mut self, cap: f64) {
        self.cap = cap;
    }

    pub fn set_is_bump(&mut self) {
        self.is_bump = true;
    }
    pub fn get_is_bump(&self) -> bool {
        self.is_bump
    }

    pub fn set_is_inst_pin(&mut self) {
        self.is_inst_pin = true;
    }
    pub fn get_is_inst_pin(&self) -> bool {
        self.is_inst_pin
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
    node_id_to_node_name: HashMap<usize, String>,
    nodes: RefCell<Vec<RCNode>>,
    resistances: Vec<RCResistance>,
}

impl RCOneNetData {
    pub fn new(name: String) -> RCOneNetData {
        RCOneNetData {
            name,
            node_name_to_node_id: HashMap::new(),
            node_id_to_node_name: HashMap::new(),
            nodes: RefCell::new(Vec::new()),
            resistances: Vec::new(),
        }
    }
    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn add_node(&mut self, one_node: RCNode) -> usize {
        let node_id = self.nodes.borrow().len();
        self.node_name_to_node_id.insert(String::from(one_node.get_name()), node_id);
        self.node_id_to_node_name.insert(node_id, String::from(one_node.get_name()));
        self.nodes.borrow_mut().push(one_node);
        node_id
    }

    pub fn get_nodes(&self) -> &RefCell<Vec<RCNode>> {
        &self.nodes
    }
    pub fn get_node_id(&self, node_name: &String) -> Option<usize> {
        self.node_name_to_node_id.get(node_name).cloned()
    }
    pub fn get_node_name(&self, node_id: usize) -> Option<&String> {
        self.node_id_to_node_name.get(&node_id)
    }

    pub fn set_node_cap(&self, node_id: usize, cap_value: f64) {
        if let Some(node) = self.nodes.borrow_mut().get_mut(node_id) {
            node.set_cap(cap_value);
        }
    }

    pub fn add_resistance(&mut self, one_resistance: RCResistance) {
        self.resistances.push(one_resistance);
    }

    pub fn get_resistances(&self) -> &Vec<RCResistance> {
        &self.resistances
    }

    pub fn print_to_yaml(&self, yaml_file_path: &str) {
        let mut file = std::fs::File::create(yaml_file_path).unwrap();
        for (index, node) in self.nodes.borrow().iter().enumerate() {
            let node_name = node.get_name();
            let node_id = format!("node_{}", index);

            writeln!(file, "{}:\n  {}", node_id, node_name).unwrap();
        }

        for (index, resistance) in self.resistances.iter().enumerate() {
            let edge_id = format!("edge_{}", index);

            writeln!(file, "{}:", edge_id).unwrap();
            writeln!(file, "  node1: {}", resistance.from_node_id).unwrap();
            writeln!(file, "  node2: {}", resistance.to_node_id).unwrap();

            writeln!(file, "  resistance: {}", resistance.resistance).unwrap();
        }
    }
}

/// All power net rc data.
#[derive(Default)]
pub struct RCData {
    rc_nets_data: HashMap<String, RCOneNetData>,
}

impl RCData {
    pub fn add_one_net_data(&mut self, one_net_data: RCOneNetData) {
        self.rc_nets_data.insert(String::from(one_net_data.get_name()), one_net_data);
    }
    pub fn get_nets_data(&self) -> &HashMap<String, RCOneNetData> {
        &self.rc_nets_data
    }

    pub fn get_one_net_data(&self, name: &str) -> &RCOneNetData {
        self.rc_nets_data.get(name).unwrap()
    }
    pub fn is_contain_net_data(&self, name: &str) -> bool {
        self.rc_nets_data.contains_key(name)
    }
}

pub fn split_spef_index_str(index_name: &str) -> (&str, &str) {
    let v: Vec<&str> = index_name.split(':').collect();
    let index_str = v.first().unwrap();
    let node_str = v.last().unwrap();
    if v.len() == 2 {
        (&index_str[1..], *node_str)
    } else {
        (&index_str[1..], "")
    }
}

/// Read rc data from spef file.
pub fn read_rc_data_from_spef(spef_file_path: &str) -> RCData {
    log::info!("read spef file {} start", spef_file_path);
    let spef_file = spef_parser::parse_spef_file(spef_file_path);
    log::info!("read spef file {} finish", spef_file_path);

    let node_name_map = &spef_file.index_to_name_map;
    let spef_data_nets = spef_file.get_nets();
    let mut rc_data = RCData::default();

    let spef_index_to_string = |index_str: &str| {
        let split_names = split_spef_index_str(index_str);
        let index = split_names.0.parse::<usize>().unwrap();
        let node_name = node_name_map.get(&index);
        if !split_names.1.is_empty() {
            let expand_node1_name = node_name.unwrap().clone() + ":" + split_names.1;
            return expand_node1_name;
        }
        String::from(node_name.unwrap())
    };

    log::info!("build net rc data start");

    // from the spef connection build bump port node and inst pin node.
    for spef_net in spef_data_nets {
        // println!("{:?}", spef_net);
        let spef_net_name = &spef_net.name;
        let net_name_str = spef_index_to_string(spef_net_name);
        log::info!("build net {} rc data", net_name_str);
        let mut one_net_data = RCOneNetData::new(net_name_str.clone());

        // build the bump and inst pin node.
        for conn_entry in spef_net.get_conns() {
            let conn_type = conn_entry.get_conn_type();
            let pin_port_name_index = conn_entry.get_pin_port_name();
            let pin_port_name = spef_index_to_string(pin_port_name_index);
            match conn_type {
                spef_parser::spef_data::ConnectionType::EXTERNAL => {
                    // bump port
                    let mut rc_node = RCNode::new(pin_port_name);
                    rc_node.set_is_bump();

                    one_net_data.add_node(rc_node);
                }
                spef_parser::spef_data::ConnectionType::INTERNAL => {
                    // inst pin
                    let mut rc_node = RCNode::new(pin_port_name);
                    rc_node.set_is_inst_pin();

                    one_net_data.add_node(rc_node);
                }
                _ => println!("TODO"),
            }
        }

        // build the cap node.
        for cap_entry in spef_net.get_caps() {
            if !cap_entry.node2.is_empty() {
                continue;
            }
            let name_index = &cap_entry.node1;
            let node_name = spef_index_to_string(name_index);
            let cap_value = cap_entry.res_or_cap;

            let node_id = one_net_data.get_node_id(&node_name);

            if node_id.is_none() {
                let mut rc_node = RCNode::new(node_name);
                rc_node.set_cap(cap_value);
                one_net_data.add_node(rc_node);
            } else {
                one_net_data.set_node_cap(node_id.unwrap(), cap_value);
            }
        }

        // build the internal node.
        for one_resistance in spef_net.get_ress() {
            let node1_name_index: &str = &one_resistance.node1;
            let node1_name = spef_index_to_string(node1_name_index);
            let node2_name_index: &str = &one_resistance.node2;
            let node2_name = spef_index_to_string(node2_name_index);
            let mut resistance_val = one_resistance.res_or_cap;

            resistance_val *= RC_COEFF; // scale the resistance value.

            let mut node_id = one_net_data.get_node_id(&node1_name);

            let node1_id = if node_id.is_none() {
                let rc_node = RCNode::new(node1_name);
                one_net_data.add_node(rc_node)
            } else {
                node_id.unwrap()
            };

            node_id = one_net_data.get_node_id(&node2_name);

            let node2_id = if node_id.is_none() {
                let rc_node2 = RCNode::new(node2_name);
                one_net_data.add_node(rc_node2)
            } else {
                node_id.unwrap()
            };

            let mut rc_resistance = RCResistance::default();
            rc_resistance.from_node_id = node1_id;
            rc_resistance.to_node_id = node2_id;
            rc_resistance.resistance = resistance_val;

            one_net_data.add_resistance(rc_resistance);
        }

        // if net_name_str == "VDD" {
        //     one_net_data.print_to_yaml("/home/taosimin/ir_example/aes/pg_netlist/rc_data.yaml");
        // }

        rc_data.add_one_net_data(one_net_data);
    }

    log::info!("build net rc data finish");
    rc_data
}

/// build rc data, rc node from pg node, rc edge from pg edge.
pub fn create_rc_data_from_topo(pg_netlist: &RustIRPGNetlist) -> RCOneNetData {
    let net_name = c_str_to_r_str(pg_netlist.net_name);
    let mut one_net_data = RCOneNetData::new(net_name.clone());

    for pg_node in pg_netlist.nodes.iter() {
        let node_id = pg_node.node_id;
        if pg_node.is_instance_pin || pg_node.is_bump {
            let node_name = c_str_to_r_str(pg_node.node_name);
            let mut rc_node = RCNode::new(node_name);

            if pg_node.is_bump {
                rc_node.set_is_bump();
            } else {
                rc_node.set_is_inst_pin();
            }

            one_net_data.add_node(rc_node);
        } else {
            let node_name = format!("{}:{}", net_name, node_id);
            let rc_node = RCNode::new(node_name);
            one_net_data.add_node(rc_node);
        }
    }

    for pg_edge in pg_netlist.edges.iter() {
        let node1_id = pg_edge.node1 as usize;
        let node2_id = pg_edge.node2 as usize;
        let mut rc_resistance = RCResistance::default();
        rc_resistance.from_node_id = node1_id;
        rc_resistance.to_node_id = node2_id;
        rc_resistance.resistance = pg_edge.resistance;
        one_net_data.add_resistance(rc_resistance);
    }

    one_net_data
}

/// Build conductance matrix from one net rc data.
pub fn build_conductance_matrix(rc_one_net_data: &RCOneNetData) -> TriMatI<f64, usize> {
    let nodes = rc_one_net_data.get_nodes();
    let resistances = rc_one_net_data.get_resistances();

    let matrix_size = nodes.borrow().len();
    let net_name = rc_one_net_data.get_name();
    log::info!("{} matrix size {}", net_name, matrix_size);

    let sum_resistance: f64 = resistances.iter().map(|x| x.resistance).sum();
    log::info!("{} sum resistance {}", net_name, sum_resistance);

    let mut g_matrix = TriMat::new((matrix_size, matrix_size));

    for node in nodes.borrow().iter() {
        if node.get_is_bump() {
            let node_name = node.get_node_name();
            let node_id = rc_one_net_data.get_node_id(node_name).unwrap();
            g_matrix.add_triplet(node_id, node_id, 1.0 / POWER_INNER_RESISTANCE);
        }
    }

    for rc_resistance in resistances {
        let node1_id = rc_resistance.from_node_id;
        let node2_id = rc_resistance.to_node_id;
        let resistance_val = rc_resistance.resistance;

        g_matrix.add_triplet(node1_id, node2_id, -1.0 / resistance_val);
        g_matrix.add_triplet(node2_id, node1_id, -1.0 / resistance_val);
        g_matrix.add_triplet(node1_id, node1_id, 1.0 / resistance_val);
        g_matrix.add_triplet(node2_id, node2_id, 1.0 / resistance_val);
    }

    g_matrix
}
