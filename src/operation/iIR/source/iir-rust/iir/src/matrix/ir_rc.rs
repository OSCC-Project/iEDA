use spef_parser::spef_parser;
use std::collections::HashMap;
use log;
use sprs::{TriMat, TriMatI};

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
    pub fn new(name: String) -> RCOneNetData {
        RCOneNetData {
            name,
            node_name_to_node_id: HashMap::new(),
            nodes: Vec::new(),
            resistances: Vec::new(),
        }
    }
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
    pub fn get_node_id(&self, node_name: &String) -> Option<usize> {
        self.node_name_to_node_id.get(node_name).cloned()
    }

    pub fn get_resistances(&self) -> &Vec<RCResistance> {
        &self.resistances
    }
}

/// All power net rc data.
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

/// Read rc data from spef file.
pub fn read_rc_data_from_spef(spef_file_path: &str) -> RCData {
    log::info!("read spef file {} start", spef_file_path);
    let spef_file = spef_parser::parse_spef_file(spef_file_path);
    log::info!("read spef file {} finish", spef_file_path);

    let node_name_map = &spef_file.index_to_name_map;
    let spef_data_nets = spef_file.get_nets();
    let mut rc_data = RCData::default();

    let spef_index_to_string = |index_str: &str| {
        let split_names = spef_parser::spef_c_api::split_spef_index_str(&index_str);
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
        let net_name_str = spef_index_to_string(&spef_net_name);
        log::info!("build net {} rc data", net_name_str);
        let mut one_net_data = RCOneNetData::new(net_name_str);

        // build the power bump and inst pin node.
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

        // build the internal PDN node.
        for one_resistance in spef_net.get_ress() {
            let node1_name_index: &str = &one_resistance.node1;
            let node1_name = spef_index_to_string(node1_name_index);
            let node2_name_index: &str = &one_resistance.node2;
            let node2_name = spef_index_to_string(node2_name_index);
            let resistance_val = one_resistance.res_or_cap;

            let rc_node1 = RCNode::new(node1_name);
            let node1_id = one_net_data.add_node(rc_node1);

            let rc_node2 = RCNode::new(node2_name);
            let node2_id = one_net_data.add_node(rc_node2);

            let mut rc_resistance = RCResistance::default();
            rc_resistance.from_node_id = node1_id;
            rc_resistance.to_node_id = node2_id;
            rc_resistance.resistance = resistance_val;
        }

        rc_data.add_one_net_data(one_net_data);
    }

    log::info!("build net rc data finish");
    rc_data
}

/// Build conductance matrix from one net rc data.
pub fn build_conductance_matrix(rc_one_net_data: &RCOneNetData) -> TriMatI<f64,usize> {
    let nodes = rc_one_net_data.get_nodes();
    let resistances = rc_one_net_data.get_resistances();

    let matrix_size = nodes.len();
    log::info!("matrix size {}", matrix_size);

    let mut g_matrix = TriMat::new((matrix_size, matrix_size));

    //TODO(to taosimin) process the bump node.
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




