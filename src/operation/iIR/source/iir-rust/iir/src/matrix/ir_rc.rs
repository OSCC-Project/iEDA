use spef_parser::spef_parser;
use std::collections::HashMap;

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
    from_node_id: u32,
    to_node_id: u32,
    resistance: f32,
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
    pub fn add_node(&mut self, one_node: RCNode) {
        self.node_name_to_node_id
            .insert(String::from(one_node.get_name()), self.nodes.len() - 1);
        self.nodes.push(one_node);
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
}

/// read rc data from spef file.
pub fn read_rc_data_from_spef(spef_file_path: &str) {
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

            let rc_node = RCNode::new(String::from(node1_name));
            one_net_data.add_node(rc_node);
        }

        rc_data.add_one_net_data(one_net_data);
    }
}
