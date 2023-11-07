use std::collections::HashMap;
use std::fmt::Debug;

/// spef line entry basic info and it's methods
#[derive(Clone, Debug)]
pub struct SpefEntryBasicInfo {
    file_name: String,
    line_no: usize,
}

impl SpefEntryBasicInfo {
    pub fn new(file_name: &str, line_no: usize) -> SpefEntryBasicInfo {
        SpefEntryBasicInfo { file_name: file_name.to_string(), line_no: line_no }
    }

    pub fn get_file_name(&self) -> &str {
        &self.file_name
    }

    pub fn get_line_no(&self) -> usize {
        self.line_no
    }
}

#[derive(Clone, Debug)]
pub enum SectionType {
    HEADER,
    PORTS,
    NAMEMAP,
    CONN,
    CAP,
    RES,
    END,
}

#[derive(Clone, Debug)]
pub struct SpefSectionEntry {
    basic_info: SpefEntryBasicInfo,
    section_type: SectionType,
}

impl SpefSectionEntry {
    pub fn new(file_name: &str, line_no: usize, section_type: SectionType) -> SpefSectionEntry {
        SpefSectionEntry { basic_info: SpefEntryBasicInfo::new(file_name, line_no), section_type }
    }

    pub fn get_basic_info(&self) -> &SpefEntryBasicInfo {
        &self.basic_info
    }

    pub fn get_section_type(&self) -> &SectionType {
        &self.section_type
    }
}

#[derive(Clone, Debug)]
pub struct SpefHeaderEntry {
    basic_info: SpefEntryBasicInfo,
    pub header_key: String,
    pub header_value: String,
}

impl SpefHeaderEntry {
    pub fn new(file_name: &str, line_no: usize, header_key: String, header_value: String) -> SpefHeaderEntry {
        SpefHeaderEntry {
            basic_info: SpefEntryBasicInfo::new(file_name, line_no),
            header_key: header_key,
            header_value: header_value,
        }
    }

    pub fn get_basic_info(&self) -> &SpefEntryBasicInfo {
        &self.basic_info
    }

    pub fn get_header_key(&self) -> &str {
        &self.header_key
    }

    pub fn get_header_value(&self) -> &str {
        &self.header_value
    }
}

/// Store each line of NameMap section
/// namemap entry example: *43353 us21\/_1057_
/// index: 43353
/// name: us21\/_1057_
#[derive(Clone, Debug)]
pub struct SpefNameMapEntry {
    basic_info: SpefEntryBasicInfo,
    pub index: usize,
    pub name: String,
}

impl SpefNameMapEntry {
    pub fn new(file_name: &str, line_no: usize, index: usize, name: &str) -> SpefNameMapEntry {
        SpefNameMapEntry { basic_info: SpefEntryBasicInfo::new(file_name, line_no), index, name: name.to_string() }
    }

    pub fn get_basic_info(&self) -> &SpefEntryBasicInfo {
        &self.basic_info
    }
}

/// Store each line of Port section
/// Port entry example: *37 I *C 633.84 0.242
/// name: "37"
/// direction: ConnectionType::INPUT
/// coordinate: (633.84, 0.242)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum ConnectionDirection {
    INPUT,
    OUTPUT,
    INOUT,
    Internal,
}

#[derive(Clone, Debug)]
pub struct SpefPortEntry {
    basic_info: SpefEntryBasicInfo,
    name: String,
    direction: ConnectionDirection,
    coordinate: (f64, f64),
}

impl SpefPortEntry {
    pub fn new(
        file_name: &str,
        line_no: usize,
        name: String,
        direction: ConnectionDirection,
        coordinate: (f64, f64),
    ) -> SpefPortEntry {
        SpefPortEntry { basic_info: SpefEntryBasicInfo::new(file_name, line_no), name, direction, coordinate }
    }

    pub fn get_basic_info(&self) -> &SpefEntryBasicInfo {
        &self.basic_info
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn get_direction(&self) -> ConnectionDirection {
        self.direction.clone()
    }

    pub fn get_coordinate(&self) -> (f64, f64) {
        // let mut result = Vec::<f64>::new();
        // result.push(self.coordinate.0);
        // result.push(self.coordinate.1);
        // result
        self.coordinate
    }
}

/// Store each line of Conn section
/// Conn entry example: *I *33272:Q O *C 635.66 405.835 *L 0 *D sky130_fd_sc_hd__dfxtp_1
/// name: "37"
/// direction: ConnectionType::INPUT
/// coordinate: (633.84, 0.242)
/// driving_cell: "sky130_fd_sc_hd__dfxtp_1"
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum ConnectionType {
    INTERNAL,
    EXTERNAL,
    UNITIALIZED,
}

#[derive(Clone, Debug)]
pub struct SpefConnEntry {
    basic_info: SpefEntryBasicInfo,
    pub conn_type: ConnectionType,
    pub conn_direction: ConnectionDirection,
    pub name: String,
    pub driving_cell: String,
    pub load: f64,
    pub layer: u32,

    pub coordinate: (f64, f64),
    pub ll_coordinate: (f64, f64),
    pub ur_coordinate: (f64, f64),
}

impl SpefConnEntry {
    pub fn new(
        file_name: &str,
        line_no: usize,
        conn_type: ConnectionType,
        conn_direction: ConnectionDirection,
        name: String,
    ) -> SpefConnEntry {
        SpefConnEntry {
            basic_info: SpefEntryBasicInfo::new(file_name, line_no),
            conn_type,
            conn_direction,
            name,
            driving_cell: String::new(),
            load: 0.0,
            layer: 0,
            coordinate: (-1.0, -1.0),
            ll_coordinate: (-1.0, -1.0),
            ur_coordinate: (-1.0, -1.0),
        }
    }

    pub fn get_basic_info(&self) -> &SpefEntryBasicInfo {
        &self.basic_info
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn get_conn_direction(&self) -> ConnectionDirection {
        self.conn_direction.clone()
    }

    pub fn get_conn_type(&self) -> ConnectionType {
        self.conn_type.clone()
    }

    pub fn get_xy_coordinate(&self) -> (f64, f64) {
        self.coordinate.clone()
    }
    pub fn get_ll_coordinate(&self) -> (f64, f64) {
        self.ll_coordinate.clone()
    }
    pub fn get_ur_coordinate(&self) -> (f64, f64) {
        self.ur_coordinate.clone()
    }

    pub fn set_layer(&mut self, layer: u32) {
        self.layer = layer;
    }
    pub fn get_layer(&self) -> u32 {
        self.layer
    }
    pub fn set_ll_corr(&mut self, coordinate: (f64, f64)) {
        self.ll_coordinate = coordinate;
    }
    pub fn set_ur_corr(&mut self, coordinate: (f64, f64)) {
        self.ur_coordinate = coordinate;
    }
    pub fn set_driving_cell(&mut self, driving_cell: String) {
        self.driving_cell = driving_cell;
    }
    pub fn get_driving_cell(&self) -> String {
        self.driving_cell.clone()
    }
    pub fn set_load(&mut self, load: f64) {
        self.load = load;
    }
    pub fn get_load(&self) -> f64 {
        self.load.clone()
    }
    pub fn set_xy_coordinate(&mut self, coordinate: (f64, f64)) {
        self.coordinate = coordinate;
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpefResCap {
    pub node1: String,
    pub node2: String,
    pub res_or_cap: f64,
}

/// Store everthing about a net
/// Conn entry example: 3 *1:2 0.000520945
/// name: "1:2"
/// direction: ConnectionType::INPUT
/// coordinate: (633.84, 0.242)
/// driving_cell: "sky130_fd_sc_hd__dfxtp_1"

#[derive(Clone, Debug, Default)]
pub struct SpefNet {
    pub name: String,
    pub line_no: usize,
    pub lcap: f64,
    pub connection: Vec<SpefConnEntry>,
    pub caps: Vec<SpefResCap>,
    pub ress: Vec<SpefResCap>,
}

impl SpefNet {
    pub fn new(line_no: usize, name: String, lcap: f64) -> SpefNet {
        SpefNet { name, line_no, lcap, connection: Vec::new(), caps: Vec::new(), ress: Vec::new() }
    }

    pub fn add_connection(&mut self, conn: &SpefConnEntry) {
        self.connection.push(conn.clone());
    }

    pub fn add_cap(&mut self, cap: (String, String, f64)) {
        let cap_item = SpefResCap { node1: cap.0, node2: cap.1, res_or_cap: cap.2 };
        self.caps.push(cap_item);
    }

    pub fn add_res(&mut self, res: (String, String, f64)) {
        let res_item = SpefResCap { node1: res.0, node2: res.1, res_or_cap: res.2 };
        self.ress.push(res_item);
    }

    pub fn get_net_name(&self) -> String {
        self.name.clone()
    }

    pub fn get_lcap(&self) -> f64 {
        self.lcap
    }

    pub fn get_conns(&self) -> &Vec<SpefConnEntry> {
        &self.connection
    }

    pub fn get_caps(&self) -> &Vec<SpefResCap> {
        &self.caps
    }

    pub fn get_ress(&self) -> &Vec<SpefResCap> {
        &self.ress
    }
}

#[derive(Clone, Debug)]
/// Spef Exchange data structure with cpp
pub struct SpefExchange {
    pub file_name: String,
    pub header: Vec<SpefHeaderEntry>,
    pub name_map: HashMap<usize, String>,
    pub ports: Vec<SpefPortEntry>,
    pub nets: Vec<SpefNet>,
}

impl SpefExchange {
    pub fn new(file_name: String) -> SpefExchange {
        SpefExchange { file_name, header: Vec::new(), name_map: HashMap::new(), ports: Vec::new(), nets: Vec::new() }
    }

    pub fn add_header_entry(&mut self, header: SpefHeaderEntry) {
        self.header.push(header);
    }
    pub fn add_namemap_entry(&mut self, namemap_entry: SpefNameMapEntry) {
        self.name_map.insert(namemap_entry.index, namemap_entry.name);
    }

    pub fn add_port_entry(&mut self, port_entry: SpefPortEntry) {
        self.ports.push(port_entry);
    }

    pub fn add_net(&mut self, net: SpefNet) {
        self.nets.push(net);
    }

    pub fn get_file_name(&self) -> &str {
        &self.file_name
    }
}

#[derive(Clone, Debug)]
pub enum SpefParserData {
    SectionEntry(SpefSectionEntry),
    HeaderEntry(SpefHeaderEntry),
    NameMapEntry(SpefNameMapEntry),
    PortEntry(SpefPortEntry),
    ConnEntry(SpefConnEntry),
    NetEntry(SpefNet),
    Exchange(SpefExchange),
    Unitialized,
}
