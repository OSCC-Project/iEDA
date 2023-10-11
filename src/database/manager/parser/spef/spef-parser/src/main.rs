use pest::Parser;
use pest_derive::Parser;
use std::collections::HashMap;
use std::fs;

#[derive(Parser)]
#[grammar = "spef.pest"]

pub struct SpefParser;

fn main() {
    enum Direction {
        I,
        O,
        B,
        X,
        Uninitialized,
    }
    enum ConnType {
        P,
        I,
        S,
        C,
        R,
        L,
        Uninitialized,
    }

    enum Name {
        IndexName(i64),
        StrName(String),
        PinPort(String),
    }

    struct ConnEntry {
        conn_type: ConnType,
        name: Name,
        direction: Direction,
        coordinates: (f64, f64),
        load_val: f64,
        driver: Name,
    }

    type HeaderSection<'a> = HashMap<&'a str, &'a str>;
    type NameMapSection<'a> = HashMap<i64, &'a str>;
    type PortsSection = Vec<(Name, Direction, (f64, f64))>;
    type ConnSection = Vec<ConnEntry>;
    type CapSection = Vec<(i64, Name, f64)>;
    type ResSection = Vec<(i64, Name, Name, f64)>;

    struct DNetSection {
        name: Name,
        total_cap: f64,
        conn_section: ConnSection,
        cap_section: CapSection,
        res_section: ResSection,
    }

    // The top spef data structure
    struct SpefData<'a> {
        header: HeaderSection<'a>,
        namemap: NameMapSection<'a>,
        ports: PortsSection,
        dnets: Vec<DNetSection>,
    }

    let unparsed_file = fs::read_to_string("aes.spef").expect("cannot read file");
    let file = SpefParser::parse(Rule::file, &unparsed_file).expect("unsuccessful parse");

    // Store info of each line.
    let mut header: HeaderSection = HashMap::new();
    let mut name_map: NameMapSection = HashMap::new();
    let mut ports: PortsSection = Vec::new();
    let mut dnets: Vec<DNetSection> = Vec::new();

    // Mark the current section and cuurent dnet with default value
    let mut current_section_name = "*HEADER";
    let mut current_dnet: DNetSection = DNetSection {
        name: Name::IndexName(0),
        total_cap: 0.0,
        conn_section: Vec::new(),
        cap_section: Vec::new(),
        res_section: Vec::new(),
    };

    // counter
    let mut header_count = 0;
    let mut name_map_count = 0;
    let mut port_count = 0;
    let mut dnet_count = 0;

    for line in file {
        match line.as_rule() {
            Rule::section => {
                let inner_rules = line;
                current_section_name = inner_rules.as_str();

                if current_section_name == "*END" {
                    dnets.push(current_dnet);
                    current_dnet = DNetSection {
                        name: Name::IndexName(0),
                        total_cap: 0.0,
                        conn_section: Vec::new(),
                        cap_section: Vec::new(),
                        res_section: Vec::new(),
                    };
                }
            }
            Rule::header_entry => {
                // println!("match a header_entry now.");
                if current_section_name == "*HEADER" {
                    header_count = header_count + 1;
                    let mut inner_rules = line.into_inner();
                    println!("{inner_rules:#?}");
                    let header_keyword = inner_rules.next().unwrap().as_str();
                    let header_value = inner_rules.next().unwrap().as_str();
                    header.insert(header_keyword, header_value);
                }
            }
            Rule::name_map_entry => {
                // println!("match a name_map_entry now.");
                if current_section_name == "*NAME_MAP" {
                    name_map_count = name_map_count + 1;
                    let mut inner_rules = line.into_inner();

                    let index_name_pair = inner_rules.next();
                    let index_pair = index_name_pair.unwrap().into_inner().next();
                    let index = index_pair.unwrap().as_str().parse::<i64>().unwrap();

                    let str_name = inner_rules.next().unwrap().as_str();
                    name_map.insert(index, str_name);
                }
            }
            Rule::ports_entry => {
                // println!("match a ports_entry now.");
                if current_section_name == "*PORTS" {
                    port_count = port_count + 1;
                    let mut inner_rules = line.into_inner();

                    let index_name_pair = inner_rules.next();
                    let index_pair = index_name_pair.unwrap().into_inner().next();
                    let index = index_pair.unwrap().as_str().parse::<i64>().unwrap();

                    let direction_span = inner_rules.next().unwrap().as_str();
                    let direction: Direction;
                    match direction_span {
                        "I" => direction = Direction::I,
                        "O" => direction = Direction::O,
                        "B" => direction = Direction::B,
                        &_ => todo!(),
                    }

                    let mut xy_coordinates_span = inner_rules.next().unwrap().into_inner();
                    let x_coordinate = xy_coordinates_span
                        .next()
                        .unwrap()
                        .as_str()
                        .parse::<f64>()
                        .unwrap();
                    let y_coordinate = xy_coordinates_span
                        .next()
                        .unwrap()
                        .as_str()
                        .parse::<f64>()
                        .unwrap();

                    ports.push((
                        Name::IndexName(index),
                        direction,
                        (x_coordinate, y_coordinate),
                    ));
                    // println!("{x_coordinate:#?}, {y_coordinate:#?}");
                }
            }
            Rule::dnet_entry => {
                // println!("{line:#?}");
                dnet_count = dnet_count + 1;
                let mut inner_rules = line.into_inner();
                let dnet_index_name = inner_rules.next().unwrap().as_str();
                let dnet_index = dnet_index_name[1..].parse::<i64>().unwrap();
                let dnet_cap = inner_rules.next().unwrap().as_str().parse::<f64>().unwrap();

                // println!("{dnet_name:#?}");
                current_dnet.name = Name::IndexName(dnet_index);
                current_dnet.total_cap = dnet_cap;
            }
            Rule::conn_entry => {
                // println!("{line:#?}");
                let mut current_conn_entry = ConnEntry {
                    conn_type: ConnType::Uninitialized,
                    name: Name::IndexName(0),
                    direction: Direction::Uninitialized,
                    coordinates: (-1.0, -1.0),
                    load_val: 0.0,
                    driver: Name::StrName("".to_string()),
                };

                let mut inner_rules = line.into_inner();
                let conn_type_span = inner_rules.next().unwrap().as_str();
                let conn_type: ConnType;
                match conn_type_span {
                    "*P" => conn_type = ConnType::P,
                    "*I" => conn_type = ConnType::I,
                    "*S" => conn_type = ConnType::S,
                    "*C" => conn_type = ConnType::C,
                    "*R" => conn_type = ConnType::R,
                    "*L" => conn_type = ConnType::L,
                    &_ => todo!(),
                }
                current_conn_entry.conn_type = conn_type;

                let pin_port = inner_rules.next().unwrap().as_str();
                current_conn_entry.name = Name::PinPort(pin_port.to_string());

                let direction_span = inner_rules.next().unwrap().as_str();
                current_conn_entry.direction = match direction_span {
                    "I" => Direction::I,
                    "O" => Direction::O,
                    "B" => Direction::B,
                    &_ => Direction::X,
                };

                let mut xy_coordinates_span = inner_rules.next().unwrap().into_inner();
                current_conn_entry.coordinates.0 = xy_coordinates_span
                    .next()
                    .unwrap()
                    .as_str()
                    .parse::<f64>()
                    .unwrap();
                current_conn_entry.coordinates.1 = xy_coordinates_span
                    .next()
                    .unwrap()
                    .as_str()
                    .parse::<f64>()
                    .unwrap();

                // Optional params
                let optional_load_val = inner_rules.next();
                match optional_load_val {
                    Some(load_val_pair) => {
                        current_conn_entry.load_val =
                            load_val_pair.as_str().parse::<f64>().unwrap();
                    }
                    None => {}
                }

                let optional_driver = inner_rules.next();
                match optional_driver {
                    Some(driver_pair) => {
                        current_conn_entry.driver = Name::StrName(driver_pair.as_str().to_string());
                    }
                    None => {}
                }

                current_dnet.conn_section.push(current_conn_entry);
            }
            Rule::cap_entry => {
                // println!("{line:#?}");
                let mut inner_rules = line.into_inner();
                let index = inner_rules.next().unwrap().as_str().parse::<i64>().unwrap();
                let pin_port = inner_rules.next().unwrap().as_str();
                let cap_val = inner_rules.next().unwrap().as_str().parse::<f64>().unwrap();

                current_dnet.cap_section.push((
                    index,
                    Name::PinPort(pin_port.to_string()),
                    cap_val,
                ));
            }
            Rule::res_entry => {
                // println!("{line:#?}");
                let mut inner_rules = line.into_inner();
                let index = inner_rules.next().unwrap().as_str().parse::<i64>().unwrap();
                let pin_port_start = inner_rules.next().unwrap().as_str();
                let pin_port_end = inner_rules.next().unwrap().as_str();
                let res_val = inner_rules.next().unwrap().as_str().parse::<f64>().unwrap();

                current_dnet.res_section.push((
                    index,
                    Name::PinPort(pin_port_start.to_string()),
                    Name::PinPort(pin_port_end.to_string()),
                    res_val,
                ));
            }
            Rule::EOI => (),
            _ => unreachable!(),
        }
    }

    let spef_data = SpefData {
        header,
        namemap: name_map,
        ports,
        dnets,
    };

    println!("Spef parsing completed, {header_count} header entries. {name_map_count} name-map pairs. {port_count} ports specified. {dnet_count} dnets parsed.");
}

fn parse_str_pair<'a> ()-> &'a str {
    let default_value = "";

    default_value
}
