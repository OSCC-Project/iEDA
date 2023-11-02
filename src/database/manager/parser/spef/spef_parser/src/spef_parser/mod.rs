pub mod spef_data;

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;
use spef_data::SpefExchange;
use std::fs;
use std::fmt::Debug;

use self::{
    ffi::{CapItem, ConnItem},
    spef_data::SpefValue,
};

#[cxx::bridge(namespace = "ista::spef")]
mod ffi {
    #[derive(Clone, Debug)]
    enum ConnectionType {
        INTERNAL,
        EXTERNAL,
        UNITIALIZED,
    }

    #[derive(Clone, Debug)]
    enum ConnectionDirection {
        INPUT,
        OUTPUT,
        INOUT,
        UNITIALIZED,
    }

    #[derive(Clone, Debug)]
    struct HeaderItem {
        key: String,
        value: String,
    }

    #[derive(Clone, Debug)]
    struct NameMapItem {
        index: usize,
        name: String,
    }

    #[derive(Clone, Debug)]
    struct PortItem {
        name: String,
        direction: ConnectionDirection,
        coordinates: [f64; 2],
    }

    #[derive(Clone, Debug)]
    struct ConnItem {
        conn_type: ConnectionType,
        conn_direction: ConnectionDirection,
        pin_name: String,
        driving_cell: String,
        load: f64,
        layer: usize,
        coordinates: [f64; 2],
        ll_coordinate: [f64; 2],
        ur_coordinate: [f64; 2],
    }

    #[derive(Clone, Debug)]
    struct CapItem {
        pin_port: [String; 2],
        cap_val: f64,
    }

    #[derive(Clone, Debug)]
    struct ResItem {
        pin_port: [String; 2],
        res: f64,
    }

    #[derive(Clone, Debug)]
    struct NetItem {
        name: String,
        lcap: f64,
        conns: Vec<ConnItem>,
        caps: Vec<CapItem>,
        ress: Vec<ResItem>,
    }
    // Shared structure between cpp and rust, it uses the types in the `exter "Rust" section`
    #[derive(Clone, Debug)]
    struct SpefFile {
        name: String,
        header: Vec<HeaderItem>,
        name_vector: Vec<NameMapItem>,
        ports: Vec<PortItem>,
        nets: Vec<NetItem>,
    }
    // functions exposed from rust to cpp
    extern "Rust" {
        fn parse_spef_file(path: &str) -> Result<SpefFile>;
    }
}

// fn test_f() -> Box<SpefEntryBasicInfo> {
//     return SpefEntryBasicInfo::new("tbd", 1);
// }

#[derive(Parser)]
#[grammar = "spef_parser/grammar/spef.pest"]
struct SpefParser;

/// process float pairs, returing a f64 or an Err.
fn process_float(pair: Pair<Rule>) -> Result<f64, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();

    // Index name pairs are parsed as f64 too, remove the preceding "*" before the index.
    // If not index name pairs, this operation doesn't make sense to them.
    let pair_str = pair.as_str();
    let clearned_str: String = pair_str.chars().filter(|&c| c != '*').collect();

    match clearned_str.parse::<f64>() {
        Ok(value) => Ok(value),
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse float".into() },
            pair_clone.as_span(),
        )),
    }
}

/// process xy coordinates, returning a (f64, f64) or an Err
fn process_coordinates(pair: Pair<Rule>) -> Result<(f64, f64), pest::error::Error<Rule>> {
    let pair_clone = pair.clone();

    let tuple_pair = pair.clone();

    let mut inner_rules = pair_clone.into_inner();
    let x_coordiante_pair = inner_rules.next();
    let y_coordinate_pair = inner_rules.next();

    match (x_coordiante_pair, y_coordinate_pair) {
        (Some(x_float_pair), Some(y_float_pair)) => {
            let x_float = process_float(x_float_pair);
            let y_float = process_float(y_float_pair);
            match (x_float, y_float) {
                (Ok(x), Ok(y)) => Ok((x, y)),
                _ => Err(pest::error::Error::new_from_span(
                    pest::error::ErrorVariant::CustomError { message: "Failed to parse float".into() },
                    tuple_pair.as_span(),
                )),
            }
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse xy_coordinates".into() },
            tuple_pair.as_span(),
        )),
    }
}

/// process string text data not include quote(All string values in spef file are not quoted).
fn process_string(pair: Pair<Rule>) -> Result<String, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair_clone.as_str().parse::<String>() {
        Ok(value) => Ok(value),
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse string".into() },
            pair_clone.as_span(),
        )),
    }
}

/// process connection direction enum
fn process_conn_dir_enum(pair: Pair<Rule>) -> Result<spef_data::ConnectionDirection, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair.as_str() {
        "I" => Ok(spef_data::ConnectionDirection::INPUT),
        "O" => Ok(spef_data::ConnectionDirection::OUTPUT),
        "B" => Ok(spef_data::ConnectionDirection::INOUT),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse connection direction".into() },
            pair_clone.as_span(),
        )),
    }
}

/// process connection type enum
fn process_conn_type_enum(pair: Pair<Rule>) -> Result<spef_data::ConnectionType, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair.as_str() {
        "*I" => Ok(spef_data::ConnectionType::INTERNAL),
        "*P" => Ok(spef_data::ConnectionType::EXTERNAL),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse connection type".into() },
            pair_clone.as_span(),
        )),
    }
}

/// process section entry
fn process_section_entry(pair: Pair<Rule>) -> Result<spef_data::SpefSectionEntry, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    let line_no = pair.line_col().0;

    let mut inner_rules = pair_clone.into_inner();

    // println!("{pair:#?}");
    let section_name_pair = inner_rules.next().unwrap();

    let section_name_result = process_string(section_name_pair);

    match section_name_result {
        Ok(result) => {
            let section_type: spef_data::SectionType = match result {
                s if s == "NAME_MAP" => spef_data::SectionType::NAMEMAP,
                s if s == "PORTS" => spef_data::SectionType::PORTS,
                s if s == "CONN" => spef_data::SectionType::CONN,
                s if s == "CAP" => spef_data::SectionType::CAP,
                s if s == "RES" => spef_data::SectionType::RES,
                s if s == "END" => spef_data::SectionType::END,
                _ => {
                    // 处理未知规则的情况
                    return Err(pest::error::Error::new_from_span(
                        pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                        pair.as_span(),
                    ));
                }
            };
            Ok(spef_data::SpefSectionEntry::new("tbd", line_no, section_type))
        }
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}

/// process pest pairs that matches spef header section entry
fn process_header_entry(pair: Pair<Rule>) -> Result<spef_data::SpefHeaderEntry, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    let line_no = pair_clone.line_col().0;

    let mut inner_rules = pair_clone.into_inner();
    // println!("{inner_rules:#?}");

    // header_keyword_pair and header_value_pair are string pairs
    let header_keyword_pair = inner_rules.next().unwrap();
    let header_value_pair = inner_rules.next().unwrap();

    let keyword_pair_result = process_string(header_keyword_pair);
    let value_pair_result = process_string(header_value_pair);

    match (keyword_pair_result, value_pair_result) {
        (Ok(header_key), Ok(header_value)) => {
            Ok(spef_data::SpefHeaderEntry::new("tbd", line_no, header_key, header_value))
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}

/// process pest pairs that matches spef namemap section entry
fn process_namemap_entry(pair: Pair<Rule>) -> Result<spef_data::SpefNameMapEntry, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    let line_no = pair_clone.line_col().0;

    let mut inner_rules = pair_clone.into_inner();
    // println!("{inner_rules:#?}");

    // name_index_pair is float pair, name_value_pair is string pair
    let name_index_pair = inner_rules.next().unwrap();
    let name_value_pair = inner_rules.next().unwrap();

    let index_pair_result = process_float(name_index_pair);
    let value_pair_result = process_string(name_value_pair);

    match (index_pair_result, value_pair_result) {
        (Ok(name_index), Ok(name_pair)) => {
            Ok(spef_data::SpefNameMapEntry::new("tbd", line_no, name_index as usize, &name_pair))
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}

/// process pest pairs that matches spef ports section entry
fn process_port_entry(pair: Pair<Rule>) -> Result<spef_data::SpefPortEntry, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    let line_no = pair_clone.line_col().0;

    let mut inner_rules = pair_clone.into_inner();
    // println!("{inner_rules:#?}");

    // name_index_pair is float pair, name_value_pair is string pair
    let name_index_pair = inner_rules.next().unwrap();
    let conn_dir_pair = inner_rules.next().unwrap();
    let coordinates_pair = inner_rules.next().unwrap();

    let index_pair_result = process_float(name_index_pair);
    let dir_pair_result = process_conn_dir_enum(conn_dir_pair);
    let coor_pair_result = process_coordinates(coordinates_pair);

    match (index_pair_result, dir_pair_result, coor_pair_result) {
        (Ok(index), Ok(direction), Ok(coordinates)) => {
            Ok(spef_data::SpefPortEntry::new("tbd", line_no, index.to_string(), direction, coordinates))
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}

/// process pest pairs that matches spef dnet section entry, creating a SpefNet
fn process_dnet_entry<'a>(
    pair: Pair<'a, Rule>,
    current_net: &'a mut spef_data::SpefNet,
) -> Result<spef_data::SpefNet, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    let line_no = pair_clone.line_col().0;

    let mut inner_rules = pair_clone.into_inner();

    let name_pair = inner_rules.next().unwrap();
    let cap_pair = inner_rules.next().unwrap();

    let name_pair_result = process_string(name_pair);
    let cap_pair_result = process_float(cap_pair);

    match (name_pair_result, cap_pair_result) {
        (Ok(name), Ok(cap)) => {
            current_net.name = name;
            current_net.line_no = line_no;
            current_net.lcap = cap;
            Ok(current_net.clone())
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}

fn process_conn_entry(pair: Pair<Rule>) -> Result<spef_data::SpefConnEntry, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    let line_no = pair_clone.line_col().0;

    let mut inner_rules = pair_clone.into_inner();

    let conn_type_pair = inner_rules.next().unwrap();
    let type_pair_result = process_conn_type_enum(conn_type_pair);

    let pin_name_pair = inner_rules.next().unwrap();
    let name_pair_result = process_string(pin_name_pair);

    let conn_dir_pair = inner_rules.next().unwrap();
    let dir_pair_result = process_conn_dir_enum(conn_dir_pair);

    // xy_coordinates may be None
    let coordinates_pair = inner_rules.next();

    let xy_coordinates = match coordinates_pair {
        Some(pair) => {
            let coor_pair_result = process_coordinates(pair);
            match coor_pair_result {
                Ok(coors) => coors,
                Err(_) => (0.0, 0.0),
            }
        }
        None => (0.0, 0.0),
    };

    // load may be None
    let load_pair = inner_rules.next();

    let load = match load_pair {
        Some(pair) => {
            let load_pair_result = process_float(pair);
            match load_pair_result {
                Ok(load) => load,
                Err(_) => 0.0,
            }
        }
        None => 0.0,
    };

    // driver_pair may be None
    let driver_pair = inner_rules.next();

    let driver_name: String = match driver_pair {
        Some(pair) => {
            let driver_pair_result = process_string(pair);
            match driver_pair_result {
                Ok(driver_cell) => driver_cell,
                Err(_) => String::new(),
            }
        }
        None => String::new(),
    };

    match (type_pair_result, name_pair_result, dir_pair_result) {
        (Ok(conn_type), Ok(pin_name), Ok(conn_dir)) => {
            let mut current_conn = spef_data::SpefConnEntry::new("tbd", line_no, conn_type, conn_dir, pin_name);
            current_conn.set_load(load);
            current_conn.set_xy_coordinates(xy_coordinates);
            current_conn.set_driving_cell(driver_name);
            Ok(current_conn)
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}

// process cap or res entry into a (String, String, f64)
fn process_cap_or_res_entry(
    pair: Pair<Rule>,
    cap_or_res: spef_data::SectionType,
) -> Result<(String, String, f64), pest::error::Error<Rule>> {
    let pair_clone = pair.clone();

    let mut inner_rules = pair_clone.into_inner();

    // println!("{inner_rules:#?}");
    match cap_or_res {
        spef_data::SectionType::CAP => {
            // let cap_entry: (String, String, f64);

            // CAP entry has at least two pair(a start pin and a cap val)
            // Capacitor can be ground (one node) or coupled (two nodes)
            let start_pin_pair = inner_rules.next().unwrap();
            let start_pin_result = process_string(start_pin_pair);

            // The second pair can be a pin_port or a cap_or_res_value
            let next_pair = inner_rules.next().unwrap();

            // println!("{next_pair:#?}");
            if next_pair.as_rule() == Rule::cap_or_res_val {
                // ground cap
                let cap_val_result = process_float(next_pair);
                match (start_pin_result, cap_val_result) {
                    (Ok(start_pin_name), Ok(cap_result)) => Ok((start_pin_name, "GROUND".to_string(), cap_result)),
                    _ => Err(pest::error::Error::new_from_span(
                        pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                        pair.as_span(),
                    )),
                }
            } else {
                // couple cap
                let end_pin_result = process_string(next_pair);
                let cap_val_pair = inner_rules.next().unwrap();
                let cap_val_result = process_float(cap_val_pair);
                match (start_pin_result, end_pin_result, cap_val_result) {
                    (Ok(start_pin_name), Ok(end_pin_name), Ok(cap_val)) => Ok((start_pin_name, end_pin_name, cap_val)),
                    _ => Err(pest::error::Error::new_from_span(
                        pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                        pair.as_span(),
                    )),
                }
            }
        }
        spef_data::SectionType::RES => {
            // let res_entry: (String, String, f64);

            // RES entry has three pairs: two nodes and a res val
            let start_pin_pair = inner_rules.next().unwrap();
            let end_pin_pair = inner_rules.next().unwrap();
            let res_val_pair = inner_rules.next().unwrap();

            let start_pin_result = process_string(start_pin_pair);
            let end_pin_result = process_string(end_pin_pair);
            let res_val_result = process_float(res_val_pair);

            match (start_pin_result, end_pin_result, res_val_result) {
                (Ok(start_pin_name), Ok(end_pin_name), Ok(res_val)) => Ok((start_pin_name, end_pin_name, res_val)),
                _ => Err(pest::error::Error::new_from_span(
                    pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                    pair.as_span(),
                )),
            }
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}

pub fn parse_spef_file(spef_file_path: &str) -> anyhow::Result<ffi::SpefFile> {
    // Add '?' to the end of a statement will possiably return an error,
    // This is used to replace the .expect()
    // !TODO Use crate anyhow to handle all the errors, this will reduce the code amount.
    let unparsed_file = fs::read_to_string(spef_file_path)?;
    let spef_entries = SpefParser::parse(Rule::file, &unparsed_file)?;

    let mut exchange_data =
        spef_data::SpefExchange::new(spef_data::SpefStringValue { value: spef_file_path.to_string() });

    let mut current_net: spef_data::SpefNet = spef_data::SpefNet::new(0, "None".to_string(), 0.0);
    let mut current_section: spef_data::SectionType = spef_data::SectionType::HEADER;

    // let mut entry_parse_result = Ok(spef_data::SpefParserData::Unitialized);
    let mut _entry_parse_result;

    for entry in spef_entries {
        _entry_parse_result = match entry.as_rule() {
            Rule::section => {
                // Section entries are not included in SpefParserData, it is used as a label for this function.
                let parse_result = process_section_entry(entry.clone());
                match parse_result {
                    Ok(result) => {
                        current_section = result.get_section_type().clone();
                        match current_section {
                            spef_data::SectionType::END => {
                                exchange_data.add_net(current_net.clone());
                                current_net = spef_data::SpefNet::new(0, "None".to_string(), 0.0);
                            }
                            _ => (),
                        };
                        Ok(spef_data::SpefParserData::SectionEntry(result))
                    }
                    Err(err) => Err(err.clone()),
                }
            }
            Rule::header_entry => {
                let parse_result = process_header_entry(entry.clone());
                match parse_result {
                    Ok(result) => {
                        exchange_data.add_header_entry(result.clone());
                        Ok(spef_data::SpefParserData::HeaderEntry(result))
                    }
                    Err(err) => Err(err.clone()),
                }
            }
            Rule::name_map_entry => {
                let parse_result = process_namemap_entry(entry.clone());
                match parse_result {
                    Ok(result) => {
                        exchange_data.add_namemap_entry(result.clone());
                        Ok(spef_data::SpefParserData::NameMapEntry(result))
                    }
                    Err(err) => Err(err.clone()),
                }
            }
            Rule::ports_entry => {
                let parse_result = process_port_entry(entry.clone());
                match parse_result {
                    Ok(result) => {
                        exchange_data.add_port_entry(result.clone());
                        Ok(spef_data::SpefParserData::PortEntry(result))
                    }
                    Err(err) => Err(err.clone()),
                }
            }
            Rule::dnet_entry => {
                // Config the current_net to record the net staring here.
                // This part doesn't return anything, it edits the current net members.
                let parse_result = process_dnet_entry(entry, &mut current_net);
                match parse_result {
                    Ok(result) => Ok(spef_data::SpefParserData::NetEntry(result)),
                    Err(err) => Err(err.clone()),
                }
            }
            Rule::conn_entry => {
                // Parse the connection entry and add it to the current_net.
                // This part doesn't return anything, it adds connection to current net.
                let parse_result = process_conn_entry(entry);
                match parse_result {
                    Ok(result) => {
                        current_net.add_connection(&result);
                        Ok(spef_data::SpefParserData::ConnEntry(result))
                    }
                    Err(err) => Err(err.clone()),
                }
            }
            Rule::cap_or_res_entry => {
                // Parse the cap or res entry and add it to the current_net according to the current_section
                // This part doesn't return anything, it adds caps or ress to current net.
                let parse_result = process_cap_or_res_entry(entry, current_section.clone());
                match parse_result {
                    Ok(result) => {
                        match current_section {
                            spef_data::SectionType::CAP => {
                                current_net.add_cap(result);
                            }
                            spef_data::SectionType::RES => {
                                current_net.add_res(result);
                            }
                            _ => {}
                        };
                        Ok(spef_data::SpefParserData::NetEntry(current_net.clone()))
                    }
                    Err(err) => Err(err.clone()),
                }
            }
            Rule::EOI => Ok(spef_data::SpefParserData::Exchange(exchange_data.clone())),
            _ => Err(pest::error::Error::new_from_span(
                pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                entry.as_span(),
            )),
        }?;
    };

    // Here we got a SpefExchange, it includes everthing about the spef file
    // But we need a SpefFile to transfer data to cpp, it also includes everything about the spef file, but with ffi types.
    Ok(convert_to_file(exchange_data))
}

/// Convert SpefExchange to SpefFile
/// SpefExchange is compound with refined rust types: SpefStringValue, SpefHeaderEntry, SpefNameMapEntry, SpefPortEntry, SpefNet, etc
/// SpefFile is compound with restriced rust types: [T, N], &str, f64, usize, enum, Vec<T>, etc
fn convert_to_file(source: SpefExchange) -> ffi::SpefFile {
    let source_clone = source.clone();
    let mut res =
        ffi::SpefFile { name: String::from(""), header: Vec::new(), name_vector: Vec::new(), ports: Vec::new(), nets: Vec::new() };

    let name = source_clone.get_file_name().get_str_value();
    res.name = name;

    for header_entry in source_clone.header {
        // let header_entry_clone = header_entry.clone();
        let key = header_entry.get_header_key();
        let value = header_entry.get_header_value();
        res.header.push(ffi::HeaderItem { key, value });
    }
    for name_map_entry in source_clone.namemap {
        // let namemap_entry_clone = name_map_entry.clone();
        let index = name_map_entry.get_index();
        let name = name_map_entry.get_name();
        res.name_vector.push(ffi::NameMapItem { index, name });
    }
    for port_entry in source.ports {
        let pin_name = port_entry.get_name();

        // spef_data has a ConnectionDirection, it is used in parse fucntions.
        // ffi has a ConnectionDirection, it is passed to cpp.
        let direction = match port_entry.get_direction() {
            spef_data::ConnectionDirection::INOUT => ffi::ConnectionDirection::INOUT,
            spef_data::ConnectionDirection::OUTPUT => ffi::ConnectionDirection::OUTPUT,
            spef_data::ConnectionDirection::INPUT => ffi::ConnectionDirection::INPUT,
            spef_data::ConnectionDirection::UNITIALIZED => ffi::ConnectionDirection::UNITIALIZED,
        };

        let coordinates: [f64; 2];
        let (x, y) = port_entry.get_coordinates();
        coordinates = [x, y];

        res.ports.push(ffi::PortItem { name: pin_name, direction, coordinates });
    }
    for net in source.nets {
        let net_name = net.get_net_name();
        let lcap = net.get_lcap();

        // conns_spef is the vector of SpefConnEntry
        // conns is the vector of ConnItem, it will be pushed into the NetItem.
        let conns_spef = net.get_conns();
        let mut conns = Vec::<ConnItem>::new();
        for conn in conns_spef {
            let pin_name = conn.get_name();
            let driving_cell = conn.get_driving_cell();
            let load = conn.get_load();
            let layer = conn.get_layer();

            let conn_type = match conn.get_conn_type() {
                spef_data::ConnectionType::INTERNAL => ffi::ConnectionType::INTERNAL,
                spef_data::ConnectionType::EXTERNAL => ffi::ConnectionType::EXTERNAL,
                spef_data::ConnectionType::UNITIALIZED => ffi::ConnectionType::UNITIALIZED,
            };

            let conn_direction = match conn.get_conn_direction() {
                spef_data::ConnectionDirection::INOUT => ffi::ConnectionDirection::INOUT,
                spef_data::ConnectionDirection::OUTPUT => ffi::ConnectionDirection::OUTPUT,
                spef_data::ConnectionDirection::INPUT => ffi::ConnectionDirection::INPUT,
                spef_data::ConnectionDirection::UNITIALIZED => ffi::ConnectionDirection::UNITIALIZED,
            };

            let xy_coordinates: [f64; 2];
            let (x, y) = conn.get_xy_coordinates();
            xy_coordinates = [x, y];

            let ll_coordinate: [f64; 2];
            let (ll_x, ll_y) = conn.get_ll_coordinates();
            ll_coordinate = [ll_x, ll_y];

            let ur_coordinate: [f64; 2];
            let (ur_x, ur_y) = conn.get_ur_coordinates();
            ur_coordinate = [ur_x, ur_y];

            conns.push(ffi::ConnItem {
                conn_type,
                conn_direction,
                pin_name,
                driving_cell,
                load,
                layer,
                coordinates: xy_coordinates,
                ll_coordinate,
                ur_coordinate,
            });
        }

        // caps_spef is the vector of (String, String, f64)
        // caps is the vector of CapItem, it will be pushed into the NetItem
        let caps_spef = net.get_caps();
        let mut caps = Vec::<CapItem>::new();
        for cap in caps_spef {
            let (start_pin, end_pin, cap_value) = cap;
            caps.push(ffi::CapItem { pin_port: [start_pin, end_pin], cap_val: cap_value });
        }

        // ress_spef is the vector of (String, String, f64)
        // ress is the vector of ResItem, it will be pushed into the NetItem
        let ress_spef = net.get_ress();
        let mut ress = Vec::<ffi::ResItem>::new();
        for res_entry in ress_spef {
            let (start_pin, end_pin, res_value) = res_entry;
            ress.push(ffi::ResItem { pin_port: [start_pin, end_pin], res: res_value });
        }
        
        res.nets.push(ffi::NetItem{name: net_name, lcap, conns, caps, ress});
    }
    
    res
}
