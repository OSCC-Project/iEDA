pub mod spef_c_api;
pub mod spef_data;

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use std::fmt::Debug;
use std::fs;

use std::time::Instant;

#[derive(Parser)]
#[grammar = "spef_parser/grammar/spef.pest"]
struct SpefParser;

fn measure_elapsed_time(start_time: Instant) -> f64 {
    let end_time = Instant::now();
    let elapsed_time = end_time.duration_since(start_time);
    
    elapsed_time.as_secs_f64()
}

/// process float pairs, returing a f64 or an Err.
fn process_float(pair: Pair<Rule>) -> Result<f64, pest::error::Error<Rule>> {
    let pair_span = pair.as_span();

    // Index name pairs are parsed as f64 too, remove the preceding "*" before the index.
    // If not index name pairs, this operation doesn't make sense to them.
    let pair_str = pair.as_str();
    let cleaned_str: String = pair_str.chars().filter(|&c| c != '*').collect();

    match cleaned_str.parse::<f64>() {
        Ok(value) => Ok(value),
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse float".into() },
            pair_span,
        )),
    }
}

/// process xy coordinate, returning a (f64, f64) or an Err
fn process_coordinate(pair: Pair<Rule>) -> Result<(f64, f64), pest::error::Error<Rule>> {
    let pair_span = pair.as_span();

    let mut inner_rules = pair.into_inner();
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
                    pair_span,
                )),
            }
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse xy_coordinate".into() },
            pair_span,
        )),
    }
}

/// process string text data not include quote(All string values in spef file are not quoted).
fn process_string(pair: Pair<Rule>) -> Result<String, pest::error::Error<Rule>> {
    let pair_span = pair.as_span();
    match pair.as_str().parse::<String>() {
        Ok(value) => Ok(value),
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse string".into() },
            pair_span,
        )),
    }
}

/// process connection direction enum
fn process_conn_dir_enum(pair: Pair<Rule>) -> Result<spef_data::ConnectionDirection, pest::error::Error<Rule>> {
    let pair_span = pair.as_span();
    match pair.as_str() {
        "I" => Ok(spef_data::ConnectionDirection::INPUT),
        "O" => Ok(spef_data::ConnectionDirection::OUTPUT),
        "B" => Ok(spef_data::ConnectionDirection::INOUT),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse connection direction".into() },
            pair_span,
        )),
    }
}

/// process connection type enum
fn process_conn_type_enum(pair: Pair<Rule>) -> Result<spef_data::ConnectionType, pest::error::Error<Rule>> {
    let pair_span = pair.as_span();
    match pair.as_str() {
        "*I" => Ok(spef_data::ConnectionType::INTERNAL),
        "*P" => Ok(spef_data::ConnectionType::EXTERNAL),
        "*N" => Ok(spef_data::ConnectionType::INTERNAL),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse connection type".into() },
            pair_span,
        )),
    }
}

/// process section entry
fn process_section_entry(pair: Pair<Rule>) -> Result<spef_data::SpefSectionEntry, pest::error::Error<Rule>> {
    let pair_span = pair.as_span();
    let line_no = pair.line_col().0;

    let mut inner_rules = pair.into_inner();

    // println!("{pair:#?}");
    let section_name_pair = inner_rules.next().unwrap();

    let section_name_result = process_string(section_name_pair);

    match section_name_result {
        Ok(result) => {
            let section_type: spef_data::SectionType = match result.as_str() {
                "NAME_MAP" => spef_data::SectionType::NAMEMAP,
                "PORTS" => spef_data::SectionType::PORTS,
                "CONN" => spef_data::SectionType::CONN,
                "CAP" => spef_data::SectionType::CAP,
                "RES" => spef_data::SectionType::RES,
                "END" => spef_data::SectionType::END,
                _ => {
                    return Err(pest::error::Error::new_from_span(
                        pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                        pair_span,
                    ));
                }
            };
            Ok(spef_data::SpefSectionEntry::new("tbd", line_no, section_type))
        }
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_span,
        )),
    }
}

/// process pest pairs that matches spef header section entry
fn process_header_entry(pair: Pair<Rule>) -> Result<spef_data::SpefHeaderEntry, pest::error::Error<Rule>> {
    let pair_span = pair.as_span();
    let line_no = pair.line_col().0;

    let mut inner_rules = pair.into_inner();
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
            pair_span,
        )),
    }
}

/// process pest pairs that matches spef namemap section entry
fn process_namemap_entry(pair: Pair<Rule>) -> Result<spef_data::SpefNameMapEntry, pest::error::Error<Rule>> {
    let pair_span = pair.as_span();
    let line_no = pair.line_col().0;

    let mut inner_rules = pair.into_inner();
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
            pair_span,
        )),
    }
}

/// process pest pairs that matches spef ports section entry
fn process_port_entry(pair: Pair<Rule>) -> Result<spef_data::SpefPortEntry, pest::error::Error<Rule>> {
    let pair_span = pair.as_span();
    let line_no = pair.line_col().0;

    let mut inner_rules = pair.into_inner();
    // println!("{inner_rules:#?}");

    // name_index_pair is float pair, name_value_pair is string pair
    let name_index_pair = inner_rules.next().unwrap();
    let conn_dir_pair = inner_rules.next().unwrap();
    let coordinate_pair = inner_rules.next();

    let index_pair_result = process_float(name_index_pair);
    let dir_pair_result = process_conn_dir_enum(conn_dir_pair);

    let mut coor_pair_result: Result<(f64, f64), pest::error::Error<Rule>> = Ok((-1.0, -1.0));
    if coordinate_pair.is_some() {
        coor_pair_result = process_coordinate(coordinate_pair.unwrap());
    }

    match (index_pair_result, dir_pair_result, coor_pair_result) {
        (Ok(index), Ok(direction), Ok(coordinate)) => {
            Ok(spef_data::SpefPortEntry::new("tbd", line_no, index.to_string(), direction, coordinate))
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_span,
        )),
    }
}

/// process pest pairs that matches spef dnet section entry, creating a SpefNet
fn process_dnet_entry<'a>(
    pair: Pair<'a, Rule>,
    current_net: &'a mut spef_data::SpefNet,
) -> Result<spef_data::SpefNet, pest::error::Error<Rule>> {
    let pair_span = pair.as_span();
    let line_no = pair.line_col().0;

    let mut inner_rules = pair.into_inner();

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
            pair_span,
        )),
    }
}

fn process_conn_entry(pair: Pair<Rule>) -> Result<spef_data::SpefConnEntry, pest::error::Error<Rule>> {
    let line_no = pair.line_col().0;

    let mut current_conn = spef_data::SpefConnEntry::new("tbd", line_no);

    let inner_rules = pair.into_inner();
    for inner_pair in inner_rules {
        match inner_pair.as_rule() {
            Rule::conn_type => {
                let conn_type = process_conn_type_enum(inner_pair).unwrap();
                current_conn.set_conn_type(conn_type);
            }
            Rule::pin_port => {
                let pin_port_name = process_string(inner_pair).unwrap();
                current_conn.set_pin_port_name(pin_port_name);
            }
            Rule::direction => {
                let conn_direction: spef_data::ConnectionDirection;
                if inner_pair.as_str() != "*N" {
                    conn_direction = process_conn_dir_enum(inner_pair).unwrap();
                } else {
                    conn_direction = spef_data::ConnectionDirection::Internal;
                }
                current_conn.set_conn_direction(conn_direction);
            }
            Rule::xy_coordinates => {
                let xy_coordinate = {
                    let coor_pair_result = process_coordinate(inner_pair);
                    coor_pair_result.unwrap_or((-1.0, -1.0))
                };
                current_conn.set_xy_coordinate(xy_coordinate);
            }
            Rule::cap_or_res_val => {
                let load = {
                    let load_pair_result = process_float(inner_pair);
                    load_pair_result.unwrap_or(0.0)
                };

                current_conn.set_load(load);
            }
            Rule::str_name => {
                let driver_cell = process_string(inner_pair).unwrap();
                current_conn.set_driving_cell(driver_cell);
            }

            _ => panic!("unknown rule."),
        }
    }

    Ok(current_conn)
}

// process cap or res entry into a (String, String, f64)
fn process_cap_or_res_entry(
    pair: Pair<Rule>,
    cap_or_res: &spef_data::SectionType,
) -> Result<(String, String, f64), pest::error::Error<Rule>> {
    let pair_span: pest::Span<'_> = pair.as_span();

    let mut inner_rules = pair.into_inner();

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
                    (Ok(start_pin_name), Ok(cap_result)) => Ok((start_pin_name, "".to_string(), cap_result)),
                    _ => Err(pest::error::Error::new_from_span(
                        pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                        pair_span,
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
                        pair_span,
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
                    pair_span,
                )),
            }
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_span,
        )),
    }
}

pub fn parse_spef_file(spef_file_path: &str) -> spef_data::SpefExchange {
    let start_time = Instant::now();

    let unparsed_file = fs::read_to_string(spef_file_path).unwrap();
    let spef_entries = SpefParser::parse(Rule::spef_file, &unparsed_file).unwrap();

    let mut exchange_data = spef_data::SpefExchange::new(spef_file_path.to_string());

    let mut current_net: spef_data::SpefNet = spef_data::SpefNet::new(0, "None".to_string(), 0.0);
    let mut current_section: spef_data::SectionType = spef_data::SectionType::HEADER;

    let spef_file_pair = spef_entries.into_iter().next().unwrap();

    for entry in spef_file_pair.into_inner() {
        // let entry_clone = entry.clone();
        // println!("Rule:    {:?}", entry_clone.as_rule());
        // println!("Span:    {:?}", entry_clone.as_span());
        // println!("Text:    {}", entry_clone.as_str());

        // let line_no = entry.line_col().0;
        // println!("line no {}", line_no);

        match entry.as_rule() {
            Rule::section => {
                // Section entries are not included in SpefParserData, it is used as a label for this function.
                let parse_result = process_section_entry(entry);
                match parse_result {
                    Ok(result) => {
                        current_section = result.get_section_type().clone();
                        match current_section {
                            spef_data::SectionType::END => {
                                exchange_data.add_net(current_net);
                                current_net = spef_data::SpefNet::new(0, "None".to_string(), 0.0);
                            }
                            _ => (),
                        };
                    }
                    Err(_) => panic!("process failed"),
                }
            }
            Rule::header_entry => {
                let parse_result = process_header_entry(entry);
                match parse_result {
                    Ok(result) => {
                        exchange_data.add_header_entry(result);
                    }
                    Err(_) => panic!("process failed"),
                }
            }
            Rule::name_map_entry => {
                let parse_result = process_namemap_entry(entry);
                match parse_result {
                    Ok(result) => {
                        exchange_data.add_name_map_entry(result);
                    }
                    Err(_) => panic!("process failed"),
                }
            }
            Rule::ports_entry => {
                let parse_result = process_port_entry(entry);
                match parse_result {
                    Ok(result) => {
                        exchange_data.add_port_entry(result);
                    }
                    Err(_) => panic!("process failed"),
                }
            }
            Rule::dnet_entry => {
                // Config the current_net to record the net staring here.
                // This part doesn't return anything, it edits the current net members.
                let _ = process_dnet_entry(entry, &mut current_net);
            }
            Rule::conn_entry => {
                // Parse the connection entry and add it to the current_net.
                // This part doesn't return anything, it adds connection to current net.
                let parse_result = process_conn_entry(entry);
                match parse_result {
                    Ok(result) => {
                        current_net.add_connection(&result);
                    }
                    Err(_) => panic!("process failed"),
                }
            }
            Rule::cap_or_res_entry => {
                // Parse the cap or res entry and add it to the current_net according to the current_section
                // This part doesn't return anything, it adds caps or ress to current net.
                let parse_result = process_cap_or_res_entry(entry, &current_section);
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
                    }
                    Err(_) => panic!("process failed"),
                }
            }

            _ => panic!("unkonwn rule {}.", entry.as_str()),
        };
    }

    let elapsed_us = measure_elapsed_time(start_time);
    println!("read spef file {} elapsed time: {} s", spef_file_path, elapsed_us);

    exchange_data
}

#[cfg(test)]
mod tests {

    use pest::iterators::Pair;
    use pest::iterators::Pairs;

    use super::*;

    fn test_process_pair(pair: Pair<Rule>) {
        // A pair is a combination of the rule which matched and a span of input
        println!("Rule:    {:?}", pair.as_rule());
        println!("Span:    {:?}", pair.as_span());
        println!("Text:    {}", pair.as_str());

        for inner_pair in pair.into_inner() {
            test_process_pair(inner_pair);
        }
    }

    fn print_parse_result(parse_result: Result<Pairs<Rule>, pest::error::Error<Rule>>) {
        let parse_result_clone = parse_result.clone();
        match parse_result {
            Ok(pairs) => {
                for pair in pairs {
                    // A pair is a combination of the rule which matched and a span of input
                    test_process_pair(pair);
                }
            }
            Err(err) => {
                // Handle parsing error
                println!("Error: {}", err);
            }
        }

        assert!(parse_result_clone.is_ok());
    }

    #[test]
    fn test_parse() {
        let input_str = r#"*D_NET in1 0.275
        *CONN
        *P in1 I
        *I r1:D I *L 0.0036
        *CAP
        1 in1 0.243
        2 r1:D 0.032
        *RES
        3 in1 r1:D 40
        *END
        "#;
        let parse_result = SpefParser::parse(Rule::spef_file, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse1() {
        let input_str = r#"1 in1 0.243
        "#;
        let parse_result = SpefParser::parse(Rule::cap_or_res_entry, input_str);

        print_parse_result(parse_result);
    }
}
