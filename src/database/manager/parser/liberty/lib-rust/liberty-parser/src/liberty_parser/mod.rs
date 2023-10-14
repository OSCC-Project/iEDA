pub mod liberty_c_api;
pub mod liberty_data;

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use std::collections::VecDeque;

use std::ffi::c_void;
use std::os::raw::c_char;

#[derive(Parser)]
#[grammar = "liberty_parser/grammar/liberty.pest"]
pub struct LibertyParser;

/// process float data.
fn process_float(pair: Pair<Rule>) -> Result<liberty_data::LibertyParserData, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair.as_str().parse::<f64>() {
        Ok(value) => Ok(liberty_data::LibertyParserData::Float(liberty_data::LibertyFloatValue { value })),
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse float".into() },
            pair_clone.as_span(),
        )),
    }
}

/// process string data.
fn process_string(pair: Pair<Rule>) -> Result<liberty_data::LibertyParserData, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();

    // println!("Rule:    {:?}", pair_clone.as_rule());
    // println!("Span:    {:?}", pair_clone.as_span());
    // println!("Text:    {}", pair_clone.as_str());

    match pair.as_str().parse::<String>() {
        Ok(value) => Ok(liberty_data::LibertyParserData::String(liberty_data::LibertyStringValue {
            value: value.trim_matches('"').to_string(),
        })),
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse float".into() },
            pair_clone.as_span(),
        )),
    }
}

/// process simple attribute
fn process_simple_attribute(
    pair: Pair<Rule>,
    parser_queue: &mut VecDeque<liberty_data::LibertyParserData>,
) -> Result<liberty_data::LibertyParserData, pest::error::Error<Rule>> {
    let file_name = "tbd";
    // let line_no = pair.as_span().start_pos().line_col().0;
    let line_no = pair.line_col().0;
    if let liberty_data::LibertyParserData::String(liberty_string_value) = parser_queue.pop_front().unwrap() {
        let lib_id = &liberty_string_value.value;
        let attribute_value = parser_queue.pop_front().unwrap();
        match attribute_value {
            liberty_data::LibertyParserData::String(s) => {
                let simple_stmt = liberty_data::LibertySimpleAttrStmt::new(
                    file_name,
                    line_no,
                    lib_id,
                    Box::new(s) as Box<dyn liberty_data::LibertyAttrValue>,
                );
                Ok(liberty_data::LibertyParserData::SimpleStmt(simple_stmt))
            }
            liberty_data::LibertyParserData::Float(f) => {
                let simple_stmt = liberty_data::LibertySimpleAttrStmt::new(
                    file_name,
                    line_no,
                    lib_id,
                    Box::new(f) as Box<dyn liberty_data::LibertyAttrValue>,
                );
                Ok(liberty_data::LibertyParserData::SimpleStmt(simple_stmt))
            }
            _ => Err(pest::error::Error::new_from_span(
                pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                pair.as_span(),
            )),
        }
    } else {
        Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        ))
    }
}

/// process complex attribute
fn process_complex_attribute(
    pair: Pair<Rule>,
    parser_queue: &mut VecDeque<liberty_data::LibertyParserData>,
) -> Result<liberty_data::LibertyParserData, pest::error::Error<Rule>> {
    let file_name = "tbd";
    let line_no = pair.line_col().0;
    let mut attri_values: Vec<Box<dyn liberty_data::LibertyAttrValue>> = Vec::new();
    if let liberty_data::LibertyParserData::String(liberty_string_value) = parser_queue.pop_front().unwrap() {
        let lib_id = &liberty_string_value.value;
        for _ in 0..parser_queue.len() {
            let attribute_value = parser_queue.pop_front().unwrap();
            match attribute_value {
                liberty_data::LibertyParserData::String(s) => attri_values.push(Box::new(s)),
                liberty_data::LibertyParserData::Float(f) => attri_values.push(Box::new(f)),
                _ => todo!(),
            }
        }
        let complex_stmt = liberty_data::LibertyComplexAttrStmt::new(file_name, line_no, lib_id, attri_values);
        Ok(liberty_data::LibertyParserData::ComplexStmt(complex_stmt))
    } else {
        Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        ))
    }
}

fn process_group_attribute(
    pair: Pair<Rule>,
    parser_queue: &mut VecDeque<liberty_data::LibertyParserData>,
) -> Result<liberty_data::LibertyParserData, pest::error::Error<Rule>> {
    let file_name = "tbd";
    let line_no = pair.line_col().0;
    let mut attri_values: Vec<Box<dyn liberty_data::LibertyAttrValue>> = Vec::new();
    let mut stmts: Vec<Box<dyn liberty_data::LibertyStmt>> = Vec::new();
    if let liberty_data::LibertyParserData::String(liberty_string_value) = parser_queue.pop_front().unwrap() {
        let lib_id = &liberty_string_value.value;
        for _ in 0..parser_queue.len() {
            let attribute_value_or_stmt = parser_queue.pop_front().unwrap();
            match attribute_value_or_stmt {
                liberty_data::LibertyParserData::String(s) => attri_values.push(Box::new(s)),
                liberty_data::LibertyParserData::Float(f) => attri_values.push(Box::new(f)),
                liberty_data::LibertyParserData::SimpleStmt(simple_stmt) => stmts.push(Box::new(simple_stmt)),
                liberty_data::LibertyParserData::ComplexStmt(complex_stmt) => stmts.push(Box::new(complex_stmt)),
                liberty_data::LibertyParserData::GroupStmt(group_stmt) => stmts.push(Box::new(group_stmt)),
                _ => todo!(),
            }
        }
        let group_stmt = liberty_data::LibertyGroupStmt::new(file_name, line_no, lib_id, attri_values, stmts);
        Ok(liberty_data::LibertyParserData::GroupStmt(group_stmt))
    } else {
        Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        ))
    }
}

/// process pest pair data.
fn process_pair(
    pair: Pair<Rule>,
    parser_queue: &mut VecDeque<liberty_data::LibertyParserData>,
) -> Result<liberty_data::LibertyParserData, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    let current_queue_len = parser_queue.len();
    for inner_pair in pair.into_inner() {
        let pair_result = process_pair(inner_pair, parser_queue);
        parser_queue.push_back(pair_result.unwrap());
    }

    // println!("Rule:    {:?}", pair_clone.as_rule());
    // println!("Span:    {:?}", pair_clone.as_span());
    // println!("Text:    {}", pair_clone.as_str());

    let mut substitute_queue: VecDeque<liberty_data::LibertyParserData> = VecDeque::new();
    while parser_queue.len() > current_queue_len {
        substitute_queue.push_front(parser_queue.pop_back().unwrap());
    }

    match pair_clone.as_rule() {
        Rule::float => process_float(pair_clone),
        Rule::multiline_string => process_string(pair_clone),
        Rule::id => process_string(pair_clone),
        Rule::simple_attribute => process_simple_attribute(pair_clone, &mut substitute_queue),
        Rule::complex_attribute => process_complex_attribute(pair_clone, &mut substitute_queue),
        Rule::group => process_group_attribute(pair_clone, &mut substitute_queue),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

pub fn parse_lib_file(lib_file_path: &str) -> Result<liberty_data::LibertyParserData, pest::error::Error<Rule>> {
    // Generate liberty.pest parser
    let input_str =
        std::fs::read_to_string(lib_file_path).unwrap_or_else(|_| panic!("Can't read file: {}", lib_file_path));
    let parse_result = LibertyParser::parse(Rule::lib_file, input_str.as_str());
    let mut parser_queue: VecDeque<liberty_data::LibertyParserData> = VecDeque::new();

    match parse_result {
        Ok(pairs) => {
            let lib_data = process_pair(pairs.into_iter().next().unwrap(), &mut parser_queue);
            assert_eq!(parser_queue.len(), 0);
            lib_data
        }
        Err(err) => {
            // Handle parsing error
            println!("Error: {}", err);
            Err(err.clone())
        }
    }
}

#[no_mangle]
pub extern "C" fn rust_parse_lib(s: *const c_char) -> *mut c_void {
    let c_str = unsafe { std::ffi::CStr::from_ptr(s) };
    let r_str = c_str.to_string_lossy().into_owned();
    println!("r str {}", r_str);

    let lib_file = parse_lib_file(&r_str);
    if let liberty_data::LibertyParserData::GroupStmt(lib_file_group) = lib_file.unwrap() {
        let lib_file_pointer = Box::new(lib_file_group);

        let raw_pointer = Box::into_raw(lib_file_pointer);
        raw_pointer as *mut c_void
    } else {
        panic!("error type");
    }
}

#[cfg(test)]
mod tests {
    use pest::error;
    use pest::iterators::Pair;
    use pest::iterators::Pairs;

    use super::*;

    fn process_pair(pair: Pair<Rule>) {
        // A pair is a combination of the rule which matched and a span of input
        println!("Rule:    {:?}", pair.as_rule());
        println!("Span:    {:?}", pair.as_span());
        println!("Text:    {}", pair.as_str());

        for inner_pair in pair.into_inner() {
            process_pair(inner_pair);
        }
    }

    fn print_parse_result(parse_result: Result<Pairs<Rule>, pest::error::Error<Rule>>) {
        let parse_result_clone = parse_result.clone();
        match parse_result {
            Ok(pairs) => {
                for pair in pairs {
                    // A pair is a combination of the rule which matched and a span of input
                    process_pair(pair);
                }
            }
            Err(err) => {
                // Handle parsing error
                println!("Error: {}", err);
            }
        }

        assert!(!parse_result_clone.is_err());
    }

    #[test]
    fn test_parse_comment() {
        let input_str = "/*test 
        test
        */";
        let parse_result = LibertyParser::parse(Rule::COMMENT, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_float() {
        let input_str = "1.774000e-01";
        let parse_result = LibertyParser::parse(Rule::float, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_lib_id() {
        let input_str = "A";
        let parse_result = LibertyParser::parse(Rule::lib_id, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_bus_id() {
        let input_str = "A[ 1 ]";
        let parse_result = LibertyParser::parse(Rule::bus_id, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_bus_id1() {
        let input_str = "A[1:2]";
        let parse_result = LibertyParser::parse(Rule::bus_id, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_bus_bus_id() {
        let input_str = "A[1][1:2]";
        let parse_result = LibertyParser::parse(Rule::bus_bus_id, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_line_comment() {
        let input_str = "//test";
        let parse_result = LibertyParser::parse(Rule::line_comment, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_multiline_string() {
        let input_str = r#""test",\
        "test""#;
        let parse_result = LibertyParser::parse(Rule::multiline_string, input_str);

        // let tokens = parse_result.unwrap().tokens();

        // for token in tokens {
        //     println!("token {:?}", token);
        // }
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_simple_attribute() {
        let input_str = r#"process       	: 1.01;"#;
        let parse_result = LibertyParser::parse(Rule::simple_attribute, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_complex_attribute() {
        let input_str = r"define(process_corner, operating_conditions, string);";
        let parse_result = LibertyParser::parse(Rule::complex_attribute, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_group_attribute() {
        let input_str = r#"operating_conditions (slow) {
            process_corner	: "SlowSlow";
            process       	: 1.00;
            voltage       	: 0.95;
            temperature   	: 125.00;
            tree_type     	: balanced_tree;
          }"#;
        let parse_result = LibertyParser::parse(Rule::group, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_lib_file() {
        let input_str = r#"   library (NangateOpenCellLibrary_slow) {

            /* Documentation Attributes */
            date                    		: "Thu 10 Feb 2011, 18:11:58";
            revision                		: "revision 1.0";
            comment                 		: "Copyright (c) 2004-2011 Nangate Inc. All Rights Reserved.";
        }"#;
        let parse_result = LibertyParser::parse(Rule::lib_file, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_lib_file_path() {
        let lib_file_path =
            "/home/taosimin/iEDA/src/database/manager/parser/liberty/lib-rust/liberty-parser/example/example1_slow.lib";

        let input_str =
            std::fs::read_to_string(lib_file_path).unwrap_or_else(|_| panic!("Can't read file: {}", lib_file_path));
        let parse_result = LibertyParser::parse(Rule::lib_file, input_str.as_str());

        print_parse_result(parse_result);
    }
}
