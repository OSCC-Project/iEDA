mod liberty_data;

use pest::Parser;
use pest_derive::Parser;

use pest::iterators::Pair;

#[derive(Parser)]
#[grammar = "liberty_parser/grammar/liberty.pest"]
pub struct LibertyParser;

fn process_float(pair: Pair<Rule>) -> Result<f64, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair.into_inner().as_str().parse::<f64>() {
        Ok(value) => Ok(value),
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse float".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_pair(pair: Pair<Rule>) -> Result<(), pest::error::Error<Rule>> {
    match pair.as_rule() {
        Rule::float => todo!(),
        Rule::EOI => todo!(),
        Rule::decimal_digits => todo!(),
        Rule::decimal_integer => todo!(),
        Rule::dec_int => todo!(),
        Rule::optional_exp => todo!(),
        Rule::optional_frac => todo!(),
        Rule::bus_index => todo!(),
        Rule::bus_slice => todo!(),
        Rule::pin_id => todo!(),
        Rule::bus_id => todo!(),
        Rule::bus_bus_id => todo!(),
        Rule::lib_id => todo!(),
        Rule::WHITESPACE => todo!(),
        Rule::line_comment => todo!(),
        Rule::multiline_comment => todo!(),
        Rule::oneline_string => todo!(),
        Rule::multiline_string => todo!(),
        Rule::semicolon_opt => todo!(),
        Rule::string => todo!(),
        Rule::attribute_value => todo!(),
        Rule::attribute_values => todo!(),
        Rule::simple_attribute_value => todo!(),
        Rule::simple_attribute => todo!(),
        Rule::complex_attribute => todo!(),
        Rule::group => todo!(),
        Rule::statement => todo!(),
        Rule::statements => todo!(),
        Rule::lib_file => todo!(),
        Rule::COMMENT => todo!(),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}

pub fn parse_lib_file(lib_file_path: &str) -> Result<(), pest::error::Error<Rule>> {
    // Generate liberty.pest parser
    let input_str =
        std::fs::read_to_string(lib_file_path).unwrap_or_else(|_| panic!("Can't read file: {}", lib_file_path));
    let parse_result = LibertyParser::parse(Rule::lib_file, input_str.as_str());

    match parse_result {
        Ok(pairs) => {
            for pair in pairs {
                println!("{:?}", pair);
                // Process each pair
                process_pair(pair)?;
            }
        }
        Err(err) => {
            // Handle parsing error
            println!("Error: {}", err);
        }
    }
    // Continue with the rest of the code
    Ok(())
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
        let input_str = "A[1]";
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
        let input_str = r#""test""#;
        let parse_result = LibertyParser::parse(Rule::multiline_string, input_str);

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
