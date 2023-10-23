use pest::Parser;
use pest_derive::Parser;

use pest::iterators::Pair;

pub mod vcd_data;

#[derive(Parser)]
#[grammar = "vcd_parser/grammar/grammar.pest"]
pub struct VCDParser;

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
    fn test_parse_unit() {
        let input_str = "ns";
        let parse_result = VCDParser::parse(Rule::scale_unit, input_str);

        print_parse_result(parse_result);
    }
    #[test]
    fn test_parse_number() {
        let input_str = "10";
        let parse_result = VCDParser::parse(Rule::scale_number, input_str);

        print_parse_result(parse_result);
    }
    #[test]
    fn test_parse_scale() {
        let input_str = "10ns";
        let parse_result = VCDParser::parse(Rule::scale, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_date() {
        let input_str = r#"$date
        Tue Aug 23 16:03:49 2022
    $end"#;
        let parse_result = VCDParser::parse(Rule::date, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_scalar_value_change() {
        let input_str = r#"#0
        $dumpvars
        bxxxx #
        0!
        1"
        $end
        #50
        1!
        b0000 #
        #100
        0"
        0!
        #150
        1!
        b0001 #
        #200
        0!
        #250
        1!
        b0010 #
        #300
        0!
        #350
        1!
        b0011 #
        #400
        0!
        #450
        1!
        b0100 #
        #500
        0!"#;
        let parse_result = VCDParser::parse(Rule::value_change_section, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_scope() {
        let input_str = r#"$date
        Tue Aug 23 16:03:49 2022
    $end
    
    $timescale
        1ns
    $end
    
    $comment Csum: 1 9ba2991b94438432 $end
    
    
    $scope module test $end
    
    $scope module top_i $end
    $var wire 1 ! clk $end
    $var wire 1 " reset $end
    $var wire 4 # out [3:0] $end
    
    $scope module sub_i $end
    $var wire 1 " reset $end
    $var wire 1 ! clk $end
    $var reg 4 # out [3:0] $end
    $upscope $end
    
    $upscope $end
    
    $upscope $end
    
    $enddefinitions $end"#;
        let parse_result = VCDParser::parse(Rule::vcd_file, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_vcd_file_path() {
        let vcd_path =
            "/home/taosimin/iEDA/src/database/manager/parser/vcd/vcd_parser/benchmark/test1.vcd";

        let input_str = std::fs::read_to_string(vcd_path)
            .unwrap_or_else(|_| panic!("Can't read file: {}", vcd_path));
        let parse_result = VCDParser::parse(Rule::vcd_file, input_str.as_str());

        print_parse_result(parse_result);
    }
}
