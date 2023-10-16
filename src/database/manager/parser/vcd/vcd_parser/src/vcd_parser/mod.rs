use pest::Parser;
use pest_derive::Parser;

use pest::iterators::Pair;

#[derive(Parser)]
#[grammar = "vcd_parser/grammar.pest"]
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
        let parse_result = VCDParser::parse(Rule::SCALE, input_str);

        print_parse_result(parse_result);
    }


}