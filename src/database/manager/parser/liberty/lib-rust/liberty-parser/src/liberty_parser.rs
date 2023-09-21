use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "liberty.pest"]
pub struct LibertyParser;

pub fn parse_lib_file(file: &str) -> Result<(), pest::error::Error<Rule>> {
    // Generate liberty.pest parser
    let parser = LibertyParser::parse(Rule::lib_file, file);
    match parser {
        Ok(pairs) => {
            for pair in pairs {
                // Process each pair
                match pair.as_rule() {
                    Rule::float => {
                        let dstr = pair.as_str();
                    }
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
                    Rule::punctuation => todo!(),
                    Rule::WHITESPACE => todo!(),
                    Rule::line_comment => todo!(),
                    Rule::multiline_comment => todo!(),
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
                }
            }
        }
        Err(err) => {
            // Handle parsing error
        }
    }
    // Continue with the rest of the code
    Ok(())
}
