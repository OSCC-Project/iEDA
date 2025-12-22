use pest::Parser;
use pest_derive::Parser;
use std::cell::RefCell;

use pest::iterators::Pair;

pub mod vcd_c_api;
pub mod vcd_calc_tc_sp;
pub mod vcd_data;

use std::rc::Rc;
// use std::sync::Mutex;

#[derive(Parser)]
#[grammar = "vcd_parser/grammar/grammar.pest"]
pub struct VCDParser;

/// process date.
fn process_date(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let date_text = pair.as_str().parse::<String>().unwrap();
    let vcd_file = vcd_file_parser.get_vcd_file();
    vcd_file.set_date(date_text);
}

fn process_version(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let version_text = pair.as_str().parse::<String>().unwrap();
    let vcd_file = vcd_file_parser.get_vcd_file();
    vcd_file.set_date(version_text);
}

/// process scale number.
fn process_scale_number(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let scale_number_str = pair.as_str().trim();
    let scale_number = scale_number_str.parse::<u32>().unwrap();
    let vcd_file = vcd_file_parser.get_vcd_file();
    vcd_file.set_time_resolution(scale_number);
}

/// process scale unit
fn process_scale_unit(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let scale_unit = pair.as_str().parse::<String>().unwrap();
    let vcd_file = vcd_file_parser.get_vcd_file();
    vcd_file.set_time_unit(&scale_unit);
}

/// process scale.
fn process_scale(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::scale_number => process_scale_number(inner_pair, vcd_file_parser),
            Rule::scale_unit => process_scale_unit(inner_pair, vcd_file_parser),
            _ => panic!("not process: rule {:?}", inner_pair.as_rule()),
        }
    }
}

/// process scale unit
fn process_comment(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let comment = pair.as_str().parse::<String>().unwrap();
    let vcd_file = vcd_file_parser.get_vcd_file();
    vcd_file.set_comment(comment);
}

/// process scope.
fn process_open_scope(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let mut inner_pairs = pair.into_inner().into_iter();
    let pair_type = inner_pairs.next().unwrap();
    let scope_type = pair_type.as_str();

    let pair_module = inner_pairs.next_back().unwrap();
    let module_name = pair_module.as_str();

    // println!("scope module: {}", module_name);

    let new_scope: Rc<RefCell<vcd_data::VCDScope>> = Rc::new(RefCell::new(
        vcd_data::VCDScope::new(String::from(module_name)),
    ));
    let new_scope_copy = new_scope.clone();
    new_scope.borrow_mut().set_scope_type(&scope_type);

    if !vcd_file_parser.is_scope_empty() {
        let scope_stack = vcd_file_parser.get_scope_stack();
        let parent_scope = scope_stack.back_mut().unwrap();

        new_scope
            .borrow_mut()
            .set_parent_scope(Rc::clone(&*parent_scope));

        parent_scope.borrow_mut().add_child_scope(new_scope);

        scope_stack.push_back(new_scope_copy);
    } else {
        vcd_file_parser.set_root_scope(new_scope);
    }
    
}

/// process signal variable.
fn process_variable(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let mut vcd_signal = vcd_data::VCDSignal::new();
    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::variable_type => vcd_signal.set_signal_type(inner_pair.as_str()),
            Rule::variable_number => {
                let signal_size_str = inner_pair.as_str();
                let signal_size = signal_size_str.parse::<u32>();
                vcd_signal.set_signal_size(signal_size.unwrap());
            }
            Rule::variable_ref => {
                let ref_name = inner_pair.as_str();
                vcd_signal.set_hash(String::from(ref_name));
            }
            Rule::module_var_name => {
                let module_var_name = inner_pair.as_str();
                vcd_signal.set_name(String::from(module_var_name));
            }
            Rule::bus_slice => {
                // let line_no = inner_pair.line_col().0;
                // println!("line_no: {}", line_no);
                let mut slice_pairs = inner_pair.into_inner().into_iter();

                if slice_pairs.len() == 1 {
                    let slice_pair = slice_pairs.next().unwrap();
                    let index = slice_pair.as_str().parse::<i32>().unwrap();
                    vcd_signal.set_bus_index(index, index);                    
                } else {
                    let slice_left_pair = slice_pairs.next().unwrap();
                    let slice_right_pair = slice_pairs.next_back().unwrap();

                    let left_index = slice_left_pair.as_str().parse::<i32>().unwrap();
                    let right_index = slice_right_pair.as_str().parse::<i32>().unwrap();

                    vcd_signal.set_bus_index(left_index, right_index);
                }
            }
            _ => todo!(),
        }
    }

    let scope_stack = vcd_file_parser.get_scope_stack();
    let the_scope = scope_stack.back_mut().unwrap();
    vcd_signal.set_scope(the_scope.clone());
    the_scope.borrow_mut().add_scope_signal(vcd_signal);
}

/// process close scope.
fn process_close_scope(_pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let scope_stack = vcd_file_parser.get_scope_stack();
    if !scope_stack.is_empty() {
        scope_stack.pop_back();
    }
    
}

/// process simulate time.
fn process_simulation_time(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let simu_time_str = pair.as_str().trim_start_matches('#');
    let simu_time = simu_time_str.parse::<i64>().unwrap();
    vcd_file_parser.set_current_time(simu_time);
}

/// process scalar value change.
fn process_scalar_value_change(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let mut scalar_value_change_pairs = pair.into_inner().into_iter();
    let scalar_value_pair = scalar_value_change_pairs.next().unwrap();

    let scalar_value = match scalar_value_pair.as_str() {
        "0" => vcd_data::VCDBit::BitZero,
        "1" => vcd_data::VCDBit::BitOne,
        "x" | "X" => vcd_data::VCDBit::BitX,
        "z" | "Z" => vcd_data::VCDBit::BitZ,
        _ => panic!("unkown value {}", scalar_value_pair.as_str()),
    };
    let hash_pair = scalar_value_change_pairs.next_back().unwrap();
    let hash_str = String::from(hash_pair.as_str());

    let current_time = vcd_file_parser.get_current_time();
    let vcd_value = vcd_data::VCDValue::BitScalar(scalar_value);

    let vcd_time_value = Box::new(vcd_data::VCDTimeAndValue {
        time: current_time,
        value: vcd_value,
    });

    let vcd_file = vcd_file_parser.get_vcd_file();
    vcd_file.add_signal_value(hash_str, vcd_time_value);
}

/// process vector value change.
fn process_bitvector_value_change(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let line_no = pair.line_col().0;
    // let pair_clone = pair.clone();
    // println!("Rule:    {:?}", pair_clone.as_rule());
    // println!("Span:    {:?}", pair_clone.as_span());
    // println!("Text:    {}", pair_clone.as_str());

    let mut bitvector_value_change_pairs = pair.into_inner().into_iter();
    let bitvector_value_pair = bitvector_value_change_pairs.next().unwrap();
    let bitvector_value_str = bitvector_value_pair.as_str();
    // println!("bitvector_value_str {} len {}", bitvector_value_str,bitvector_value_str.len());

    let mut vcd_bit_vec: Vec<vcd_data::VCDBit> = Vec::with_capacity(bitvector_value_str.len());
    for bit in bitvector_value_str.chars() {
        let bit_value = match bit {
            '0' => vcd_data::VCDBit::BitZero,
            '1' => vcd_data::VCDBit::BitOne,
            'x' | 'X' => vcd_data::VCDBit::BitX,
            'z' | 'Z' => vcd_data::VCDBit::BitZ,
            _ => panic!("unkown value {} in {}", bit, line_no),
        };

        vcd_bit_vec.push(bit_value);
    }

    let hash_pair = bitvector_value_change_pairs.next_back().unwrap();
    let hash_str = String::from(hash_pair.as_str());

    let current_time = vcd_file_parser.get_current_time();
    let vcd_value = vcd_data::VCDValue::BitVector(vcd_bit_vec);

    let vcd_time_value = Box::new(vcd_data::VCDTimeAndValue {
        time: current_time,
        value: vcd_value,
    });

    let vcd_file = vcd_file_parser.get_vcd_file();
    vcd_file.add_signal_value(hash_str, vcd_time_value);
}

/// process real value change.
fn process_real_value_change(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    let mut real_value_change_pairs = pair.into_inner().into_iter();

    let real_value_pair: Pair<'_, Rule> = real_value_change_pairs.next().unwrap();
    let bit_real = real_value_pair.as_str().parse::<f64>().unwrap();

    let hash_pair = real_value_change_pairs.next_back().unwrap();
    let hash_str = String::from(hash_pair.as_str());

    let current_time = vcd_file_parser.get_current_time();
    let vcd_value = vcd_data::VCDValue::BitReal(bit_real);

    let vcd_time_value = Box::new(vcd_data::VCDTimeAndValue {
        time: current_time,
        value: vcd_value,
    });

    let vcd_file = vcd_file_parser.get_vcd_file();
    vcd_file.add_signal_value(hash_str, vcd_time_value);
}

/// process vcd data.
fn process_vcd(pair: Pair<Rule>, vcd_file_parser: &mut vcd_data::VCDFileParser) {
    // let pair_clone = pair.clone();
    // println!("Rule:    {:?}", pair_clone.as_rule());
    // println!("Span:    {:?}", pair_clone.as_span());
    // println!("Text:    {}", pair_clone.as_str());

    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::date_text => process_date(inner_pair, vcd_file_parser),
            Rule::version_text => process_version(inner_pair, vcd_file_parser),
            Rule::scale_text => process_scale(inner_pair, vcd_file_parser),
            Rule::comment_text => process_comment(inner_pair, vcd_file_parser),
            Rule::open_scope => process_open_scope(inner_pair, vcd_file_parser),
            Rule::variable => process_variable(inner_pair, vcd_file_parser),
            Rule::close_scope => process_close_scope(inner_pair, vcd_file_parser),
            Rule::simulation_time => process_simulation_time(inner_pair, vcd_file_parser),
            Rule::scalar_value_change => process_scalar_value_change(inner_pair, vcd_file_parser),
            Rule::bitvector_value_change => {
                process_bitvector_value_change(inner_pair, vcd_file_parser)
            }
            Rule::real_value_change => process_real_value_change(inner_pair, vcd_file_parser),

            _ => panic!("not process: rule {:?}", inner_pair.as_rule()),
        }
    }
}

/// process vcd file data.
pub fn parse_vcd_file(vcd_file_path: &str) -> Result<vcd_data::VCDFile, pest::error::Error<Rule>> {
    let input_str = std::fs::read_to_string(vcd_file_path)
        .unwrap_or_else(|_| panic!("Can't read file: {}", vcd_file_path));
    let parse_result = VCDParser::parse(Rule::vcd_file, input_str.as_str());

    match parse_result {
        Ok(pairs) => {
            let mut vcd_file_parser = vcd_data::VCDFileParser::new();
            process_vcd(pairs.into_iter().next().unwrap(), &mut vcd_file_parser);
            Ok(vcd_file_parser.get_vcd_file_imute())
        }
        Err(err) => {
            // Handle parsing error
            println!("Error: {}", err);
            Err(err.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    // use pest::error;
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
        let parse_result = VCDParser::parse(Rule::scale_text, input_str);

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

    #[test]
    fn test_build_vcd_data() {
        let vcd_path =
            "/home/taosimin/iEDA/src/database/manager/parser/vcd/vcd_parser/benchmark/test1.vcd";
        let parse_result = parse_vcd_file(vcd_path);
        assert!(parse_result.is_ok());
    }
}
