//mod verilog_data;

pub mod verilog_data;

use pest::Parser;
use pest_derive::Parser;

use pest::iterators::Pair;
use pest::iterators::Pairs;

#[derive(Parser)]
#[grammar = "verilog_parser/grammar/verilog.pest"]
pub struct VerilogParser;

fn process_module_id(pair: Pair<Rule>) -> Result<String, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    // println!("{:?}", pair);
    // println!("{:?}", pair_clone);
    match pair_clone.as_str().parse::<String>() {
        Ok(value) => Ok(value.trim_matches('"').to_string()),
        Err(_) => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse module id".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_port_or_wire_id(pair: Pair<Rule>) -> Result<Box<dyn verilog_data::VerilogVirtualBaseID>, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    // println!("{:?}", pair);
    // println!("{:?}", pair_clone);
    match pair_clone.as_rule() {
            Rule::port_or_wire_id => {
                let id = pair_clone.as_str();
                let verilog_id = verilog_data::VerilogID::new(id);
                let verilog_virtual_base_id: Box<dyn verilog_data::VerilogVirtualBaseID> = Box::new(verilog_id);
                 // Box::new(verilog_id) as Box<dyn verilog_data::VerilogVirtualBaseID>
                // println!("{:?}",verilog_virtual_base_id);
               
                Ok(verilog_virtual_base_id)

            },
            _ => todo!(),
        }

}

fn process_inner_port_declaration(pair: Pair<Rule>,dcl_type:verilog_data::DclType) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>>{
    println!("{:#?}", pair);
    let pair_clone = pair.clone();
    let pair_clone2 = pair.clone();
    let mut inner_pair = pair.into_inner();
    let port_list_or_bus_slice_pair = inner_pair.next();
    // println!("{:#?}", port_list_or_bus_slice_pair);
    match port_list_or_bus_slice_pair.clone().unwrap().as_rule() {
        Rule::bus_slice => {                   
            // let mut decimal_digits = pair.into_inner();
            let mut decimal_digits_pair = port_list_or_bus_slice_pair.unwrap().into_inner();
            let range_from = decimal_digits_pair.next().unwrap().as_str().parse::<i32>().unwrap();
            let range_to = decimal_digits_pair.next().unwrap().as_str().parse::<i32>().unwrap();
            let range = Some((range_from, range_to));
            let mut verilog_dcl_vec : Vec<Box<verilog_data::VerilogDcl>>= Vec::new();
            // inner_pair.next():get the second pair:port_list
            let port_list_pair = inner_pair.next();
            for port_pair in port_list_pair.unwrap().into_inner() {
                println!("{:#?}", port_pair);
                let dcl_name = port_pair.as_str().to_string(); 
                let verilog_dcl = verilog_data::VerilogDcl::new(0, dcl_type.clone(), &dcl_name, range);
                // let verilog_virtual_base_stmt: Box<dyn verilog_data::VerilogVirtualBaseStmt> = Box::new(verilog_dcl);
                verilog_dcl_vec.push(Box::new(verilog_dcl.clone()));
            }
            let verilog_dcls = verilog_data::VerilogDcls::new(0, verilog_dcl_vec);
            // print verilog_dcls for debug.
            println!("{:#?}", verilog_dcls);
            Ok(Box::new(verilog_dcls) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
            
        }
        Rule::port_list => {
            let range = None;
            let mut verilog_dcl_vec : Vec<Box<verilog_data::VerilogDcl>>= Vec::new();
            let port_list_pair = port_list_or_bus_slice_pair;
            for port_pair in port_list_pair.unwrap().into_inner() {
                let dcl_name = port_pair.as_str().to_string(); 
                let verilog_dcl = verilog_data::VerilogDcl::new(0, dcl_type.clone(), &dcl_name, range);
                // let verilog_virtual_base_stmt: Box<dyn verilog_data::VerilogVirtualBaseStmt> = Box::new(verilog_dcl);
                verilog_dcl_vec.push(Box::new(verilog_dcl.clone()));
            }
            let verilog_dcls = verilog_data::VerilogDcls::new(0, verilog_dcl_vec);
            println!("{:#?}", verilog_dcls);
            Ok(Box::new(verilog_dcls) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_port_declaration(pair: Pair<Rule>) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    println!("{:#?}", pair);
    println!("{:?}", pair_clone);
    match pair.as_rule() {
        Rule::input_declaration => {
            // let port_list_or_bus_slice_pair = pair_clone.into_inner().next().unwrap();
            // println!("{:#?}", port_list_or_bus_slice_pair);
            let dcl_type = verilog_data::DclType::KInput;
            let verilog_dcls = process_inner_port_declaration(pair,dcl_type);
            verilog_dcls
        }
        Rule::output_declaration => {
            // let port_list_or_bus_slice_pair = pair_clone.into_inner().next().unwrap();
            // println!("{:#?}", port_list_or_bus_slice_pair);
            let dcl_type = verilog_data::DclType::KOutput;
            let verilog_dcls = process_inner_port_declaration(pair,dcl_type);
            verilog_dcls
        }
        Rule::inout_declaration => {
            let dcl_type = verilog_data::DclType::KInout;
            let verilog_dcls = process_inner_port_declaration(pair,dcl_type);
            verilog_dcls
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_inner_wire_declaration(pair: Pair<Rule>,dcl_type:verilog_data::DclType) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>>{
    println!("{:#?}", pair);
    let pair_clone = pair.clone();
    let pair_clone2 = pair.clone();
    let mut inner_pair = pair.into_inner();
    let wire_list_or_bus_slice_pair = inner_pair.next();
    // println!("{:#?}", wire_list_or_bus_slice_pair);
    match wire_list_or_bus_slice_pair.clone().unwrap().as_rule() {
        Rule::bus_slice => {                   
            // let mut decimal_digits = pair.into_inner();
            let mut decimal_digits_pair = wire_list_or_bus_slice_pair.unwrap().into_inner();
            let range_from = decimal_digits_pair.next().unwrap().as_str().parse::<i32>().unwrap();
            let range_to = decimal_digits_pair.next().unwrap().as_str().parse::<i32>().unwrap();
            let range = Some((range_from, range_to));
            let mut verilog_dcl_vec : Vec<Box<verilog_data::VerilogDcl>>= Vec::new();
            // inner_pair.next():get the second pair:wire_list
            let wire_list_pair = inner_pair.next();
            for wire_pair in wire_list_pair.unwrap().into_inner() {
                println!("{:#?}", wire_pair);
                let dcl_name = wire_pair.as_str().to_string(); 
                let verilog_dcl = verilog_data::VerilogDcl::new(0, dcl_type.clone(), &dcl_name, range);
                // let verilog_virtual_base_stmt: Box<dyn verilog_data::VerilogVirtualBaseStmt> = Box::new(verilog_dcl);
                verilog_dcl_vec.push(Box::new(verilog_dcl.clone()));
            }
            let verilog_dcls = verilog_data::VerilogDcls::new(0, verilog_dcl_vec);
            // print verilog_dcls for debug.
            println!("{:#?}", verilog_dcls);
            Ok(Box::new(verilog_dcls) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
            
        }
        Rule::wire_list => {
            let range = None;
            let mut verilog_dcl_vec : Vec<Box<verilog_data::VerilogDcl>>= Vec::new();
            let wire_list_pair = wire_list_or_bus_slice_pair;
            for wire_pair in wire_list_pair.unwrap().into_inner() {
                let dcl_name = wire_pair.as_str().to_string(); 
                let verilog_dcl = verilog_data::VerilogDcl::new(0, dcl_type.clone(), &dcl_name, range);
                // let verilog_virtual_base_stmt: Box<dyn verilog_data::VerilogVirtualBaseStmt> = Box::new(verilog_dcl);
                verilog_dcl_vec.push(Box::new(verilog_dcl.clone()));
            }
            let verilog_dcls = verilog_data::VerilogDcls::new(0, verilog_dcl_vec);
            println!("{:#?}", verilog_dcls);
            Ok(Box::new(verilog_dcls) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_wire_declaration(pair: Pair<Rule>) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    println!("{:#?}", pair);
    println!("{:?}", pair_clone);
    match pair.as_rule() {
        Rule::wire_declaration => {
            // let port_list_or_bus_slice_pair = pair_clone.into_inner().next().unwrap();
            // println!("{:#?}", port_list_or_bus_slice_pair);
            let dcl_type = verilog_data::DclType::KWire;
            let verilog_dcls = process_inner_wire_declaration(pair,dcl_type);
            verilog_dcls
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

// fn process_pair(pair: Pair<Rule>) -> Result<(), pest::error::Error<Rule>>{
//     match pair_clone.as_rule() {
//                     // record line_no,file_name
//         Rule::module_id => process_module_id(pair_clone),
//         Rule::port_list => process_port_list(pair_clone),
//         Rule::port_block_declaration => process_port_block_declaration(pair_clone),
//         Rule::wire_block_declaration => process_wire_block_declaration(pair_clone, &mut substitute_queue),
//         Rule::inst_block_declaration => process_inst_block_declaration(pair_clone, &mut substitute_queue),
//         _ => Err(pest::error::Error::new_from_span(
//         pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
//         pair_clone.as_span(),
// )),}
    
//     match pair.as_rule() {

//         Rule::COMMENT => todo!(),
//         _ => Err(pest::error::Error::new_from_span(
//             pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
//             pair.as_span(),
//         )),
//     }
// }
pub fn parse_verilog_file(verilog_file_path: &str) -> Result<verilog_data::VerilogModule, pest::error::Error<Rule>> {
    // Generate verilog.pest parser
    let input_str = std::fs::read_to_string(verilog_file_path).unwrap_or_else(|_| panic!("Can't read file: {}", verilog_file_path));
    // println!("{:#?}", input_str);
    let parse_result = VerilogParser::parse(Rule::verilog_file, input_str.as_str());
    // println!("{:#?}", parse_result);

    // add the module_id,port_list,port_block_declaration,wire_block_declaration,inst_block_declaration datastructure to store the processed data.
    let file_name = "tbd";
    let line_no = 0;
    let mut verilog_module = verilog_data::VerilogModule::new(1, "MyModule", vec![], vec![]);
    let mut module_name;
    let mut port_list: Vec<Box<dyn verilog_data::VerilogVirtualBaseID>> = Vec::new();
    // first verilog_dcl push to verilog_dcls, then verilog_dcls push to module_stmts
    // let mut verilog_dcls: Vec<Box<dyn verilog_data::VerilogVirtualBaseStmt>> = Vec::new();
    let mut module_stmts: Vec<Box <dyn verilog_data::VerilogVirtualBaseStmt>> = Vec::new();

    match parse_result {
        Ok(pairs) => {
            // pairs:module_declaration+
            for pair in pairs {
                // println!("{:?}", pair);
                // Process each pair
                let mut inner_pairs = pair.into_inner();
                for inner_pair in inner_pairs {
                    // the way similar to tao
                    // let pair_result = process_pair(inner_pair, parser_queue);
                    // parser_queue.push_back(pair_result.unwrap());

                    println!("{:#?}", inner_pair);
                    match inner_pair.as_rule() {
        
                        // record line_no,file_name
                        Rule::module_id => {
                            module_name = process_module_id(inner_pair).unwrap();
                        }
                        Rule::port_list => {
                            // process_port_list(inner_pair);
                            for inner_inner_pair in inner_pair.into_inner() {
                                let  port_id = process_port_or_wire_id(inner_inner_pair).unwrap();
                                // println!("{:#?}", port_id);
                                port_list.push(port_id);
                            }
                        }
                        Rule::port_block_declaration => {
                            for inner_inner_pair in inner_pair.into_inner() {
                                // println!("{:#?}", inner_inner_pair);
                                let verilog_dcls =  process_port_declaration(inner_inner_pair).unwrap();
                                // at the positon, only print trait debug.
                                // println!("{:#?}", verilog_dcls);
                                module_stmts.push(verilog_dcls);
                            }
                        }
                        Rule::wire_block_declaration => {
                            for inner_inner_pair in inner_pair.into_inner() {
                                println!("{:#?}", inner_inner_pair);
                                let verilog_dcls =  process_wire_declaration(inner_inner_pair).unwrap();
                                // at the positon, only print trait debug.
                                println!("{:#?}", verilog_dcls);
                                module_stmts.push(verilog_dcls);
                            }
                            let _a = 0;
                        }
                        Rule::inst_block_declaration => unreachable!(),
                        // other rule: no clone the pair
                        // _ => Err(pest::error::Error::new_from_span(
                        // pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                        // inner_pair.as_span(),
                        // )),
                        Rule::EOI => (),
                        _ => unreachable!(),
                    }
                }
            }
        }
        Err(err) => {
            // Handle parsing error
            println!("Error: {}", err);
        }
    }

    // store the verilogModule.

    // delete later
    Ok(verilog_module)
    //Err(pest::error::Error::new_from_span(pest::error::ErrorVariant::CustomError { message: "Unknown rule" }, span))
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
        let parse_result = VerilogParser::parse(Rule::COMMENT, input_str);
        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_port_or_wire_id() {
        let input_str = "rid_nic400_axi4_ps2_1_";
        let parse_result = VerilogParser::parse(Rule::port_or_wire_id, input_str);

        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_port_list() {
        let input_str = "in1, in2, clk1, clk2, clk3, out";
        let parse_result = VerilogParser::parse(Rule::port_list, input_str);
        println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_port_or_wire_id1() {
        let input_str = "\\clk_cfg[6]";
        let parse_result = VerilogParser::parse(Rule::port_or_wire_id, input_str);

        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_input_declaration() {
        let input_str = "input chiplink_rx_clk_pad;";
        let parse_result = VerilogParser::parse(Rule::input_declaration, input_str);

        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_input_declaration1() {
        let input_str = "input [1:0] din;";
        let parse_result = VerilogParser::parse(Rule::input_declaration, input_str);
        println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_port_block_declaration() {
        let input_str = r#"output [3:0] osc_25m_out_pad;
        input osc_100m_in_pad;
        output osc_100m_out_pad;"#;
        let parse_result = VerilogParser::parse(Rule::port_block_declaration, input_str);
        println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_wire_declaration() {
        let _input_str = "wire \\vga_b[0] ;";
        let input_str1 = "wire ps2_dat;";
        let parse_result = VerilogParser::parse(Rule::wire_declaration, input_str1);

        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_wire_block_declaration() {
        let input_str = r#"wire \core_dip[2] ;
        wire \core_dip[1] ;
        wire \core_dip[0] ;
        wire \u0_rcg/u0_pll_bp ;
        wire \u0_rcg/u0_pll_fbdiv_5_ ;
        wire \u0_rcg/u0_pll_postdiv2_1_ ;
        wire \u0_rcg/u0_pll_clk ;"#;
        let parse_result = VerilogParser::parse(Rule::wire_block_declaration, input_str);

        print_parse_result(parse_result);
    }


    #[test] 
    fn test_parse_first_port_connection() {
        let input_str = r#".I(\u0_soc_top/u0_ysyx_210539/writeback_io_excep_en )"#;
        let parse_result = VerilogParser::parse(Rule::first_port_connection, input_str);

        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_port_connection() {
        let input_str = r#",
        .Z(hold_net_52144)"#;
        let parse_result = VerilogParser::parse(Rule::port_connection, input_str);

        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_port_block_connection() {
        let input_str = r#"(.I(\u0_soc_top/u0_ysyx_210539/writeback_io_excep_en ),
        .Z(hold_net_52144));"#;
        let parse_result = VerilogParser::parse(Rule::port_block_connection, input_str);

        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_inst_declaration() {
        let input_str = r#"BUFFD1BWP30P140 hold_buf_52144 (.I(\u0_soc_top/u0_ysyx_210539/writeback_io_excep_en ),
        .Z(hold_net_52144));"#;
        let parse_result = VerilogParser::parse(Rule::inst_declaration, input_str);

        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_module_id() {
        let input_str = "soc_top_0";
        let parse_result = VerilogParser::parse(Rule::module_id, input_str);
        println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_inst_block_declaration() {
        let input_str = r#"DEL150MD1BWP40P140HVT hold_buf_52163 (.I(\u0_soc_top/u0_ysyx_210539/csrs/n3692 ),
        .Z(hold_net_52163));
        DEL150MD1BWP40P140HVT hold_buf_52164 (.I(\u0_soc_top/u0_ysyx_210539/icache/Ram_bw_3_io_wdata[123] ),
        .Z(hold_net_52164));"#;
        let parse_result = VerilogParser::parse(Rule::inst_block_declaration, input_str);
        println!("{:#?}",parse_result);
        // print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_module_declaration() {
        let input_str = r#"module preg_w4_reset_val0_0 (
            clock,
            reset,
            din,
            dout,
            wen);
        input clock;
        input reset;
        input [3:0] din;
        output [3:0] dout;
        input wen;
        
        // Internal wires
        wire n4;
        wire n1;
        
        DFQD1BWP40P140 data_reg_1_ (.CP(clock),
            .D(n4),
            .Q(dout[1]));
        MUX2NUD1BWP40P140 U3 (.I0(dout[1]),
            .I1(din[1]),
            .S(wen),
            .ZN(n1));
        NR2D1BWP40P140 U4 (.A1(reset),
            .A2(n1),
            .ZN(n4));
        endmodule"#;
        let parse_result = VerilogParser::parse(Rule::module_declaration, input_str);
        println!("{:#?}",parse_result);
        // print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_verilog_file1() {
        let verilog_file_path  = "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/asic_top_flatten.v";
        let input_str =
        std::fs::read_to_string(verilog_file_path).unwrap_or_else(|_| panic!("Can't read file: {}", verilog_file_path));
       
        let parse_result = VerilogParser::parse(Rule::verilog_file, input_str.as_str());
        // println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_verilog_file2() {
        let verilog_file_path  = "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/example1.v";
        let input_str =
        std::fs::read_to_string(verilog_file_path).unwrap_or_else(|_| panic!("Can't read file: {}", verilog_file_path));
       
        let parse_result = VerilogParser::parse(Rule::verilog_file, input_str.as_str());
        // println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_wire_list_with_scalar_constant() {
        let _input_str  = r#"1'b0, 1'b0, rid_nic400_axi4_ps2_1_, 1'b0"#;
        let input_str1 = r#"\u0_rcg/n33 ,  \u0_rcg/n33 ,  \u0_rcg/n33"#;

        let parse_result = VerilogParser::parse(Rule::wire_list_with_scalar_constant, input_str1);
        println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

    #[test] 
    fn test_parse_verilog_file3() {
        let verilog_file_path  = "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/asic_top_DC_downsize.v";
        let input_str =
        std::fs::read_to_string(verilog_file_path).unwrap_or_else(|_| panic!("Can't read file: {}", verilog_file_path));
       
        let parse_result = VerilogParser::parse(Rule::verilog_file, input_str.as_str());
        println!("{:#?}",parse_result);
        // print_parse_result(parse_result);
    }

}