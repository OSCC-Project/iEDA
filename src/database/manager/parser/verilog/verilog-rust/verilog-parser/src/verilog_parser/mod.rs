//mod verilog_data;

use pest::Parser;
use pest_derive::Parser;

use pest::iterators::Pair;
use pest::iterators::Pairs;

#[derive(Parser)]
#[grammar = "verilog_parser/grammar/verilog.pest"]
pub struct VerilogParser;

fn process_pair(pair: Pair<Rule>) -> Result<(), pest::error::Error<Rule>>{
    match pair.as_rule() {

        Rule::COMMENT => todo!(),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}
pub fn parse_verilog_file(verilog_file_path: &str) -> Result<(), pest::error::Error<Rule>> {
    // Generate verilog.pest parser
    let input_str = std::fs::read_to_string(verilog_file_path).unwrap_or_else(|_| panic!("Can't read file: {}", verilog_file_path));
    println!("{:#?}", input_str);
    let parse_result = VerilogParser::parse(Rule::module_declaration, input_str.as_str());

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
        let input_str = r#"output osc_25m_out_pad;
        input osc_100m_in_pad;
        output osc_100m_out_pad;"#;
        let parse_result = VerilogParser::parse(Rule::port_block_declaration, input_str);
        
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
        // println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

}